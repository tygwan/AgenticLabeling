#!/usr/bin/env python
"""
Cloudflare Tunnel Tracker

This script monitors a Cloudflare tunnel process, extracts the dynamically generated
tunnel URL, and makes it available to other services like n8n and MCP.
"""

import os
import re
import json
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# Configuration
CONFIG_DIR = Path(__file__).resolve().parent.parent / "mcp" / "config"
TUNNEL_URL_FILE = CONFIG_DIR / "cloudflare_tunnel.json"
DEFAULT_WEBHOOK_URL = "https://example.trycloudflare.com"  # Fallback URL

# Ensure config directory exists
CONFIG_DIR.mkdir(exist_ok=True)


class CloudflareTunnelTracker:
    """Tracks and manages a Cloudflare tunnel process."""
    
    def __init__(self, port: int = 8000):
        """
        Initialize the tunnel tracker.
        
        Args:
            port: Local port to forward through the tunnel
        """
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Load previously saved URL if available
        self._load_saved_url()
    
    def _load_saved_url(self) -> None:
        """Load the previously saved tunnel URL from file if it exists."""
        if TUNNEL_URL_FILE.exists():
            try:
                with open(TUNNEL_URL_FILE, 'r') as f:
                    data = json.load(f)
                    self.url = data.get('url')
                    print(f"Loaded saved tunnel URL: {self.url}")
            except Exception as e:
                print(f"Error loading saved tunnel URL: {e}")
    
    def _save_url(self) -> None:
        """Save the current tunnel URL to file."""
        if self.url:
            try:
                with open(TUNNEL_URL_FILE, 'w') as f:
                    json.dump({'url': self.url, 'timestamp': time.time()}, f, indent=2)
                print(f"Saved tunnel URL to {TUNNEL_URL_FILE}")
            except Exception as e:
                print(f"Error saving tunnel URL: {e}")
    
    def start(self, callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Start the Cloudflare tunnel.
        
        Args:
            callback: Optional function to call when the URL is detected
            
        Returns:
            The tunnel URL (may be None initially but updated later)
        """
        if self.running:
            return self.url or DEFAULT_WEBHOOK_URL
        
        self.running = True
        
        # Start the cloudflared process
        cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{self.port}"]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Start thread to monitor for the URL
        self.monitor_thread = threading.Thread(
            target=self._monitor_output,
            args=(callback,),
            daemon=True
        )
        self.monitor_thread.start()
        
        # Return the URL (might be from a previous run)
        return self.url or DEFAULT_WEBHOOK_URL
    
    def _monitor_output(self, callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Monitor the cloudflared process output for the tunnel URL.
        
        Args:
            callback: Function to call when the URL is detected
        """
        # Regular expression to match the tunnel URL
        url_pattern = re.compile(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com')
        
        assert self.process is not None, "Process must be initialized before monitoring"
        
        while self.running and self.process.poll() is None:
            # Read from stderr as cloudflared logs most info there
            line = self.process.stderr.readline()
            if not line:
                continue
            
            print(f"Cloudflared: {line.strip()}")
            
            # Check for tunnel URL in the output
            matches = url_pattern.findall(line)
            if matches:
                self.url = matches[0]
                print(f"Detected Cloudflare tunnel URL: {self.url}")
                
                # Save the URL to file
                self._save_url()
                
                # Call the callback with the URL if provided
                if callback:
                    callback(self.url)
    
    def get_url(self) -> str:
        """Get the current tunnel URL or default if not available."""
        return self.url or DEFAULT_WEBHOOK_URL
    
    def stop(self) -> None:
        """Stop the tunnel process if running."""
        self.running = False
        
        if self.process and self.process.poll() is None:
            # Try to gracefully terminate
            self.process.terminate()
            try:
                # Wait up to 5 seconds for process to terminate
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                self.process.kill()
            
            print("Cloudflare tunnel stopped")


# Singleton instance
_tracker_instance = None

def get_tracker() -> CloudflareTunnelTracker:
    """Get the singleton tunnel tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CloudflareTunnelTracker()
    return _tracker_instance


def start_tunnel(port: int = 8000, callback: Optional[Callable[[str], None]] = None) -> str:
    """
    Start a cloudflare tunnel for the given port.
    
    Args:
        port: The local port to expose
        callback: Optional function to call when the URL is detected
        
    Returns:
        The tunnel URL (may be from a previous run initially)
    """
    tracker = get_tracker()
    return tracker.start(callback)


def get_tunnel_url() -> str:
    """
    Get the current cloudflare tunnel URL.
    
    Returns:
        The tunnel URL or a default fallback
    """
    tracker = get_tracker()
    return tracker.get_url()


def stop_tunnel() -> None:
    """Stop the cloudflare tunnel if running."""
    tracker = get_tracker()
    tracker.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Cloudflare tunnel for MCP integration")
    parser.add_argument("--port", type=int, default=8000, help="Local port to expose (default: 8000)")
    parser.add_argument("--action", choices=["start", "stop", "url"], default="start", 
                        help="Action to perform (default: start)")
    
    args = parser.parse_args()
    
    try:
        if args.action == "start":
            url = start_tunnel(args.port)
            print(f"Starting Cloudflare tunnel for localhost:{args.port}")
            print(f"Initial URL (may change): {url}")
            
            # Keep the script running to maintain the tunnel
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down tunnel...")
                stop_tunnel()
                
        elif args.action == "stop":
            stop_tunnel()
            print("Cloudflare tunnel stopped")
            
        elif args.action == "url":
            url = get_tunnel_url()
            print(f"Current Cloudflare tunnel URL: {url}")
            
    except Exception as e:
        print(f"Error: {e}") 