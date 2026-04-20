#!/usr/bin/env python
"""
N8N Container Launcher

This script starts the n8n Docker container with the current Cloudflare tunnel URL
from the MCP API. It ensures that n8n is properly configured to work with MCP.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Try to import the tunnel tracker for direct access
try:
    from scripts.cloudflare_tunnel_tracker import get_tunnel_url
    DIRECT_ACCESS = True
except ImportError:
    print("Warning: Cannot import tunnel tracker directly, will attempt API fallback")
    DIRECT_ACCESS = False


def get_tunnel_url_from_api(api_url: str = "http://localhost:8000") -> Optional[str]:
    """
    Get the Cloudflare tunnel URL from the MCP API.
    
    Args:
        api_url: Base URL for the MCP API
        
    Returns:
        The tunnel URL or None if not available
    """
    try:
        response = requests.get(f"{api_url}/tunnel-url", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("url")
        else:
            print(f"Error getting tunnel URL from API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to MCP API: {e}")
        return None


def get_current_tunnel_url() -> str:
    """
    Get the current Cloudflare tunnel URL using the most reliable method available.
    
    Returns:
        The tunnel URL or a default value
    """
    # Try direct access first
    if DIRECT_ACCESS:
        try:
            url = get_tunnel_url()
            if url:
                return url
        except Exception as e:
            print(f"Error getting tunnel URL directly: {e}")
    
    # Fallback to API
    url = get_tunnel_url_from_api()
    if url:
        return url
    
    # Default fallback
    print("Warning: Could not retrieve tunnel URL, using default")
    return "https://example.trycloudflare.com"


def start_n8n_container(webhook_url: str, n8n_volume_path: str, port: int = 5678) -> None:
    """
    Start the n8n Docker container with the specified webhook URL.
    
    Args:
        webhook_url: The Cloudflare tunnel URL to use for webhooks
        n8n_volume_path: Path to the n8n data volume
        port: Port to expose n8n on
    """
    # Check if webhook URL is available
    if not webhook_url or webhook_url == "https://example.trycloudflare.com":
        print("Warning: Using default webhook URL. Webhooks may not work correctly.")
    
    # Normalize volume path
    volume_path = os.path.abspath(os.path.expanduser(n8n_volume_path))
    if not os.path.exists(volume_path):
        os.makedirs(volume_path, exist_ok=True)
        print(f"Created n8n data directory: {volume_path}")
    
    # Command to start n8n container
    if sys.platform == "win32":
        # Windows version using backticks for multiline
        cmd = [
            "docker", "run", "--rm",
            "--name", "n8n-docker",
            "-p", f"{port}:{port}",
            "-v", f"{volume_path}:/home/node/.n8n",
            "-e", f"WEBHOOK_URL={webhook_url}",
            "n8nio/n8n:latest"
        ]
    else:
        # Linux/macOS version
        cmd = [
            "docker", "run", "--rm",
            "--name", "n8n-docker",
            "-p", f"{port}:{port}",
            "-v", f"{volume_path}:/home/node/.n8n",
            "-e", f"WEBHOOK_URL={webhook_url}",
            "n8nio/n8n:latest"
        ]
    
    print(f"Starting n8n container with webhook URL: {webhook_url}")
    print(f"n8n will be available at: http://localhost:{port}")
    
    # Execute the docker run command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping n8n container...")
        subprocess.run(["docker", "stop", "n8n-docker"])
    except Exception as e:
        print(f"Error starting n8n container: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start n8n container with MCP integration")
    parser.add_argument("--volume", default="E:/Ubuntu_AGI/n8n",
                        help="Path to n8n data volume (default: E:/Ubuntu_AGI/n8n)")
    parser.add_argument("--port", type=int, default=5678,
                        help="Port to expose n8n on (default: 5678)")
    parser.add_argument("--webhook-url", 
                        help="Explicitly set webhook URL (default: auto-detect from MCP)")
    
    args = parser.parse_args()
    
    # Get webhook URL from arguments or auto-detect
    webhook_url = args.webhook_url or get_current_tunnel_url()
    
    # Start n8n container
    start_n8n_container(webhook_url, args.volume, args.port) 