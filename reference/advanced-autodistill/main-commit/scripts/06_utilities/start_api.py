#!/usr/bin/env python
"""
MCP API Server Launcher

This script starts the FastAPI server for the Model Control Panel (MCP)
and initializes the Cloudflare tunnel for external access.
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# 프로젝트 루트를 명시적으로 설정
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# MCP src 디렉토리와 기타 필요한 디렉토리들 추가
sys.path.append(str(project_root / "mcp"))
sys.path.append(str(project_root / "mcp" / "src"))
sys.path.append(str(project_root / "mcp" / "src" / "routers"))
sys.path.append(str(project_root / "mcp" / "src" / "models"))
sys.path.append(str(project_root / "mcp" / "src" / "utils"))

# 필요한 디렉토리 생성
os.makedirs(project_root / "mcp" / "categories", exist_ok=True)
os.makedirs(project_root / "mcp" / "config", exist_ok=True)

# 종속성 확인
required_packages = [
    "scikit-learn",
    "requests",
    "fastapi",
    "uvicorn"
]

def check_and_install_dependencies():
    """Check for required dependencies and install them if missing."""
    import importlib.util
    import subprocess
    
    missing_packages = []
    for package in required_packages:
        # Check if package is installed using importlib
        try:
            if package == "scikit-learn":
                # scikit-learn is imported as sklearn
                check_package = "sklearn"
            else:
                check_package = package.split("[")[0]  # Handle packages like "fastapi[all]"
            
            spec = importlib.util.find_spec(check_package)
            if spec is None:
                missing_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    # Install missing packages if any
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Dependencies installed successfully")


def start_cloudflare_tunnel():
    """
    Start the Cloudflare tunnel in a separate process.
    """
    try:
        # Check if cloudflared is installed
        subprocess.run(["cloudflared", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Start the tunnel tracker script
        cmd = [sys.executable, os.path.join(project_root, "scripts", "cloudflare_tunnel_tracker.py")]
        print(f"Starting Cloudflare tunnel with command: {' '.join(cmd)}")
        
        # Start in a background process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Start a thread to monitor and log the output
        def monitor_output():
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(f"Cloudflared: {line.strip()}")
        
        threading.Thread(target=monitor_output, daemon=True).start()
        
        print("Cloudflare tunnel process started")
        
    except Exception as e:
        print(f"Error starting Cloudflare tunnel: {e}")
        print("Continuing without Cloudflare tunnel...")


if __name__ == "__main__":
    # Check and install dependencies
    check_and_install_dependencies()
    
    # Start Cloudflare tunnel in background
    start_cloudflare_tunnel()
    
    # 현재 디렉토리를 MCP src로 변경
    os.chdir(str(project_root / "mcp" / "src"))
    
    import uvicorn
    
    print(f"Starting MCP API server from {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # 앱 실행
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000, 
        reload=True
    ) 