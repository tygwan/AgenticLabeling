#!/usr/bin/env python
"""
MCP 및 Cloudflare 설정 업데이트 스크립트

이 스크립트는 MCP(Minimum Control Program)와 Cloudflare 터널 설정을 업데이트합니다.
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import json

# 스크립트가 있는 디렉토리를 시스템 경로에 추가
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir.parent))

# 데이터 유틸리티 모듈 가져오기
from scripts.data_utils import (
    update_mcp_config,
    get_cloudflare_command,
    generate_docker_command,
    MCP_CONFIG
)

def parse_args():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="MCP 및 Cloudflare 설정 업데이트")
    
    # 기본 인수
    parser.add_argument("--webhook-url", 
                        help="Cloudflare 터널 URL (예: https://example-tunnel.trycloudflare.com)")
    parser.add_argument("--port", type=int, default=5678,
                        help="MCP 서버 포트 (기본값: 5678)")
    parser.add_argument("--volume-path", default="E:/Ubuntu_AGI/n8n",
                        help="Docker 볼륨 마운트 경로 (기본값: E:/Ubuntu_AGI/n8n)")
    
    # 명령 플래그
    parser.add_argument("--update-config", action="store_true",
                       help="MCP 구성 파일 업데이트")
    parser.add_argument("--show-docker-command", action="store_true",
                       help="Docker 실행 명령 표시")
    parser.add_argument("--show-cloudflare-command", action="store_true",
                       help="Cloudflare 터널 명령 표시")
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    
    # 웹훅 URL 지정되지 않으면 현재 설정 사용
    webhook_url = args.webhook_url or MCP_CONFIG.get("webhook_url")
    
    # MCP 구성 업데이트
    if args.update_config or args.webhook_url or args.port:
        config = update_mcp_config(webhook_url, args.port)
        print(f"MCP 구성이 업데이트되었습니다:")
        print(json.dumps(config, indent=2))
    
    # Cloudflare 터널 명령 표시
    if args.show_cloudflare_command:
        cmd = get_cloudflare_command()
        print("\nCloudflare 터널 명령:")
        print(f"\n{cmd}\n")
    
    # Docker 실행 명령 표시
    if args.show_docker_command:
        cmd = generate_docker_command(webhook_url, args.volume_path)
        print("\nDocker 실행 명령:")
        print(f"\n{cmd}\n")
    
    # 아무 플래그도 지정되지 않으면 기본 정보 표시
    if not any([args.update_config, args.show_cloudflare_command, args.show_docker_command,
               args.webhook_url, args.port != 5678]):
        print("현재 MCP 설정:")
        print(json.dumps(MCP_CONFIG, indent=2))
        print("\n사용법: python update_mcp.py --webhook-url <URL> --update-config")
        print("자세한 사용법은 -h 또는 --help 옵션을 사용하세요.")

if __name__ == "__main__":
    main() 