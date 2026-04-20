#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
디버그용 최소 웹앱

그라디오(Gradio)가 정상적으로 작동하는지 확인하기 위한 최소한의 웹앱입니다.
"""

import os
import sys
import logging
import traceback

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug_webapp.log")
    ]
)
logger = logging.getLogger("debug_webapp")

def main():
    """최소 웹앱 실행"""
    try:
        # 그라디오 임포트
        logger.info("그라디오 임포트 중...")
        import gradio as gr
        logger.info(f"그라디오 버전: {gr.__version__}")
        
        # 최소 인터페이스 정의
        logger.info("최소 인터페이스 정의 중...")
        def greet(name):
            return f"안녕하세요, {name}님!"
        
        demo = gr.Interface(
            fn=greet,
            inputs="text",
            outputs="text"
        )
        
        # 인터페이스 실행
        logger.info("인터페이스 실행 중...")
        demo.launch(server_port=7890, server_name="0.0.0.0")
        logger.info("인터페이스 실행 완료")
        
    except ImportError as e:
        logger.error(f"임포트 오류: {e}")
        logger.error("gradio 패키지가 설치되어 있는지 확인하세요: pip install gradio")
        sys.exit(1)
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 