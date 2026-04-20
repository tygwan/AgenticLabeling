#!/bin/bash
# Few-Shot Learning Platform 실행 스크립트

# 로그 디렉토리 확인
LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# 현재 시간을 로그 파일명에 사용
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/fsl_platform_${TIMESTAMP}.log"

echo "Few-Shot Learning Platform을 시작합니다..."
echo "로그 파일: $LOG_FILE"

# 기존 프로세스 확인 및 종료
check_and_kill_process() {
    local port=$1
    local pid=$(lsof -t -i:$port -sTCP:LISTEN 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "포트 $port에서 실행 중인 프로세스($pid)를 종료합니다."
        kill -9 $pid 2>/dev/null
        sleep 1
    fi
}

# 포트 7860 사용 중이면 종료
check_and_kill_process 7860
check_and_kill_process 7861
check_and_kill_process 7862

# 환경 변수 설정 (Gradio 포트 충돌 방지)
export GRADIO_SERVER_PORT=7860
export GRADIO_ANALYTICS_ENABLED=False

# Few-Shot Learning Platform 실행 (Support Set Viewer 모드)
python run_few_shot_platform.py --mywebapp --port=7860 > "$LOG_FILE" 2>&1 &
PID=$!

echo "Few-Shot Learning Platform이 시작되었습니다(PID: $PID)."
echo "웹 인터페이스: http://localhost:7860"
echo "종료하려면: kill $PID"

# 웹 브라우저 실행 (3초 후)
sleep 3
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:7860 &
elif command -v open > /dev/null; then
    open http://localhost:7860 &
fi 