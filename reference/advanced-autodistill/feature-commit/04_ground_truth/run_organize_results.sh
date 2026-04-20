#!/bin/bash

# Few-Shot 분류 결과 정리 도구 실행 스크립트

echo "Few-Shot 분류 결과 정리 도구를 시작합니다..."

# 스크립트 실행 디렉토리로 변경
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 필요한 환경변수 설정
export PYTHONPATH="$PWD:$PYTHONPATH"

# Python 가상 환경 활성화 (필요한 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
fi

# 필요한 패키지 확인 및 설치
python -c "import json" >/dev/null 2>&1 || pip install json
python -c "import shutil" >/dev/null 2>&1 || pip install shutil
python -c "import concurrent.futures" >/dev/null 2>&1 || pip install futures

# 카테고리 확인
if [ -z "$1" ]; then
    echo "사용법: $0 <카테고리> [옵션]"
    echo "옵션:"
    echo "  --shot-values=1,5,10,30       정리할 shot 값들 (쉼표로 구분)"
    echo "  --threshold-values=0.5,0.6,0.7,0.8,0.9  정리할 threshold 값들 (쉼표로 구분)"
    echo "  --source-dir=<경로>           원본 이미지 디렉토리 (기본값: data/{category}/5.dataset/val)"
    echo "  --force                       기존 분류 폴더가 있을 경우 덮어쓰기"
    echo ""
    echo "예시: $0 test_category --force"
    exit 1
fi

CATEGORY="$1"
shift
EXTRA_ARGS="$@"

# 실행 권한 확인 및 부여
if [ ! -x "scripts/organize_classification_results.py" ]; then
    chmod +x scripts/organize_classification_results.py
fi

# 결과 정리 도구 실행
python scripts/organize_classification_results.py --category "$CATEGORY" $EXTRA_ARGS

# 결과 확인
if [ $? -eq 0 ]; then
    echo "결과 정리가 완료되었습니다."
    echo "각 shot*threshold 조합마다 class_0, class_1, class_2, class_3, unknown 폴더가 생성되었습니다."
    echo "요약 정보는 data/$CATEGORY/7.results/classification_summary.json 파일에서 확인할 수 있습니다."
else
    echo "결과 정리 중 오류가 발생했습니다. organize_results.log 파일을 확인하세요."
fi 