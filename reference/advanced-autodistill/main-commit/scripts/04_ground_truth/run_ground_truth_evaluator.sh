#!/bin/bash

# 폴더 기반 Ground Truth 평가 도구 실행 스크립트

echo "폴더 기반 Ground Truth 평가 도구를 시작합니다..."

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
python -c "import matplotlib" >/dev/null 2>&1 || pip install matplotlib
python -c "import seaborn" >/dev/null 2>&1 || pip install seaborn
python -c "import sklearn" >/dev/null 2>&1 || pip install scikit-learn
python -c "import pandas" >/dev/null 2>&1 || pip install pandas
python -c "import numpy" >/dev/null 2>&1 || pip install numpy

# 카테고리 확인
if [ -z "$1" ]; then
    echo "사용법: $0 <카테고리> [ground_truth_폴더] [--visualize]"
    echo "예시: $0 test_category data/test_category/7.results/ground_truth --visualize"
    exit 1
fi

CATEGORY="$1"
GROUND_TRUTH_DIR=""
VISUALIZE=""

# 추가 매개변수 처리
shift
while [ "$#" -gt 0 ]; do
    case "$1" in
        --visualize)
            VISUALIZE="--visualize"
            ;;
        *)
            if [ -z "$GROUND_TRUTH_DIR" ]; then
                GROUND_TRUTH_DIR="$1"
            fi
            ;;
    esac
    shift
done

# Ground Truth 디렉토리 지정 (제공된 경우)
if [ -n "$GROUND_TRUTH_DIR" ]; then
    GROUND_TRUTH_PARAM="--ground-truth-dir $GROUND_TRUTH_DIR"
else
    GROUND_TRUTH_PARAM=""
fi

# 평가 도구 실행
python scripts/evaluate_ground_truth.py --category "$CATEGORY" $GROUND_TRUTH_PARAM $VISUALIZE

# 결과 확인
if [ $? -eq 0 ]; then
    echo "평가가 완료되었습니다."
    echo "결과는 data/$CATEGORY/7.results/ 디렉토리에서 확인할 수 있습니다."
else
    echo "평가 중 오류가 발생했습니다. ground_truth_eval.log 파일을 확인하세요."
fi 