#!/bin/bash

# Ground Truth Labeling System 실행 스크립트
# 이 스크립트는 Ground Truth Labeling 시스템을 실행하는 데 사용됩니다.

# 로그 파일 설정
LOG_FILE="ground_truth_labeler.log"
echo "Ground Truth Labeling 시스템 시작 시간: $(date)" > $LOG_FILE

echo "==============================================="
echo "      Ground Truth Labeling System 실행기      "
echo "==============================================="
echo ""
echo "웹앱을 실행하고 Ground Truth Labeling 탭으로 이동합니다."
echo ""

# 파라미터 확인
SHARE_MODE=""
if [[ "$1" == "--share" ]]; then
    SHARE_MODE="--share"
    echo "공유 모드가 활성화되었습니다. 외부에서 접근 가능한 URL이 생성됩니다."
    echo ""
fi

# 필요한 모듈 확인
if [ ! -f "scripts/ground_truth_labeler.py" ]; then
    echo "오류: scripts/ground_truth_labeler.py 파일을 찾을 수 없습니다."
    echo "먼저 ground_truth_labeler.py 모듈을 생성하세요."
    exit 1
fi

# 결과 변환 안내
echo "먼저 Few-Shot Learning 결과를 변환하세요:"
echo "python scripts/convert_few_shot_results.py --category=<카테고리> --shot=<shot> --threshold=<threshold>"
echo ""

# 웹앱 실행
echo "웹앱을 실행합니다..."
echo "실행 중... (Ctrl+C로 중단)"
echo ""

# 메인 웹앱 실행
python scripts/main_webapp.py $SHARE_MODE 2>&1 | tee -a $LOG_FILE

echo ""
echo "웹앱이 종료되었습니다."