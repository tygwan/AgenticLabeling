#!/bin/bash

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== Few-Shot Learning 종합 분석 스크립트 =====${NC}"
echo "ResNet 및 DINO 모델을 사용하여 모든 shot 및 threshold 조합으로 실험을 실행하고,"
echo "ground truth 분석과 annotation 동기화를 수행합니다."

# 카테고리 설정 (기본값: test_category)
CATEGORY=${1:-"test_category"}
echo -e "${BLUE}사용 카테고리: ${YELLOW}${CATEGORY}${NC}"

# 각 모델과 shot에 대해 개별적으로 실행 (threshold 하나씩 실행)
echo -e "\n${GREEN}[1/3] 각 모델 및 shot에 대한 실험 실행 중...${NC}"

# 각 shot과 threshold 조합에 대한 배열 정의
SHOTS=(1 5 10 30)
THRESHOLDS=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)
MODELS=("resnet" "dino")

# 각 모델, shot, threshold 조합에 대해 실행
for MODEL in "${MODELS[@]}"; do
  echo -e "${BLUE}${MODEL} 모델로 실험 실행 중...${NC}"
  
  for SHOT in "${SHOTS[@]}"; do
    for THRESHOLD in "${THRESHOLDS[@]}"; do
      echo -e "${YELLOW}${MODEL} 모델, shot ${SHOT}, threshold ${THRESHOLD} 실행 중...${NC}"
      python scripts/classifier_cosine_experiment.py --category=${CATEGORY} --model=${MODEL} --shot=${SHOT} --threshold=${THRESHOLD}
    done
  done
done

# 2. Ground truth 분석
echo -e "\n${GREEN}[2/3] Ground truth 분석 중...${NC}"
python scripts/classifier_cosine_experiment.py --category=${CATEGORY} --analyze-ground-truth

# 3. 각 실험에 대해 annotation 동기화 및 비교
echo -e "\n${GREEN}[3/3] 실험 결과 annotation 동기화 및 비교 중...${NC}"

for MODEL in "${MODELS[@]}"; do
  echo -e "${BLUE}${MODEL} 모델 annotation 처리 중...${NC}"
  
  for SHOT in "${SHOTS[@]}"; do
    for THRESHOLD in "${THRESHOLDS[@]}"; do
      EXPERIMENT_ID="shot_${SHOT}_threshold_${THRESHOLD}"
      echo -e "${YELLOW}${MODEL} 모델 ${EXPERIMENT_ID} annotation 동기화 중...${NC}"
      
      # Annotation 동기화
      python scripts/classifier_cosine_experiment.py --category=${CATEGORY} --model=${MODEL} --sync-annotations --experiment-id=${EXPERIMENT_ID}
      
      # Annotation 비교
      echo -e "${YELLOW}${MODEL} 모델 ${EXPERIMENT_ID} ground truth와 비교 중...${NC}"
      python scripts/classifier_cosine_experiment.py --category=${CATEGORY} --model=${MODEL} --compare-annotations --experiment-id=${EXPERIMENT_ID}
    done
  done
done

echo -e "\n${GREEN}===== 모든 분석 작업이 완료되었습니다! =====${NC}"
echo "결과는 다음 위치에서 확인할 수 있습니다:"
echo "- ResNet 모델 결과: data/${CATEGORY}/7.results/resnet/"
echo "- DINO 모델 결과: data/${CATEGORY}/7.results/dino/"
echo "- Ground truth 분석: data/${CATEGORY}/7.results/ground_truth/annotations/"
echo "- 모델 비교 결과: data/${CATEGORY}/7.results/model_comparison/" 