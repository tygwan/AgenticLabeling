#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shot-Threshold Experiment Runner

이 스크립트는 여러 Shot 및 Threshold 조합으로 Few-Shot Learning 실험을 자동으로
실행하고 결과를 종합 분석합니다.

사용법:
    python run_shot_threshold_experiments.py --category test_category --model resnet
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("shot_threshold_experiments")

def run_experiment(category: str, model: str, shots: list, thresholds: list, skip_existing: bool = True):
    """
    지정된 Shot 및 Threshold 조합으로 실험 실행
    
    Args:
        category: 실험 대상 카테고리 이름
        model: 사용할 모델 (resnet, dino, clip)
        shots: 실험할 shot 값 목록
        thresholds: 실험할 threshold 값 목록
        skip_existing: 이미 결과가 있는 실험은 건너뛰기
    """
    logger.info(f"Shot-Threshold 실험 시작: 카테고리={category}, 모델={model}")
    logger.info(f"Shot 값: {shots}")
    logger.info(f"Threshold 값: {thresholds}")
    
    # 실험 스크립트 경로
    experiment_script = os.path.join("scripts", "classifier_cosine_experiment.py")
    
    # 실험별 지표 저장 변수
    results = []
    
    # 모든 Shot-Threshold 조합 실행
    total_experiments = len(shots) * len(thresholds)
    completed = 0
    
    for shot in shots:
        for threshold in thresholds:
            # 진행 상황 출력
            completed += 1
            logger.info(f"실험 {completed}/{total_experiments}: Shot={shot}, Threshold={threshold:.2f}")
            
            # 실험 ID 생성
            experiment_id = f"shot_{shot}_threshold_{threshold:.2f}"
            
            # 결과 확인 (이미 있으면 건너뛰기)
            results_dir = os.path.join("data", "test_category", "7.results", f"shot_{shot}", f"threshold_{threshold:.2f}")
            predictions_file = os.path.join(results_dir, "predictions.csv")
            
            if skip_existing and os.path.exists(predictions_file):
                logger.info(f"이미 결과가 있습니다. 건너뜁니다: {predictions_file}")
                
                # 기존 결과에서 지표 추출
                try:
                    # 예측 결과 로드
                    predictions_df = pd.read_csv(predictions_file)
                    
                    # 기본 지표 계산
                    total_images = len(predictions_df)
                    unknown_count = sum(1 for c in predictions_df["predicted_class"] if c == "Unknown")
                    classified_count = total_images - unknown_count
                    
                    # 결과 추가
                    results.append({
                        "shot": shot,
                        "threshold": threshold,
                        "total_images": total_images,
                        "classified_images": classified_count,
                        "unknown_images": unknown_count,
                        "classification_rate": classified_count / total_images if total_images > 0 else 0
                    })
                    
                except Exception as e:
                    logger.warning(f"기존 결과 분석 중 오류 발생: {e}")
                
                continue
            
            # 명령 생성
            cmd = [
                "python", experiment_script,
                "--category", category,
                "--model", model,
                "--shots", str(shot),
                "--thresholds", str(threshold)
            ]
            
            # 실험 실행
            try:
                logger.info(f"명령 실행: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"실험 완료: {experiment_id}")
                
                # 성공 로그 출력
                for line in result.stdout.splitlines():
                    if "분류 결과 요약" in line or "분류 성공률" in line or "클래스별 분류 결과" in line:
                        logger.info(f"  {line}")
                
                # 결과 분석 (predictions.csv 파일에서 지표 추출)
                if os.path.exists(predictions_file):
                    try:
                        # 예측 결과 로드
                        predictions_df = pd.read_csv(predictions_file)
                        
                        # 기본 지표 계산
                        total_images = len(predictions_df)
                        unknown_count = sum(1 for c in predictions_df["predicted_class"] if c == "Unknown")
                        classified_count = total_images - unknown_count
                        
                        # 결과 추가
                        results.append({
                            "shot": shot,
                            "threshold": threshold,
                            "total_images": total_images,
                            "classified_images": classified_count,
                            "unknown_images": unknown_count,
                            "classification_rate": classified_count / total_images if total_images > 0 else 0
                        })
                        
                    except Exception as e:
                        logger.warning(f"결과 분석 중 오류 발생: {e}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"실험 실행 중 오류 발생: {e}")
                logger.error(f"표준 오류: {e.stderr}")
    
    # 결과 요약 저장
    if results:
        # DataFrame 생성 및 저장
        results_df = pd.DataFrame(results)
        
        # 시간 기반 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join("data", "test_category", "7.results", f"experiment_summary_{timestamp}.csv")
        
        # 결과 저장
        results_df.to_csv(output_file, index=False)
        logger.info(f"실험 요약 저장 완료: {output_file}")
        
        # 결과 시각화
        visualize_results(results_df, os.path.dirname(output_file))
        
    logger.info("모든 Shot-Threshold 실험 완료")
    return results

def visualize_results(results_df: pd.DataFrame, output_dir: str):
    """
    실험 결과 시각화
    
    Args:
        results_df: 실험 결과 DataFrame
        output_dir: 결과 저장 디렉토리
    """
    # 시각화 저장 디렉토리
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Classification Rate 그래프
    plt.figure(figsize=(12, 8))
    for shot in sorted(results_df["shot"].unique()):
        subset = results_df[results_df["shot"] == shot]
        plt.plot(subset["threshold"], subset["classification_rate"], 
                 marker='o', label=f"{shot}-shot")
    
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Classification Rate")
    plt.title("Classification Rate by Threshold and Shot")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, "classification_rate.png"))
    plt.close()
    
    # 2. Unknown Rate 그래프
    plt.figure(figsize=(12, 8))
    for shot in sorted(results_df["shot"].unique()):
        subset = results_df[results_df["shot"] == shot]
        unknown_rate = subset["unknown_images"] / subset["total_images"]
        plt.plot(subset["threshold"], unknown_rate, 
                 marker='o', label=f"{shot}-shot")
    
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Unknown Rate")
    plt.title("Unknown Rate by Threshold and Shot")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, "unknown_rate.png"))
    plt.close()
    
    # 3. 히트맵 생성 (Classification Rate)
    create_heatmap(results_df, "classification_rate", 
                   "Classification Rate Heatmap", 
                   os.path.join(viz_dir, "classification_rate_heatmap.png"))
    
    # 4. 히트맵 생성 (Unknown Rate)
    results_df["unknown_rate"] = results_df["unknown_images"] / results_df["total_images"]
    create_heatmap(results_df, "unknown_rate", 
                   "Unknown Rate Heatmap", 
                   os.path.join(viz_dir, "unknown_rate_heatmap.png"))
    
    logger.info(f"시각화 결과 저장 완료: {viz_dir}")

def create_heatmap(df: pd.DataFrame, value_column: str, title: str, output_file: str):
    """
    Shot과 Threshold에 대한 히트맵 생성
    
    Args:
        df: 데이터 DataFrame
        value_column: 히트맵에 표시할 값 열 이름
        title: 히트맵 제목
        output_file: 저장할 파일 경로
    """
    # 피벗 테이블 생성
    pivot_df = df.pivot_table(
        index="shot", 
        columns="threshold", 
        values=value_column,
        aggfunc="mean"  # 중복 값이 있을 경우 평균 사용
    )
    
    # 히트맵 생성
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def analyze_existing_results(category: str):
    """
    기존 실험 결과 분석
    
    Args:
        category: 실험 대상 카테고리 이름
    """
    logger.info(f"기존 실험 결과 분석: 카테고리={category}")
    
    # 결과 디렉토리
    results_dir = os.path.join("data", category, "7.results")
    
    if not os.path.exists(results_dir):
        logger.error(f"결과 디렉토리를 찾을 수 없습니다: {results_dir}")
        return
    
    # 결과 요약 파일 찾기
    experiment_summary = os.path.join(results_dir, "experiment_comparison.csv")
    
    if os.path.exists(experiment_summary):
        logger.info(f"실험 요약 파일 발견: {experiment_summary}")
        
        # 결과 로드 및 시각화
        try:
            results_df = pd.read_csv(experiment_summary)
            visualize_results(results_df, results_dir)
            logger.info("기존 결과 분석 완료")
        except Exception as e:
            logger.error(f"결과 분석 중 오류 발생: {e}")
    else:
        logger.info(f"실험 요약 파일을 찾을 수 없습니다. 개별 실험 결과 수집 시도...")
        
        # 개별 실험 결과 수집
        results = []
        
        # Shot 폴더 탐색
        for shot_dir in os.listdir(results_dir):
            shot_path = os.path.join(results_dir, shot_dir)
            
            if os.path.isdir(shot_path) and shot_dir.startswith("shot_"):
                try:
                    # Shot 값 추출
                    shot = int(shot_dir.split("_")[1])
                    
                    # Threshold 폴더 탐색
                    for threshold_dir in os.listdir(shot_path):
                        threshold_path = os.path.join(shot_path, threshold_dir)
                        
                        if os.path.isdir(threshold_path) and threshold_dir.startswith("threshold_"):
                            try:
                                # Threshold 값 추출
                                threshold = float(threshold_dir.split("_")[1])
                                
                                # 예측 결과 파일 확인
                                predictions_file = os.path.join(threshold_path, "predictions.csv")
                                
                                if os.path.exists(predictions_file):
                                    # 예측 결과 로드
                                    predictions_df = pd.read_csv(predictions_file)
                                    
                                    # 기본 지표 계산
                                    total_images = len(predictions_df)
                                    unknown_count = sum(1 for c in predictions_df["predicted_class"] if c == "Unknown")
                                    classified_count = total_images - unknown_count
                                    
                                    # 결과 추가
                                    results.append({
                                        "shot": shot,
                                        "threshold": threshold,
                                        "total_images": total_images,
                                        "classified_images": classified_count,
                                        "unknown_images": unknown_count,
                                        "classification_rate": classified_count / total_images if total_images > 0 else 0
                                    })
                            except Exception as e:
                                logger.warning(f"Threshold 디렉토리 처리 중 오류 발생 ({threshold_dir}): {e}")
                except Exception as e:
                    logger.warning(f"Shot 디렉토리 처리 중 오류 발생 ({shot_dir}): {e}")
        
        # 결과 요약 저장 및 시각화
        if results:
            # DataFrame 생성 및 저장
            results_df = pd.DataFrame(results)
            
            # 결과 저장
            output_file = os.path.join(results_dir, "experiment_comparison.csv")
            results_df.to_csv(output_file, index=False)
            logger.info(f"실험 요약 저장 완료: {output_file}")
            
            # 결과 시각화
            visualize_results(results_df, results_dir)
            logger.info("기존 결과 분석 완료")
        else:
            logger.warning("분석할 실험 결과를 찾을 수 없습니다.")

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="Shot-Threshold 실험 자동화")
    
    parser.add_argument("--category", type=str, default="test_category",
                        help="실험 대상 카테고리 이름 (기본값: test_category)")
    parser.add_argument("--model", type=str, default="resnet",
                        choices=["clip", "dino", "resnet"],
                        help="특징 추출에 사용할 모델 이름 (기본값: resnet)")
    parser.add_argument("--shots", type=str, default="1,5,10,30",
                        help="실험할 shot 값 (쉼표로 구분, 기본값: 1,5,10,30)")
    parser.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.75,0.8,0.85,0.9",
                        help="실험할 threshold 값 (쉼표로 구분, 기본값: 0.5,0.6,0.7,0.75,0.8,0.85,0.9)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="실험 실행 없이 기존 결과 분석만 수행")
    parser.add_argument("--no-skip", action="store_true",
                        help="이미 결과가 있는 실험도 다시 실행")
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    
    # 기존 결과 분석만 수행
    if args.analyze_only:
        analyze_existing_results(args.category)
        return
    
    # Shot 및 Threshold 값 파싱
    try:
        shots = [int(s.strip()) for s in args.shots.split(",")]
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    except Exception as e:
        logger.error(f"Shot 또는 Threshold 값 파싱 오류: {e}")
        return
    
    # 실험 실행
    run_experiment(
        category=args.category,
        model=args.model,
        shots=shots,
        thresholds=thresholds,
        skip_existing=not args.no_skip
    )

if __name__ == "__main__":
    main() 