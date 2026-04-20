#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Segmentation 모델 학습

생성된 YOLO segmentation 데이터셋을 사용하여 모델을 학습합니다.

사용법:
    python train_yolo_segmentation.py --data data/test_category/9.yolo-dataset/dataset.yaml --epochs 100
"""

import os
import argparse
import logging
import torch
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_yolo_segmentation(
    data_yaml: str,
    model: str = "yolov8n-seg.pt",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "",
    project: str = "runs/segment",
    output: str = "runs/segment",
    name: str = "train",
    patience: int = 50,
    save: bool = True,
    plots: bool = True,
    val: bool = True,
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    copy_paste: float = 0.0,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    **kwargs
):
    """
    YOLO segmentation 모델 학습
    
    Args:
        data_yaml: 데이터셋 YAML 파일 경로
        model: 사용할 모델 (pretrained 또는 config)
        epochs: 학습 에포크 수
        batch: 배치 크기
        imgsz: 이미지 크기
        device: 사용할 디바이스 ('', 'cpu', '0', '0,1' 등)
        project: 프로젝트 디렉토리
        name: 실험 이름
        patience: Early stopping patience
        save: 체크포인트 저장 여부
        plots: 플롯 생성 여부
        val: 검증 실행 여부
        lr0: 초기 학습률
        lrf: 최종 학습률 (lr0 * lrf)
        momentum: SGD momentum
        weight_decay: 가중치 감쇠
        warmup_epochs: 워밍업 에포크
        box: box loss gain
        cls: cls loss gain
        dfl: dfl loss gain
        copy_paste: Copy-paste 증강 확률 (0.0-1.0)
        hsv_h: HSV Hue 증강 (0.0-1.0)
        hsv_s: HSV Saturation 증강 (0.0-1.0)
        hsv_v: HSV Value 증강 (0.0-1.0)
        degrees: 회전 각도 (+/- deg)
        translate: 이동 변환 (+/- fraction)
        scale: 스케일 변환 (+/- gain)
        shear: 전단 변환 (+/- deg)
        perspective: 원근 변환 (+/- fraction)
        flipud: 상하 뒤집기 확률 (0.0-1.0)
        fliplr: 좌우 뒤집기 확률 (0.0-1.0)
        mosaic: 모자이크 증강 확률 (0.0-1.0)
        mixup: 믹스업 증강 확률 (0.0-1.0)
        **kwargs: 추가 학습 파라미터
    """
    
    # 데이터셋 존재 확인
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {data_yaml}")
    
    # GPU 사용 가능 확인
    if device == "":
        device = "0" if torch.cuda.is_available() else "cpu"
    
    logger.info("=" * 60)
    logger.info("YOLO Segmentation 모델 학습 시작")
    logger.info("=" * 60)
    logger.info(f"데이터셋: {data_yaml}")
    logger.info(f"모델: {model}")
    logger.info(f"에포크: {epochs}")
    logger.info(f"배치 크기: {batch}")
    logger.info(f"이미지 크기: {imgsz}")
    logger.info(f"디바이스: {device}")
    logger.info(f"출력 디렉토리: {project}/{name}")
    
    # 데이터 증강 설정 출력
    logger.info("데이터 증강 설정:")
    logger.info(f"  copy_paste: {copy_paste}")
    logger.info(f"  hsv_h: {hsv_h}, hsv_s: {hsv_s}, hsv_v: {hsv_v}")
    logger.info(f"  degrees: {degrees}, translate: {translate}, scale: {scale}")
    logger.info(f"  shear: {shear}, perspective: {perspective}")
    logger.info(f"  flipud: {flipud}, fliplr: {fliplr}")
    logger.info(f"  mosaic: {mosaic}, mixup: {mixup}")
    logger.info("=" * 60)
    
    # YOLO 모델 로드
    try:
        yolo_model = YOLO(model)
        logger.info(f"모델 로드 성공: {model}")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise
    
    # 학습 시작
    try:
        results = yolo_model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            patience=patience,
            save=save,
            plots=plots,
            val=val,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            box=box,
            cls=cls,
            dfl=dfl,
            copy_paste=copy_paste,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            **kwargs
        )
        
        logger.info("=" * 60)
        logger.info("학습 완료!")
        logger.info(f"최종 모델: {results.save_dir}/weights/best.pt")
        logger.info(f"학습 결과: {results.save_dir}")
        
        # 모델 성능 요약
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            logger.info("최종 성능 지표:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
        
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='YOLO Segmentation 모델 학습')
    
    # 필수 인자
    parser.add_argument('--data', required=True, 
                        help='데이터셋 YAML 파일 경로 (예: data/test_category/9.yolo-dataset/dataset.yaml)')
    
    # 모델 설정
    parser.add_argument('--model', default='yolov8n-seg.pt',
                        help='사용할 모델 (기본: yolov8n-seg.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에포크 수 (기본: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='배치 크기 (기본: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='이미지 크기 (기본: 640)')
    parser.add_argument('--device', default='',
                        help='사용할 디바이스 (기본: 자동 선택)')
    
    # 출력 설정
    parser.add_argument('--project', default=None,
                        help='프로젝트 디렉토리 (기본: <data_path>/runs/segment)')
    parser.add_argument('--name', default=None,
                        help='실험 이름 (기본: <dataset_name>_<model_name>_<timestamp>)')
    
    # 학습 파라미터
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (기본: 50)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='초기 학습률 (기본: 0.01)')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='최종 학습률 비율 (기본: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD momentum (기본: 0.937)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='가중치 감쇠 (기본: 0.0005)')
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                        help='워밍업 에포크 (기본: 3.0)')
    
    # Loss gains
    parser.add_argument('--box', type=float, default=7.5,
                        help='Box loss gain (기본: 7.5)')
    parser.add_argument('--cls', type=float, default=0.5,
                        help='Class loss gain (기본: 0.5)')
    parser.add_argument('--dfl', type=float, default=1.5,
                        help='DFL loss gain (기본: 1.5)')
    
    # 데이터 증강 파라미터
    parser.add_argument('--copy-paste', type=float, default=0.0,
                        help='Copy-paste 증강 확률 (0.0-1.0)')
    parser.add_argument('--hsv-h', type=float, default=0.015,
                        help='HSV Hue 증강 (0.0-1.0)')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                        help='HSV Saturation 증강 (0.0-1.0)')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                        help='HSV Value 증강 (0.0-1.0)')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='회전 각도 (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='이동 변환 (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='스케일 변환 (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0,
                        help='전단 변환 (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0,
                        help='원근 변환 (+/- fraction)')
    parser.add_argument('--flipud', type=float, default=0.0,
                        help='상하 뒤집기 확률 (0.0-1.0)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='좌우 뒤집기 확률 (0.0-1.0)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='모자이크 증강 확률 (0.0-1.0)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='믹스업 증강 확률 (0.0-1.0)')
    
    # 기타 설정
    parser.add_argument('--no-save', action='store_true',
                        help='체크포인트 저장 안함')
    parser.add_argument('--no-plots', action='store_true',
                        help='플롯 생성 안함')
    parser.add_argument('--no-val', action='store_true',
                        help='검증 실행 안함')
    parser.add_argument('--verbose', action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 동적 경로 및 이름 설정
    data_path = Path(args.data)
    
    # --project 인수가 제공되지 않은 경우, data.yaml 위치 기반으로 자동 설정
    if args.project is None:
        args.project = str(data_path.parent / 'runs' / 'segment')
        
    # --name 인수가 제공되지 않은 경우, 데이터셋/모델/타임스탬프 기반으로 자동 생성
    if args.name is None:
        dataset_name = data_path.parent.name
        model_name = Path(args.model).stem.replace('-seg', '') # 모델명에서 '-seg' 제거
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f"{dataset_name}_{model_name}_{timestamp}"
        
    # 인자 변환
    kwargs = {
        'save': not args.no_save,
        'plots': not args.no_plots,
        'val': not args.no_val,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
    }
    
    # 학습 실행
    try:
        results = train_yolo_segmentation(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            box=args.box,
            cls=args.cls,
            dfl=args.dfl,
            copy_paste=getattr(args, 'copy_paste'),
            hsv_h=getattr(args, 'hsv_h'),
            hsv_s=getattr(args, 'hsv_s'),
            hsv_v=getattr(args, 'hsv_v'),
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            **kwargs
        )
        
        print(f"\n학습 완료! 결과는 다음 위치에 저장되었습니다:")
        print(f"📁 {results.save_dir}")
        print(f"🏆 최고 모델: {results.save_dir}/weights/best.pt")
        print(f"📊 마지막 모델: {results.save_dir}/weights/last.pt")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        logger.error(f"학습 실패: {e}")
        raise

if __name__ == "__main__":
    main() 