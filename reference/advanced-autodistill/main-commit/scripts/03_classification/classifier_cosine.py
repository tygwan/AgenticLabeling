#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosine Similarity Classification Module

This script implements image classification based on cosine similarity between 
preprocessed images and support set images.

Key features:
1. Extract feature vectors from support set and preprocessed images
2. Calculate cosine similarity between vectors
3. Classify images based on highest similarity
4. Support K-shot learning and threshold selection via MCP

Usage:
    python classifier_cosine.py --category <category_name> --model <model_name> 
                               --k_shot <k> --threshold <threshold_value>

# 코사인 유사도 분류기 실행
python scripts/classifier_cosine.py --category test_category --model dino --k_shot 5 --threshold 0.75 --visualize


Models supported:
- CLIP
- DINOv2
- ResNet-50
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project utilities
from scripts.data_utils import get_category_path, load_class_mapping

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cosine_classifier")

# Model type definitions
MODEL_TYPES = {
    "clip": "CLIP ViT-B/32",
    "dino": "DINOv2 ViT-B/14",
    "resnet": "ResNet-50"
}

class FeatureExtractor:
    """Feature extractor class for converting images to feature vectors."""
    
    def __init__(self, model_name: str = "dino"):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the model to use for feature extraction
                        Options: "clip", "dino", "resnet"
        """
        self.model_name = model_name.lower()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing feature extractor with {MODEL_TYPES.get(self.model_name, self.model_name)}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the selected feature extraction model."""
        try:
            if self.model_name == "clip":
                try:
                    import clip
                    self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                except ImportError:
                    logger.error("CLIP package is not installed. Install with: pip install 'clip @ git+https://github.com/openai/CLIP.git'")
                    raise
                except Exception as e:
                    logger.error(f"Error loading CLIP model: {str(e)}")
                    raise
                
            elif self.model_name == "dino":
                try:
                    import torch
                    from transformers import AutoImageProcessor, AutoModel
                    
                    self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                    self.model = AutoModel.from_pretrained('facebook/dinov2-base')
                    self.model.to(self.device)
                except ImportError:
                    logger.error("Transformers package is not installed. Install with: pip install transformers")
                    raise
                except Exception as e:
                    logger.error(f"Error loading DINOv2 model: {str(e)}")
                    raise
                
            elif self.model_name == "resnet":
                try:
                    import torch
                    import torchvision.models as models
                    from torchvision import transforms
                    
                    self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                    # Remove the last fully connected layer to get the features
                    self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
                    self.model.to(self.device)
                    self.model.eval()
                    
                    self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225]),
                    ])
                except ImportError:
                    logger.error("Torchvision package is not installed. Install with: pip install torchvision")
                    raise
                except Exception as e:
                    logger.error(f"Error loading ResNet model: {str(e)}")
                    raise
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            logger.info(f"Successfully loaded {self.model_name} model")
        except Exception as e:
            logger.error(f"Failed to initialize feature extractor: {str(e)}")
            raise
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract feature vector from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array
        """
        with torch.no_grad():
            image = Image.open(image_path).convert('RGB')
            
            if self.model_name == "clip":
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                features = self.model.encode_image(image_input)
                
            elif self.model_name == "dino":
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.pooler_output
                
            elif self.model_name == "resnet":
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                features = self.model(image_input)
            
            # Convert to numpy and flatten
            return features.cpu().numpy().flatten()
            
    def extract_features_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Array of feature vectors
        """
        features = []
        for img_path in tqdm(image_paths, desc=f"Extracting features with {self.model_name}"):
            try:
                feat = self.extract_features(img_path)
                features.append(feat)
            except Exception as e:
                logger.error(f"Error extracting features from {img_path}: {e}")
                features.append(None)
        
        # Filter out None values
        features = [f for f in features if f is not None]
        
        return np.array(features)

class CosineSimilarityClassifier:
    """Classifier using cosine similarity to compare image features."""
    
    def __init__(
        self, 
        model_name: str = "dino",
        k_shot: int = 5,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the model to use for feature extraction
            k_shot: Number of examples to use for classification (K-shot learning)
            similarity_threshold: Minimum similarity score to assign a class
        """
        self.feature_extractor = FeatureExtractor(model_name)
        self.k_shot = k_shot
        self.similarity_threshold = similarity_threshold
        self.support_features = {}
        self.support_paths = {}
        self.category = None
        
    def load_support_set(self, category_name: str, shot_dir: str = None) -> Dict[str, List[str]]:
        """
        Load the support set for a category.
        
        Args:
            category_name: Name of the category
            shot_dir: Optional path to a specific shot directory (e.g., 'shot5')
            
        Returns:
            Dictionary mapping class names to lists of image paths
        """
        # Store the category name
        self.category = category_name
        
        # Get the support set directory
        category_path = get_category_path(category_name)
        support_dir = os.path.join(category_path, "2.support-set")
        
        if not os.path.exists(support_dir):
            raise FileNotFoundError(f"Support set directory not found: {support_dir}")
        
        # Load the class mapping
        class_mapping = load_class_mapping(category_name)
        logger.info(f"Loaded class mapping: {class_mapping}")
        
        # Shot 폴더 이름 결정 (기본: shotX)
        shot_folder = f"shot{self.k_shot}"
        if shot_dir:
            shot_folder = shot_dir if not os.path.isabs(shot_dir) else os.path.basename(shot_dir)
        
        logger.info(f"Using shot folder: {shot_folder}")
        
        # Create a dictionary to store image paths for each class
        support_images = {}
        
        # 폴더 구조 확인: 2.support-set/shot1/class_0 형식인지 확인
        if os.path.exists(os.path.join(support_dir, shot_folder)):
            logger.info(f"Found shot folder directly under support_dir: {shot_folder}")
            shot_path = os.path.join(support_dir, shot_folder)
            
            # shot폴더/class_0 구조
            for class_dir in os.listdir(shot_path):
                class_dir_path = os.path.join(shot_path, class_dir)
                
                if os.path.isdir(class_dir_path) and class_dir.lower().startswith('class_'):
                    # 클래스 이름 정규화 (Class_0 형식으로 통일)
                    if class_dir.lower().startswith('class_'):
                        class_num = class_dir.split('_')[1]
                        normalized_class_name = f"Class_{class_num}"
                    else:
                        normalized_class_name = class_dir
                        
                    logger.info(f"Using normalized class name: {normalized_class_name} (from folder: {class_dir})")
                    
                    # 이미지 파일 수집
                    image_files = []
                    for file_name in os.listdir(class_dir_path):
                        file_path = os.path.join(class_dir_path, file_name)
                        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            image_files.append(file_path)
                    
                    # 클래스별로 이미지 저장
                    if normalized_class_name not in support_images:
                        support_images[normalized_class_name] = []
                    
                    support_images[normalized_class_name].extend(image_files)
                    logger.info(f"Added {len(image_files)} images for {normalized_class_name} from {shot_folder}/{class_dir}")
        else:
            # 기존 구조: 클래스 폴더(class_0) 내에 shot 폴더(shot5)가 있는 구조
            for class_dir in os.listdir(support_dir):
                class_dir_path = os.path.join(support_dir, class_dir)
                
                if os.path.isdir(class_dir_path) and class_dir.lower().startswith('class_'):
                    # 클래스 이름 정규화 (Class_0 형식으로 통일)
                    if class_dir.lower().startswith('class_'):
                        class_num = class_dir.split('_')[1]
                        normalized_class_name = f"Class_{class_num}"
                    else:
                        normalized_class_name = class_dir
                        
                    logger.info(f"Using normalized class name: {normalized_class_name} (from folder: {class_dir})")
                    
                    # Shot 서브디렉토리 확인
                    shot_path = os.path.join(class_dir_path, shot_folder)
                    if os.path.exists(shot_path) and os.path.isdir(shot_path):
                        logger.info(f"Found shot directory for {normalized_class_name}: {shot_path}")
                        
                        # 이미지 파일 수집
                        image_files = []
                        for file_name in os.listdir(shot_path):
                            file_path = os.path.join(shot_path, file_name)
                            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                image_files.append(file_path)
                        
                        # 클래스별로 이미지 저장
                        if normalized_class_name not in support_images:
                            support_images[normalized_class_name] = []
                        
                        support_images[normalized_class_name].extend(image_files)
                        logger.info(f"Added {len(image_files)} images for {normalized_class_name} from {shot_folder}")
                    else:
                        logger.warning(f"Shot directory not found for {normalized_class_name}: {shot_path}")
        
        # 각 클래스별 이미지 수 로깅
        for cls, images in support_images.items():
            logger.info(f"Loaded {len(images)} support images for class {cls}")
            
            # 너무 적은 이미지가 있는 클래스 확인
            if len(images) < self.k_shot:
                logger.warning(f"클래스 {cls}에 필요한 {self.k_shot}개 이미지보다 적은 {len(images)}개가 있습니다.")
        
        # 클래스 매핑에 있는 모든 클래스가 있는지 확인
        for class_name in class_mapping.keys():
            # 클래스 이름 정규화 (매핑에 있는 클래스명도 정규화해서 비교)
            if class_name.lower().startswith('class_'):
                class_num = class_name.split('_')[1]
                normalized_map_class = f"Class_{class_num}"
            else:
                normalized_map_class = class_name
                
            if normalized_map_class not in support_images:
                logger.warning(f"클래스 매핑에 있는 클래스 {class_name} (정규화: {normalized_map_class})의 이미지가 없습니다.")
        
        self.support_paths = support_images
        return support_images
    
    def set_support_images_directly(self, support_images: Dict[str, List[str]]):
        """
        직접 support set 이미지를 설정하는 메서드
        
        Args:
            support_images: 클래스명을 키로, 이미지 경로 리스트를 값으로 갖는 사전
            
        Returns:
            설정된 support_images 사전
        """
        # 이미지 경로가 유효한지 확인
        for class_name, image_paths in support_images.items():
            valid_paths = []
            for img_path in image_paths:
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    valid_paths.append(img_path)
                else:
                    logger.warning(f"이미지를 찾을 수 없음: {img_path}")
            
            # 유효한 경로만 유지
            support_images[class_name] = valid_paths
            logger.info(f"클래스 {class_name}에 대해 {len(valid_paths)}개 이미지 로드됨")
        
        self.support_paths = support_images
        return support_images
    
    def extract_support_features(self):
        """Extract features from all support set images."""
        for class_name, image_paths in self.support_paths.items():
            logger.info(f"Extracting features for support class: {class_name}")
            class_features = self.feature_extractor.extract_features_batch(image_paths)
            self.support_features[class_name] = class_features
    
    def compute_similarity(self, query_feature: np.ndarray) -> Dict[str, float]:
        """
        Compute cosine similarity between query feature and support set.
        
        Args:
            query_feature: Feature vector of the query image
            
        Returns:
            Dictionary mapping class names to similarity scores
        """
        similarities = {}
        
        for class_name, features in self.support_features.items():
            # Calculate cosine similarity for each support image in the class
            class_similarities = []
            for feature in features:
                sim = self._cosine_similarity(query_feature, feature)
                class_similarities.append(sim)
            
            # Use the average similarity
            similarities[class_name] = np.mean(class_similarities)
        
        return similarities
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a, b: Input vectors
            
        Returns:
            Cosine similarity score (0-1)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a single image using cosine similarity.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Classification result with class and similarity
        """
        # 원본 클래스 추출 (파일 경로에서 추론)
        original_class = self._extract_original_class(image_path)
        
        # Extract features from the query image
        try:
            query_feature = self.feature_extractor.extract_features(image_path)
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return {
                "class": "Error", 
                "similarity": 0.0, 
                "all_scores": {}, 
                "image": image_path, 
                "original_class": original_class,
                "error": str(e)
            }
        
        # Calculate similarities to support classes
        similarities = self.compute_similarity(query_feature)
        
        # Find the most similar class
        if not similarities:
            return {
                "class": "Unknown", 
                "similarity": 0.0, 
                "all_scores": {}, 
                "image": image_path, 
                "original_class": original_class,
                "error": "No similarities computed"
            }
        
        sorted_similarities = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        best_class, best_score = sorted_similarities[0]
        
        # 두 번째로 높은 점수와의 차이 계산 (클래스 구분력 지표)
        margin = 0.0
        second_best_class = None
        second_best_score = 0.0
        
        if len(sorted_similarities) > 1:
            second_best_class, second_best_score = sorted_similarities[1]
            margin = best_score - second_best_score
        
        # 세 번째로 높은 점수와의 차이도 계산 (추가 정보)
        third_best_class = None
        third_best_score = 0.0
        
        if len(sorted_similarities) > 2:
            third_best_class, third_best_score = sorted_similarities[2]
        
        # 분류 결정 로직: threshold 미만이면 Unknown으로 분류
        final_class = best_class
        classification_status = "accepted"
        rejection_reason = None
        
        if best_score < self.similarity_threshold:
            final_class = "Unknown"
            classification_status = "rejected"
            rejection_reason = f"Best similarity ({best_score:.4f}) below threshold ({self.similarity_threshold:.4f})"
        
        # 신뢰도 레벨 결정 로직
        confidence_level = "low"
        if margin > 0.2:
            confidence_level = "high"
        elif margin > 0.1:
            confidence_level = "medium"
        
        return {
            "class": final_class,
            "predicted_class": best_class,  # 임계값 이전의 예측 클래스 (항상 있음)
            "similarity": float(best_score),  # confidence를 similarity로 변경
            "margin": float(margin),  # 구분력 지표
            "top_classes": [
                {"class": best_class, "similarity": float(best_score)},
                {"class": second_best_class, "similarity": float(second_best_score)} if second_best_class else None,
                {"class": third_best_class, "similarity": float(third_best_score)} if third_best_class else None
            ],
            "all_scores": {k: float(v) for k, v in similarities.items()},
            "image": image_path,
            "original_class": original_class,
            "confidence_level": confidence_level,
            "classification_status": classification_status,
            "rejection_reason": rejection_reason
        }
    
    def _extract_original_class(self, image_path: str) -> Optional[str]:
        """
        이미지 경로에서 원본 클래스 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            추출된 클래스 이름 또는 None
        """
        try:
            # preprocessed 디렉토리 구조: .../6.preprocessed/Class_0/image.jpg
            path_parts = Path(image_path).parts
            
            # Class_X 형식의 폴더 찾기
            for part in path_parts:
                if part.lower().startswith('class_'):
                    # 클래스 이름 정규화 (Class_0 형식으로 통일)
                    class_num = part.split('_')[1]
                    return f"Class_{class_num}"
                
            # 다른 형식으로 클래스가 인코딩된 경우 처리
            return None
        except Exception as e:
            logger.debug(f"원본 클래스 추출 실패: {e}")
            return None
    
    def classify_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a batch of images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of classification results
        """
        # 서포트셋 검증
        self.validate_support_set()
        
        # 분류 시작
        results = []
        errors = 0
        unknowns = 0
        
        for img_path in tqdm(image_paths, desc="Classifying images"):
            try:
                result = self.classify_image(img_path)
                # 추가 정보 기록
                result["image_filename"] = os.path.basename(img_path)
                results.append(result)
                
                # 오류 및 Unknown 클래스 카운트
                if result.get("class") == "Error":
                    errors += 1
                elif result.get("class") == "Unknown":
                    unknowns += 1
                    
            except Exception as e:
                logger.error(f"Error classifying {img_path}: {e}")
                results.append({
                    "class": "Error", 
                    "similarity": 0.0, 
                    "image": img_path,
                    "image_filename": os.path.basename(img_path),
                    "error": str(e)
                })
                errors += 1
        
        # 결과 요약 출력
        total = len(image_paths)
        successful = total - errors - unknowns
        logger.info(f"분류 결과 요약: 전체 {total}개, 성공 {successful}개, Unknown {unknowns}개, 오류 {errors}개")
        logger.info(f"분류 성공률: {successful/total*100:.2f}%")
        
        # 클래스별 분류 수 계산
        class_counts = {}
        for r in results:
            if r.get("class") not in class_counts:
                class_counts[r.get("class")] = 0
            class_counts[r.get("class")] += 1
        
        logger.info("클래스별 분류 결과:")
        for cls, count in class_counts.items():
            logger.info(f"  - {cls}: {count}개 ({count/total*100:.2f}%)")
        
        return results
        
    def validate_support_set(self):
        """
        Support set의 유효성 검증 및 진단 정보 출력
        
        Returns:
            bool: 유효성 여부
        """
        if not self.support_paths:
            logger.error("Support set이 로드되지 않았습니다.")
            return False
            
        if not self.support_features:
            logger.error("Support set 특성이 추출되지 않았습니다. extract_support_features()를 호출하세요.")
            return False
            
        # 각 클래스별 검증
        valid = True
        logger.info("=== Support Set 검증 ===")
        for class_name, paths in self.support_paths.items():
            features = self.support_features.get(class_name, [])
            
            if not paths:
                logger.error(f"클래스 {class_name}에 이미지가 없습니다.")
                valid = False
                continue
                
            if len(features) == 0:
                logger.error(f"클래스 {class_name}에 추출된 특성이 없습니다.")
                valid = False
                continue
                
            if len(paths) != len(features):
                logger.warning(f"클래스 {class_name}의 이미지 수({len(paths)})와 특성 수({len(features)})가 일치하지 않습니다.")
            
            logger.info(f"클래스 {class_name}: {len(paths)}개 이미지, {len(features)}개 특성")
            
            # 첫 번째 이미지 경로 출력
            if paths:
                logger.info(f"  - 샘플 이미지: {os.path.basename(paths[0])}")
        
        # 클래스 간 특성 유사도 검증 (클래스 구분이 잘 되는지 확인)
        logger.info("=== 클래스 간 특성 유사도 ===")
        classes = list(self.support_features.keys())
        for i, cls1 in enumerate(classes):
            for cls2 in classes[i+1:]:
                if not self.support_features[cls1].size or not self.support_features[cls2].size:
                    continue
                    
                # 각 클래스의 평균 특성 계산
                avg_feature1 = np.mean(self.support_features[cls1], axis=0)
                avg_feature2 = np.mean(self.support_features[cls2], axis=0)
                
                # 코사인 유사도 계산
                similarity = self._cosine_similarity(avg_feature1, avg_feature2)
                similarity_msg = f"{cls1} - {cls2}: {similarity:.4f}"
                
                # 유사도가 높으면 경고
                if similarity > 0.8:
                    logger.warning(f"⚠️ 높은 유사도! {similarity_msg}")
                    valid = False
                else:
                    logger.info(f"✓ {similarity_msg}")
        
        return valid
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save classification results to a JSON file.
        
        Args:
            results: List of classification results
            output_path: Path to save the results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
    def visualize_results(self, results: List[Dict[str, Any]], output_dir: str):
        """
        Visualize classification results with images and confidence scores.
        
        Args:
            results: List of classification results
            output_dir: Directory to save visualization images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            if "image" not in result:
                continue
                
            image_path = result["image"]
            predicted_class = result["class"]
            confidence = result["similarity"]
            
            # Load and display the image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(img)
            
            # Add title with prediction and confidence
            if predicted_class:
                title = f"Predicted: {predicted_class} ({confidence:.2f})"
            else:
                title = f"No class assigned (max similarity: {confidence:.2f})"
                
            plt.title(title)
            plt.axis('off')
            
            # Add scores as text
            all_scores = result["all_scores"]
            text = ""
            for cls, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                text += f"{cls}: {score:.2f}\n"
            
            plt.figtext(0.02, 0.02, text, fontsize=8)
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"result_{i}.jpg")
            plt.savefig(output_path)
            plt.close()
            
        logger.info(f"Visualizations saved to {output_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cosine similarity-based image classification")
    parser.add_argument("--category", type=str, required=True, help="Category name")
    parser.add_argument("--model", type=str, default="dino", choices=["clip", "dino", "resnet"], 
                        help="Feature extraction model")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of support images per class")
    parser.add_argument("--threshold", type=float, default=0.75, 
                        help="Similarity threshold for classification")
    parser.add_argument("--visualize", action="store_true", help="Visualize classification results")
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited images")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # 로깅 레벨 설정
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)
    
    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if args.debug:
        logger.debug("디버그 모드 활성화")
    
    logger.info(f"Starting cosine similarity classification for category: {args.category}")
    logger.info(f"Model: {args.model}, K-shot: {args.k_shot}, Threshold: {args.threshold}")
    
    try:
        # Support set 디렉토리 구조 탐색
        category_path = get_category_path(args.category)
        support_dir = os.path.join(category_path, "2.support-set")
        
        if not os.path.exists(support_dir):
            logger.error(f"Support set 디렉토리를 찾을 수 없습니다: {support_dir}")
            return
            
        # 기존 구조 (class_0 내에 shot1, shot5 등 폴더)
        shot_folder = f"shot{args.k_shot}"
        
        # 구조 분석
        logger.info("Support set 구조 분석 중...")
        available_classes = []
        missing_shots = []
        
        for class_dir in os.listdir(support_dir):
            class_dir_path = os.path.join(support_dir, class_dir)
            if os.path.isdir(class_dir_path) and class_dir.lower().startswith('class_'):
                available_classes.append(class_dir)
                
                # 지정된 shot 폴더가 있는지 확인
                shot_path = os.path.join(class_dir_path, shot_folder)
                if not os.path.exists(shot_path) or not os.path.isdir(shot_path):
                    missing_shots.append(class_dir)
        
        if not available_classes:
            logger.error("클래스 디렉토리를 찾을 수 없습니다.")
            return
            
        logger.info(f"사용 가능한 클래스: {available_classes}")
        
        if missing_shots:
            logger.warning(f"다음 클래스에 {shot_folder} 폴더가 없습니다: {missing_shots}")
        
        # Initialize the classifier
        classifier = CosineSimilarityClassifier(
            model_name=args.model,
            k_shot=args.k_shot,
            similarity_threshold=args.threshold
        )
        
        # Load support set
        logger.info("Support set 로드 중...")
        support_images = classifier.load_support_set(args.category, shot_folder)
        
        # 지원하는 클래스 확인
        logger.info(f"지원하는 클래스: {list(support_images.keys())}")
        
        # Support set이 비어있으면 종료
        if not support_images:
            logger.error("Support set이 비어있습니다. 종료합니다.")
            return
            
        # Extract features for support set
        logger.info("Support set에서 특성 추출 중...")
        classifier.extract_support_features()
        
        # Support set 검증
        classifier.validate_support_set()
        
        # Get preprocessed images (6.preprocessed 폴더의 이미지를 query로 사용)
        preprocessed_dir = os.path.join(category_path, "6.preprocessed")
        if not os.path.exists(preprocessed_dir):
            logger.error(f"전처리된 이미지 디렉토리를 찾을 수 없습니다: {preprocessed_dir}")
            return
        
        # 모든 전처리된 이미지 경로 수집
        logger.info("전처리된 이미지 수집 중...")
        preprocessed_images = []
        class_images = {}  # 클래스별 이미지 저장
        
        for item in os.listdir(preprocessed_dir):
            item_path = os.path.join(preprocessed_dir, item)
            if os.path.isdir(item_path) and item.startswith("Class_"):
                class_name = item  # 예: "Class_0"
                class_images[class_name] = []
                
                # 각 클래스 폴더에서 이미지 파일 수집
                for root, _, files in os.walk(item_path):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            file_path = os.path.join(root, f)
                            preprocessed_images.append(file_path)
                            class_images[class_name].append(file_path)
                
                logger.info(f"클래스 {class_name}에서 {len(class_images[class_name])}개 이미지 로드됨")
        
        logger.info(f"총 {len(preprocessed_images)}개의 전처리된 이미지를 찾았습니다.")
        
        # 테스트 모드이면 이미지 제한
        if args.test:
            max_test_images = min(10, len(preprocessed_images))
            logger.info(f"테스트 모드: {max_test_images}개 이미지로 제한합니다.")
            preprocessed_images = preprocessed_images[:max_test_images]
        
        # Classify images
        logger.info("이미지 분류 중...")
        results = classifier.classify_batch(preprocessed_images)
        
        # 결과 저장 디렉토리 생성
        results_dir = os.path.join(category_path, "7.results", f"shot_{args.k_shot}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(results_dir, "classification_results.json")
        classifier.save_results(results, results_file)
        logger.info(f"분류 결과가 저장되었습니다: {results_file}")
        
        # Visualize results if requested
        if args.visualize:
            logger.info("결과 시각화 중...")
            viz_dir = os.path.join(results_dir, "visualization")
            os.makedirs(viz_dir, exist_ok=True)
            classifier.visualize_results(results, viz_dir)
            logger.info(f"시각화 결과가 저장되었습니다: {viz_dir}")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        
    logger.info("분류 완료")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # 테스트 코드: ResNet 모델 로딩을 테스트합니다
        print("ResNet 모델 로딩 테스트 시작...")
        try:
            extractor = FeatureExtractor("resnet")
            print("✅ 성공: ResNet 모델이 정상적으로 로드되었습니다.")
        except Exception as e:
            print(f"❌ 실패: ResNet 모델 로드 중 오류 발생: {e}") 