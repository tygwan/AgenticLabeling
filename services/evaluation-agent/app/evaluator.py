"""Model evaluation implementation with mAP, Confusion Matrix, and detailed metrics."""
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ModelEvaluator:
    """Evaluates models and calculates comprehensive metrics."""

    def __init__(self, results_dir: str = "/data/evaluations"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._evaluations: Dict[str, Dict] = {}

    def evaluate_detection(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate object detection predictions with detailed metrics.

        Args:
            predictions: List of {boxes, labels, scores} per image
            ground_truth: List of {boxes, labels} per image
            iou_threshold: IoU threshold for positive match
            classes: List of class names for confusion matrix

        Returns:
            Dict with mAP, precision, recall, confusion matrix, etc.
        """
        # Collect all classes if not provided
        if classes is None:
            classes = self._collect_classes(predictions, ground_truth)

        # Initialize per-class stats
        class_stats = {cls: {"tp": 0, "fp": 0, "fn": 0, "scores": []} for cls in classes}

        # Confusion matrix: rows=GT, cols=Pred
        n_classes = len(classes)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # +1 for background

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = np.array(pred.get("boxes", []))
            pred_labels = pred.get("labels", [])
            pred_scores = pred.get("scores", [1.0] * len(pred_boxes))

            gt_boxes = np.array(gt.get("boxes", []))
            gt_labels = gt.get("labels", [])

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue

            if len(gt_boxes) == 0:
                # All predictions are false positives
                for label in pred_labels:
                    if label in class_stats:
                        class_stats[label]["fp"] += 1
                    total_fp += 1
                continue

            if len(pred_boxes) == 0:
                # All ground truth are false negatives
                for label in gt_labels:
                    if label in class_stats:
                        class_stats[label]["fn"] += 1
                    total_fn += 1
                continue

            # Calculate IoU matrix
            iou_matrix = self._compute_iou_matrix(pred_boxes, gt_boxes)

            # Match predictions to ground truth (greedy matching by score)
            sorted_indices = np.argsort(pred_scores)[::-1]
            matched_gt = set()

            for pred_idx in sorted_indices:
                pred_label = pred_labels[pred_idx] if pred_idx < len(pred_labels) else "unknown"
                pred_score = pred_scores[pred_idx] if pred_idx < len(pred_scores) else 1.0

                best_iou = 0
                best_gt_idx = -1

                for gt_idx in range(len(gt_boxes)):
                    if gt_idx in matched_gt:
                        continue

                    gt_label = gt_labels[gt_idx] if gt_idx < len(gt_labels) else "unknown"

                    # Only match same class
                    if pred_label != gt_label:
                        continue

                    if iou_matrix[pred_idx, gt_idx] > best_iou:
                        best_iou = iou_matrix[pred_idx, gt_idx]
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    # True positive
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                    if pred_label in class_stats:
                        class_stats[pred_label]["tp"] += 1
                        class_stats[pred_label]["scores"].append((pred_score, True))

                    # Update confusion matrix (correct prediction)
                    if pred_label in class_to_idx:
                        confusion_matrix[class_to_idx[pred_label], class_to_idx[pred_label]] += 1
                else:
                    # False positive
                    total_fp += 1
                    if pred_label in class_stats:
                        class_stats[pred_label]["fp"] += 1
                        class_stats[pred_label]["scores"].append((pred_score, False))

                    # Update confusion matrix (wrong class or background)
                    if pred_label in class_to_idx:
                        confusion_matrix[n_classes, class_to_idx[pred_label]] += 1  # Background predicted as class

            # False negatives (unmatched ground truth)
            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in matched_gt:
                    total_fn += 1
                    gt_label = gt_labels[gt_idx] if gt_idx < len(gt_labels) else "unknown"
                    if gt_label in class_stats:
                        class_stats[gt_label]["fn"] += 1

                    # Update confusion matrix (missed detection)
                    if gt_label in class_to_idx:
                        confusion_matrix[class_to_idx[gt_label], n_classes] += 1

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        # Calculate per-class metrics and AP
        per_class_metrics = {}
        aps = []

        for cls in classes:
            stats = class_stats[cls]
            cls_precision = stats["tp"] / (stats["tp"] + stats["fp"] + 1e-10)
            cls_recall = stats["tp"] / (stats["tp"] + stats["fn"] + 1e-10)
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall + 1e-10)

            # Calculate AP for this class
            ap = self._calculate_ap(stats["scores"])
            aps.append(ap)

            per_class_metrics[cls] = {
                "precision": float(cls_precision),
                "recall": float(cls_recall),
                "f1_score": float(cls_f1),
                "ap": float(ap),
                "true_positives": stats["tp"],
                "false_positives": stats["fp"],
                "false_negatives": stats["fn"],
                "support": stats["tp"] + stats["fn"],
            }

        # Calculate mAP
        mAP = np.mean(aps) if aps else 0.0

        # Store evaluation result
        eval_id = f"eval_{uuid.uuid4().hex[:12]}"
        self._evaluations[eval_id] = {
            "type": "detection",
            "created_at": datetime.utcnow().isoformat(),
            "metrics": {
                "mAP": float(mAP),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
            },
            "per_class": per_class_metrics,
            "confusion_matrix": confusion_matrix.tolist(),
            "classes": classes,
        }

        return {
            "evaluation_id": eval_id,
            "mAP": float(mAP),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "iou_threshold": iou_threshold,
            "per_class": per_class_metrics,
            "confusion_matrix": confusion_matrix.tolist(),
            "classes": classes + ["background"],
        }

    def _collect_classes(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> List[str]:
        """Collect all unique classes from predictions and ground truth."""
        classes = set()
        for pred in predictions:
            classes.update(pred.get("labels", []))
        for gt in ground_truth:
            classes.update(gt.get("labels", []))
        return sorted(list(classes))

    def _calculate_ap(self, scores: List[Tuple[float, bool]]) -> float:
        """Calculate Average Precision from score-label pairs."""
        if not scores:
            return 0.0

        # Sort by score descending
        scores = sorted(scores, key=lambda x: x[0], reverse=True)

        precisions = []
        recalls = []

        tp = 0
        fp = 0
        total_positives = sum(1 for _, is_tp in scores if is_tp)

        if total_positives == 0:
            return 0.0

        for score, is_tp in scores:
            if is_tp:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp)
            recall = tp / total_positives

            precisions.append(precision)
            recalls.append(recall)

        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            precisions_above_t = [p for p, r in zip(precisions, recalls) if r >= t]
            if precisions_above_t:
                ap += max(precisions_above_t) / 11

        return ap

    def calculate_map_at_thresholds(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_thresholds: List[float] = None,
    ) -> Dict[str, Any]:
        """Calculate mAP at multiple IoU thresholds (COCO-style mAP50-95)."""
        if iou_thresholds is None:
            iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95

        maps = []
        threshold_results = {}

        for thresh in iou_thresholds:
            result = self.evaluate_detection(predictions, ground_truth, thresh)
            maps.append(result["mAP"])
            threshold_results[f"mAP{int(thresh*100)}"] = result["mAP"]

        return {
            "mAP50": threshold_results.get("mAP50", 0),
            "mAP50-95": float(np.mean(maps)),
            "per_threshold": threshold_results,
        }

    def _compute_iou_matrix(
        self, boxes1: np.ndarray, boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU between all pairs of boxes."""
        n1, n2 = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])

        return iou_matrix

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes (x1, y1, x2, y2 format)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-10)

    def evaluate_classification(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> Dict[str, Any]:
        """Evaluate classification predictions with confusion matrix."""
        pred_labels = [p.get("class", p.get("label")) for p in predictions]
        gt_labels = [g.get("class", g.get("label")) for g in ground_truth]

        # Collect all classes
        classes = sorted(list(set(pred_labels + gt_labels)))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        n_classes = len(classes)

        # Build confusion matrix
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

        correct = 0
        total = len(pred_labels)

        for pred, gt in zip(pred_labels, gt_labels):
            if pred == gt:
                correct += 1
            if pred in class_to_idx and gt in class_to_idx:
                confusion_matrix[class_to_idx[gt], class_to_idx[pred]] += 1

        # Per-class metrics
        per_class = {}
        macro_precision = 0
        macro_recall = 0
        macro_f1 = 0

        for cls in classes:
            idx = class_to_idx[cls]
            tp = confusion_matrix[idx, idx]
            fp = confusion_matrix[:, idx].sum() - tp
            fn = confusion_matrix[idx, :].sum() - tp

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            per_class[str(cls)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(confusion_matrix[idx, :].sum()),
            }

            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1

        n = len(classes)

        # Store evaluation
        eval_id = f"eval_{uuid.uuid4().hex[:12]}"
        self._evaluations[eval_id] = {
            "type": "classification",
            "created_at": datetime.utcnow().isoformat(),
            "metrics": {
                "accuracy": correct / total if total > 0 else 0,
                "macro_precision": macro_precision / n if n > 0 else 0,
                "macro_recall": macro_recall / n if n > 0 else 0,
                "macro_f1": macro_f1 / n if n > 0 else 0,
            },
            "per_class": per_class,
            "confusion_matrix": confusion_matrix.tolist(),
            "classes": classes,
        }

        return {
            "evaluation_id": eval_id,
            "accuracy": correct / total if total > 0 else 0,
            "macro_precision": float(macro_precision / n) if n > 0 else 0,
            "macro_recall": float(macro_recall / n) if n > 0 else 0,
            "macro_f1": float(macro_f1 / n) if n > 0 else 0,
            "total": total,
            "correct": correct,
            "per_class": per_class,
            "confusion_matrix": confusion_matrix.tolist(),
            "classes": classes,
        }

    def evaluate_segmentation(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> Dict[str, Any]:
        """Evaluate segmentation predictions with detailed metrics."""
        total_iou = 0
        total_dice = 0
        total_pixel_acc = 0
        count = 0

        per_image_metrics = []

        for pred, gt in zip(predictions, ground_truth):
            pred_mask = np.array(pred.get("mask", []))
            gt_mask = np.array(gt.get("mask", []))

            if pred_mask.size == 0 or gt_mask.size == 0:
                continue

            # Ensure boolean masks
            pred_mask = pred_mask.astype(bool)
            gt_mask = gt_mask.astype(bool)

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            iou = intersection / (union + 1e-10)
            dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-10)

            # Pixel accuracy
            correct_pixels = (pred_mask == gt_mask).sum()
            total_pixels = pred_mask.size
            pixel_acc = correct_pixels / total_pixels

            total_iou += iou
            total_dice += dice
            total_pixel_acc += pixel_acc
            count += 1

            per_image_metrics.append({
                "iou": float(iou),
                "dice": float(dice),
                "pixel_accuracy": float(pixel_acc),
            })

        eval_id = f"eval_{uuid.uuid4().hex[:12]}"

        result = {
            "evaluation_id": eval_id,
            "mIoU": float(total_iou / count) if count > 0 else 0,
            "dice_coefficient": float(total_dice / count) if count > 0 else 0,
            "pixel_accuracy": float(total_pixel_acc / count) if count > 0 else 0,
            "samples_evaluated": count,
        }

        self._evaluations[eval_id] = {
            "type": "segmentation",
            "created_at": datetime.utcnow().isoformat(),
            "metrics": result,
            "per_image": per_image_metrics,
        }

        return result

    def evaluate_model(
        self,
        model_path: str,
        dataset_path: str,
        task: str = "detection",
    ) -> Dict[str, Any]:
        """Evaluate a trained YOLO model on a dataset."""
        from ultralytics import YOLO

        model = YOLO(model_path)
        results = model.val(data=dataset_path, verbose=False)

        # Extract metrics
        metrics = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        }

        # Extract per-class metrics if available
        if hasattr(results, 'maps') and results.maps is not None:
            class_names = model.names
            per_class = {}
            for i, ap in enumerate(results.maps):
                if i < len(class_names):
                    per_class[class_names[i]] = {"mAP50-95": float(ap)}
            metrics["per_class"] = per_class

        # Extract confusion matrix if available
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            cm = results.confusion_matrix.matrix
            metrics["confusion_matrix"] = cm.tolist()
            metrics["classes"] = list(model.names.values()) + ["background"]

        # Speed metrics
        if hasattr(results, 'speed'):
            metrics["speed"] = {
                "preprocess_ms": results.speed.get("preprocess", 0),
                "inference_ms": results.speed.get("inference", 0),
                "postprocess_ms": results.speed.get("postprocess", 0),
            }

        eval_id = f"eval_{uuid.uuid4().hex[:12]}"
        self._evaluations[eval_id] = {
            "type": "model",
            "model_path": model_path,
            "dataset_path": dataset_path,
            "created_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
        metrics["evaluation_id"] = eval_id

        return metrics

    def compare_models(
        self,
        model_paths: List[str],
        dataset_path: str,
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Compare multiple models on the same dataset."""
        results = {}
        best_scores = {m: (None, -float("inf")) for m in metrics}

        for path in model_paths:
            model_metrics = self.evaluate_model(path, dataset_path)
            results[path] = model_metrics

            for metric in metrics:
                # Handle nested metrics like mAP50-95
                score = model_metrics.get(metric, 0)
                if score > best_scores[metric][1]:
                    best_scores[metric] = (path, score)

        # Rank models by primary metric (mAP50-95)
        ranking = sorted(
            results.items(),
            key=lambda x: x[1].get("mAP50-95", 0),
            reverse=True
        )

        return {
            "models": results,
            "best": {m: path for m, (path, _) in best_scores.items()},
            "ranking": [{"model": path, "mAP50-95": r.get("mAP50-95", 0)} for path, r in ranking],
        }

    def get_evaluation(self, evaluation_id: str) -> Optional[Dict]:
        """Get stored evaluation by ID."""
        return self._evaluations.get(evaluation_id)

    def generate_report(
        self,
        evaluation_id: str,
        format: str = "json",
        include_visualizations: bool = False,
    ) -> Dict[str, Any]:
        """Generate evaluation report."""
        evaluation = self._evaluations.get(evaluation_id)

        if not evaluation:
            return {"error": f"Evaluation {evaluation_id} not found"}

        report = {
            "evaluation_id": evaluation_id,
            "type": evaluation.get("type"),
            "created_at": evaluation.get("created_at"),
            "metrics": evaluation.get("metrics", {}),
            "per_class": evaluation.get("per_class", {}),
        }

        if "confusion_matrix" in evaluation:
            report["confusion_matrix"] = {
                "matrix": evaluation["confusion_matrix"],
                "classes": evaluation.get("classes", []),
            }

        if include_visualizations:
            # Generate visualization data (for frontend rendering)
            report["visualizations"] = self._generate_visualization_data(evaluation)

        if format == "json":
            return report
        elif format == "markdown":
            return {"content": self._format_as_markdown(report)}

        return report

    def _generate_visualization_data(self, evaluation: Dict) -> Dict:
        """Generate data for visualizations."""
        viz_data = {}

        # Confusion matrix heatmap data
        if "confusion_matrix" in evaluation:
            viz_data["confusion_matrix"] = {
                "type": "heatmap",
                "data": evaluation["confusion_matrix"],
                "x_labels": evaluation.get("classes", []),
                "y_labels": evaluation.get("classes", []),
                "title": "Confusion Matrix",
            }

        # Per-class bar chart data
        if "per_class" in evaluation:
            per_class = evaluation["per_class"]
            viz_data["per_class_metrics"] = {
                "type": "bar_chart",
                "labels": list(per_class.keys()),
                "datasets": [
                    {
                        "label": "Precision",
                        "data": [v.get("precision", 0) for v in per_class.values()],
                    },
                    {
                        "label": "Recall",
                        "data": [v.get("recall", 0) for v in per_class.values()],
                    },
                    {
                        "label": "F1",
                        "data": [v.get("f1", v.get("f1_score", 0)) for v in per_class.values()],
                    },
                ],
                "title": "Per-Class Metrics",
            }

        return viz_data

    def _format_as_markdown(self, report: Dict) -> str:
        """Format report as markdown."""
        lines = [
            f"# Evaluation Report",
            f"",
            f"**Evaluation ID:** {report.get('evaluation_id')}",
            f"**Type:** {report.get('type')}",
            f"**Created:** {report.get('created_at')}",
            f"",
            f"## Overall Metrics",
            f"",
        ]

        metrics = report.get("metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"- **{key}:** {value:.4f}" if isinstance(value, float) else f"- **{key}:** {value}")

        if report.get("per_class"):
            lines.extend([
                f"",
                f"## Per-Class Metrics",
                f"",
                f"| Class | Precision | Recall | F1 | Support |",
                f"|-------|-----------|--------|-----|---------|",
            ])

            for cls, vals in report["per_class"].items():
                p = vals.get("precision", 0)
                r = vals.get("recall", 0)
                f1 = vals.get("f1", vals.get("f1_score", 0))
                s = vals.get("support", 0)
                lines.append(f"| {cls} | {p:.4f} | {r:.4f} | {f1:.4f} | {s} |")

        return "\n".join(lines)
