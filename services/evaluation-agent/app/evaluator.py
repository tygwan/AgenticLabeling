"""Model evaluation implementation."""
import numpy as np
from typing import Any, Dict, List, Optional


class ModelEvaluator:
    """Evaluates models and calculates metrics."""

    def evaluate_detection(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluate object detection predictions.

        Args:
            predictions: List of {boxes, labels, scores} per image
            ground_truth: List of {boxes, labels} per image
            iou_threshold: IoU threshold for positive match

        Returns:
            Dict with mAP, precision, recall, etc.
        """
        all_precisions = []
        all_recalls = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = np.array(pred.get("boxes", []))
            gt_boxes = np.array(gt.get("boxes", []))

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue

            if len(gt_boxes) == 0:
                false_positives += len(pred_boxes)
                continue

            if len(pred_boxes) == 0:
                false_negatives += len(gt_boxes)
                continue

            # Calculate IoU matrix
            iou_matrix = self._compute_iou_matrix(pred_boxes, gt_boxes)

            # Match predictions to ground truth
            matched_gt = set()
            for i in range(len(pred_boxes)):
                best_iou = 0
                best_gt_idx = -1

                for j in range(len(gt_boxes)):
                    if j not in matched_gt and iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_gt_idx = j

                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1

            false_negatives += len(gt_boxes) - len(matched_gt)

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "iou_threshold": iou_threshold,
        }

    def _compute_iou_matrix(
        self, boxes1: np.ndarray, boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU between all pairs of boxes."""
        # boxes format: [x1, y1, x2, y2] or [x, y, w, h]
        n1, n2 = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])

        return iou_matrix

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        # Convert to x1, y1, x2, y2 if needed
        if len(box1) == 4:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)

            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

            union = area1 + area2 - intersection

            return intersection / (union + 1e-10)
        return 0.0

    def evaluate_classification(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> Dict[str, Any]:
        """Evaluate classification predictions."""
        pred_labels = [p.get("class", p.get("label")) for p in predictions]
        gt_labels = [g.get("class", g.get("label")) for g in ground_truth]

        correct = sum(p == g for p, g in zip(pred_labels, gt_labels))
        total = len(pred_labels)

        # Per-class metrics
        classes = set(gt_labels)
        per_class = {}

        for cls in classes:
            tp = sum(p == g == cls for p, g in zip(pred_labels, gt_labels))
            fp = sum(p == cls and g != cls for p, g in zip(pred_labels, gt_labels))
            fn = sum(p != cls and g == cls for p, g in zip(pred_labels, gt_labels))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)

            per_class[str(cls)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / (precision + recall + 1e-10)),
                "support": sum(g == cls for g in gt_labels),
            }

        return {
            "accuracy": correct / total if total > 0 else 0,
            "total": total,
            "correct": correct,
            "per_class": per_class,
        }

    def evaluate_segmentation(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> Dict[str, Any]:
        """Evaluate segmentation predictions."""
        total_iou = 0
        total_dice = 0
        count = 0

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

            total_iou += iou
            total_dice += dice
            count += 1

        return {
            "mIoU": float(total_iou / count) if count > 0 else 0,
            "dice_coefficient": float(total_dice / count) if count > 0 else 0,
            "samples_evaluated": count,
        }

    def evaluate_model(
        self,
        model_path: str,
        dataset_path: str,
        task: str = "detection",
    ) -> Dict[str, Any]:
        """Evaluate a trained YOLO model on a dataset."""
        from ultralytics import YOLO

        model = YOLO(model_path)
        results = model.val(data=dataset_path)

        metrics = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        }

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
                score = model_metrics.get(metric, 0)
                if score > best_scores[metric][1]:
                    best_scores[metric] = (path, score)

        return {
            "models": results,
            "best": {m: path for m, (path, _) in best_scores.items()},
        }

    def generate_report(
        self,
        evaluation_id: str,
        format: str = "json",
        include_visualizations: bool = False,
    ) -> Dict[str, Any]:
        """Generate evaluation report."""
        # Placeholder - would load saved evaluation results
        return {
            "evaluation_id": evaluation_id,
            "format": format,
            "message": "Report generation placeholder",
        }
