import os
import sys
import json
import logging
import numpy as np
import gradio as gr
from pathlib import Path
import cv2
from PIL import Image
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ground_truth_labeler.log")
    ]
)
logger = logging.getLogger(__name__)

# Add project directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Default class list - will be updated when loading experiments
DEFAULT_CLASSES = ["unknown", "fence_person", "sidewalk", "car", "traffic cone"]

class GroundTruthLabeler:
    def __init__(self):
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.current_category = None
        self.current_experiment = None
        self.experiments = {}
        self.images = []
        self.labels = {}
        self.ground_truth = {}
        self.class_list = DEFAULT_CLASSES.copy()
        self.pagination = {
            "current_page": 0,
            "images_per_page": 100,
            "total_pages": 0
        }
        self.selected_images = set()
        self.filter_settings = {
            "class": "all",
            "confidence": 0.0,
            "is_modified": False
        }
        self.selected_as_ground_truth = None
        
    def get_categories(self):
        """Get list of available categories"""
        categories = []
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    categories.append(item.name)
        return categories
    
    def get_experiments(self, category):
        """Get available experiments for a category"""
        self.current_category = category
        experiments = []
        results_dir = self.data_dir / category / "7.results"
        
        if results_dir.exists():
            for item in results_dir.iterdir():
                if item.is_dir():
                    # Look for results.json to confirm it's an experiment
                    if (item / "results.json").exists():
                        experiments.append(item.name)
        
        return experiments
    
    def load_experiment(self, category, experiment):
        """Load experiment data"""
        self.current_category = category
        self.current_experiment = experiment
        
        results_path = self.data_dir / category / "7.results" / experiment / "results.json"
        if not results_path.exists():
            return f"Error: Results file not found at {results_path}"
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            self.experiments[experiment] = results
            
            # Extract unique class names
            classes = set(["unknown"])
            for img_data in results.values():
                if "predicted_class" in img_data:
                    classes.add(img_data["predicted_class"])
            
            self.class_list = sorted(list(classes))
            
            # Prepare image list
            self.images = list(results.keys())
            
            # Initialize labels with predictions
            self.labels = {}
            for img_name, img_data in results.items():
                predicted = img_data.get("predicted_class", "unknown")
                confidence = img_data.get("confidence", 0.0)
                self.labels[img_name] = {
                    "original": predicted,
                    "current": predicted,
                    "confidence": confidence,
                    "is_modified": False
                }
            
            # Update pagination
            self.pagination["total_pages"] = (len(self.images) + self.pagination["images_per_page"] - 1) // self.pagination["images_per_page"]
            self.pagination["current_page"] = 0
            
            # Clear selection
            self.selected_images = set()
            
            return f"Loaded {len(self.images)} images from experiment {experiment}"
        
        except Exception as e:
            logger.error(f"Error loading experiment: {e}")
            return f"Error loading experiment: {str(e)}"
    
    def get_ground_truth_path(self, category):
        """Get path to ground truth file"""
        return self.data_dir / category / "7.results" / "ground_truth.json"
    
    def load_ground_truth(self, category):
        """Load existing ground truth if available"""
        gt_path = self.get_ground_truth_path(category)
        if gt_path.exists():
            try:
                with open(gt_path, 'r') as f:
                    self.ground_truth = json.load(f)
                return f"Loaded {len(self.ground_truth)} ground truth labels"
            except Exception as e:
                logger.error(f"Error loading ground truth: {e}")
                return f"Error loading ground truth: {str(e)}"
        return "No existing ground truth found"
    
    def save_ground_truth(self):
        """Save current labels as ground truth"""
        if not self.current_category or not self.labels:
            return "No data to save"
        
        gt_path = self.get_ground_truth_path(self.current_category)
        
        try:
            # Convert current labels to ground truth format
            ground_truth = {}
            for img_name, label_data in self.labels.items():
                ground_truth[img_name] = label_data["current"]
            
            # Save to file
            with open(gt_path, 'w') as f:
                json.dump(ground_truth, f, indent=2)
            
            self.ground_truth = ground_truth
            return f"Saved {len(ground_truth)} ground truth labels to {gt_path}"
        
        except Exception as e:
            logger.error(f"Error saving ground truth: {e}")
            return f"Error saving ground truth: {str(e)}"
    
    def set_as_ground_truth(self, experiment):
        """Set current experiment as ground truth baseline"""
        if experiment not in self.experiments:
            return f"Error: Experiment {experiment} not loaded"
        
        self.selected_as_ground_truth = experiment
        
        # Update labels based on the selected experiment
        results = self.experiments[experiment]
        for img_name, img_data in results.items():
            predicted = img_data.get("predicted_class", "unknown")
            if img_name in self.labels:
                self.labels[img_name]["current"] = predicted
                self.labels[img_name]["is_modified"] = False
        
        return f"Set experiment {experiment} as ground truth baseline"
    
    def apply_ground_truth_to_labels(self):
        """Apply saved ground truth to current labels"""
        if not self.ground_truth:
            return "No ground truth available"
        
        updated = 0
        for img_name, gt_class in self.ground_truth.items():
            if img_name in self.labels:
                if self.labels[img_name]["current"] != gt_class:
                    self.labels[img_name]["current"] = gt_class
                    self.labels[img_name]["is_modified"] = True
                    updated += 1
        
        return f"Applied ground truth to {updated} images"
    
    def get_current_page_images(self):
        """Get images for the current page with filtering"""
        filtered_images = self.filter_images()
        
        start_idx = self.pagination["current_page"] * self.pagination["images_per_page"]
        end_idx = start_idx + self.pagination["images_per_page"]
        
        return filtered_images[start_idx:end_idx]
    
    def filter_images(self):
        """Apply filters to image list"""
        filtered = []
        for img_name in self.images:
            label_info = self.labels.get(img_name, {})
            
            # Class filter
            if self.filter_settings["class"] != "all" and label_info.get("current") != self.filter_settings["class"]:
                continue
            
            # Confidence filter
            if label_info.get("confidence", 0) < self.filter_settings["confidence"]:
                continue
            
            # Modified filter
            if self.filter_settings["is_modified"] and not label_info.get("is_modified", False):
                continue
            
            filtered.append(img_name)
        
        # Update pagination based on filtered results
        self.pagination["total_pages"] = (len(filtered) + self.pagination["images_per_page"] - 1) // self.pagination["images_per_page"]
        if self.pagination["current_page"] >= self.pagination["total_pages"] and self.pagination["total_pages"] > 0:
            self.pagination["current_page"] = self.pagination["total_pages"] - 1
        
        return filtered
    
    def change_page(self, direction):
        """Change the current page (next/prev)"""
        if direction == "next" and self.pagination["current_page"] < self.pagination["total_pages"] - 1:
            self.pagination["current_page"] += 1
        elif direction == "prev" and self.pagination["current_page"] > 0:
            self.pagination["current_page"] -= 1
        
        return self.pagination["current_page"]
    
    def get_image_path(self, img_name):
        """Get full path to an image"""
        return self.data_dir / self.current_category / "1.images" / img_name
    
    def update_label(self, img_name, new_class):
        """Update label for a single image"""
        if img_name in self.labels:
            original = self.labels[img_name]["current"]
            self.labels[img_name]["current"] = new_class
            self.labels[img_name]["is_modified"] = (new_class != self.labels[img_name]["original"])
            return f"Updated {img_name} from {original} to {new_class}"
        return f"Image {img_name} not found"
    
    def update_selected_labels(self, new_class):
        """Update labels for all selected images"""
        if not self.selected_images:
            return "No images selected"
        
        count = 0
        for img_name in self.selected_images:
            if img_name in self.labels:
                self.labels[img_name]["current"] = new_class
                self.labels[img_name]["is_modified"] = (new_class != self.labels[img_name]["original"])
                count += 1
        
        return f"Updated {count} images to class {new_class}"
    
    def toggle_image_selection(self, img_name):
        """Toggle selection status of an image"""
        if img_name in self.selected_images:
            self.selected_images.remove(img_name)
            return f"Deselected {img_name}"
        else:
            self.selected_images.add(img_name)
            return f"Selected {img_name}"
    
    def clear_selection(self):
        """Clear all selected images"""
        count = len(self.selected_images)
        self.selected_images = set()
        return f"Cleared {count} selected images"
    
    def update_filter(self, filter_type, value):
        """Update filter settings"""
        if filter_type == "class":
            self.filter_settings["class"] = value
        elif filter_type == "confidence":
            self.filter_settings["confidence"] = float(value)
        elif filter_type == "is_modified":
            self.filter_settings["is_modified"] = value
        
        # Reset to first page when filter changes
        self.pagination["current_page"] = 0
        
        return f"Updated {filter_type} filter to {value}"
    
    def get_statistics(self):
        """Get statistics about current labels"""
        stats = {
            "total": len(self.labels),
            "modified": 0,
            "by_class": {}
        }
        
        for class_name in self.class_list:
            stats["by_class"][class_name] = 0
        
        for label_info in self.labels.values():
            current_class = label_info.get("current", "unknown")
            if current_class in stats["by_class"]:
                stats["by_class"][current_class] += 1
            
            if label_info.get("is_modified", False):
                stats["modified"] += 1
        
        return stats
    
    def export_metrics(self):
        """Export metrics comparing experiments to ground truth"""
        if not self.ground_truth:
            return "No ground truth available for metrics"
        
        metrics = {}
        
        for exp_name, exp_data in self.experiments.items():
            correct = 0
            total = 0
            class_metrics = {cls: {"correct": 0, "total": 0} for cls in self.class_list}
            
            for img_name, gt_class in self.ground_truth.items():
                if img_name in exp_data:
                    total += 1
                    predicted = exp_data[img_name].get("predicted_class", "unknown")
                    
                    # Update class-specific metrics
                    if gt_class in class_metrics:
                        class_metrics[gt_class]["total"] += 1
                        if predicted == gt_class:
                            class_metrics[gt_class]["correct"] += 1
                            correct += 1
            
            accuracy = correct / total if total > 0 else 0
            
            # Calculate per-class accuracy
            class_accuracy = {}
            for cls, counts in class_metrics.items():
                class_accuracy[cls] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            
            metrics[exp_name] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "class_accuracy": class_accuracy
            }
        
        # Save metrics to file
        metrics_path = self.data_dir / self.current_category / "7.results" / "metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
        
        return metrics

def create_ui():
    """Create the Gradio UI"""
    labeler = GroundTruthLabeler()
    
    with gr.Blocks(title="Ground Truth Labeling Tool") as app:
        gr.Markdown("# Ground Truth Labeling Tool")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Category and experiment selection
                category_dropdown = gr.Dropdown(
                    choices=labeler.get_categories(),
                    label="Select Category",
                    interactive=True
                )
                experiment_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Experiment",
                    interactive=True
                )
                load_btn = gr.Button("Load Experiment")
                load_result = gr.Textbox(label="Status", lines=2)
                
                # Ground truth management
                gr.Markdown("### Ground Truth Management")
                set_gt_dropdown = gr.Dropdown(
                    choices=[],
                    label="Set as Ground Truth Baseline",
                    interactive=True
                )
                set_gt_btn = gr.Button("Set as Ground Truth")
                load_gt_btn = gr.Button("Load Existing Ground Truth")
                save_gt_btn = gr.Button("Save Current Labels as Ground Truth")
                gt_result = gr.Textbox(label="Ground Truth Status", lines=2)
                
                # Filters
                gr.Markdown("### Filters")
                filter_class = gr.Dropdown(
                    choices=["all"] + labeler.class_list,
                    label="Filter by Class",
                    value="all",
                    interactive=True
                )
                filter_confidence = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Minimum Confidence"
                )
                filter_modified = gr.Checkbox(
                    label="Show Modified Only",
                    value=False
                )
                apply_filter_btn = gr.Button("Apply Filters")
                
                # Statistics
                gr.Markdown("### Statistics")
                stats_text = gr.Textbox(label="Statistics", lines=6)
                refresh_stats_btn = gr.Button("Refresh Statistics")
                export_metrics_btn = gr.Button("Export Metrics")
                
            with gr.Column(scale=3):
                # Main labeling interface
                gr.Markdown("### Image Gallery")
                
                # Pagination controls
                with gr.Row():
                    prev_btn = gr.Button("← Previous Page")
                    page_info = gr.Textbox(label="Page", value="0 / 0")
                    next_btn = gr.Button("Next Page →")
                
                # Batch actions
                with gr.Row():
                    batch_class = gr.Dropdown(
                        choices=labeler.class_list,
                        label="Assign Selected To",
                        interactive=True
                    )
                    apply_batch_btn = gr.Button("Apply to Selected")
                    clear_selection_btn = gr.Button("Clear Selection")
                
                # Image gallery with selection
                gallery = gr.Gallery(
                    label="Images",
                    columns=5,
                    rows=5,
                    show_label=True,
                    object_fit="contain",
                    height="600px"
                ).style(grid=5, height="600px")
                
                # Single image view
                with gr.Row():
                    with gr.Column(scale=2):
                        selected_image = gr.Image(label="Selected Image", type="filepath")
                    with gr.Column(scale=1):
                        image_info = gr.Textbox(label="Image Info", lines=5)
                        single_class = gr.Dropdown(
                            choices=labeler.class_list,
                            label="Assign Class",
                            interactive=True
                        )
                        apply_single_btn = gr.Button("Apply")
        
        # Event handlers
        def update_experiments(category):
            experiments = labeler.get_experiments(category)
            return gr.Dropdown(choices=experiments)
        
        def load_experiment_handler(category, experiment):
            result = labeler.load_experiment(category, experiment)
            
            # Update experiment dropdown for ground truth selection
            experiments = list(labeler.experiments.keys())
            
            # Update class dropdown for filters and batch actions
            classes = ["all"] + labeler.class_list
            
            # Update gallery
            gallery_data = update_gallery()
            
            return [
                result,
                gr.Dropdown(choices=experiments),
                gr.Dropdown(choices=labeler.class_list),
                gr.Dropdown(choices=labeler.class_list),
                gr.Dropdown(choices=classes, value="all"),
                *gallery_data
            ]
        
        def update_gallery():
            # Get current page images
            current_images = labeler.get_current_page_images()
            
            # Prepare gallery data
            gallery_images = []
            for img_name in current_images:
                img_path = labeler.get_image_path(img_name)
                if os.path.exists(img_path):
                    label_info = labeler.labels.get(img_name, {})
                    current_class = label_info.get("current", "unknown")
                    confidence = label_info.get("confidence", 0.0)
                    is_modified = label_info.get("is_modified", False)
                    
                    # Create label text with visual indicators
                    modified_marker = "✏️" if is_modified else ""
                    selected_marker = "✓" if img_name in labeler.selected_images else ""
                    label = f"{img_name}\n{current_class} ({confidence:.2f})\n{modified_marker}{selected_marker}"
                    
                    gallery_images.append((str(img_path), label))
            
            # Update page info
            page_text = f"{labeler.pagination['current_page'] + 1} / {max(1, labeler.pagination['total_pages'])}"
            
            return [gallery_images, page_text]
        
        def change_page_handler(direction):
            labeler.change_page(direction)
            return update_gallery()
        
        def apply_filters_handler():
            # Apply filters and update gallery
            return update_gallery()
        
        def update_filter_handler(filter_type, value):
            labeler.update_filter(filter_type, value)
            return update_gallery()
        
        def select_image_handler(evt: gr.SelectData):
            """Handle image selection from gallery"""
            if not labeler.images or not labeler.get_current_page_images():
                return ["No image selected", None, ""]
            
            # Get the selected image from the current page
            try:
                idx = evt.index
                current_page_images = labeler.get_current_page_images()
                if idx < len(current_page_images):
                    img_name = current_page_images[idx]
                    img_path = labeler.get_image_path(img_name)
                    
                    # Toggle selection for batch operations
                    labeler.toggle_image_selection(img_name)
                    
                    # Get label info
                    label_info = labeler.labels.get(img_name, {})
                    current_class = label_info.get("current", "unknown")
                    original_class = label_info.get("original", "unknown")
                    confidence = label_info.get("confidence", 0.0)
                    is_modified = label_info.get("is_modified", False)
                    
                    info_text = (
                        f"Filename: {img_name}\n"
                        f"Current class: {current_class}\n"
                        f"Original class: {original_class}\n"
                        f"Confidence: {confidence:.4f}\n"
                        f"Modified: {'Yes' if is_modified else 'No'}\n"
                        f"Selected: {'Yes' if img_name in labeler.selected_images else 'No'}"
                    )
                    
                    # Update gallery to show selection status
                    gallery_data = update_gallery()
                    
                    return [info_text, str(img_path), current_class, *gallery_data]
            except Exception as e:
                logger.error(f"Error selecting image: {e}")
                return ["Error selecting image", None, "", *update_gallery()]
        
        def update_single_label(img_path, new_class):
            if not img_path:
                return ["No image selected", *update_gallery()]
            
            img_name = os.path.basename(img_path)
            result = labeler.update_label(img_name, new_class)
            
            # Update image info after label change
            label_info = labeler.labels.get(img_name, {})
            current_class = label_info.get("current", "unknown")
            original_class = label_info.get("original", "unknown")
            confidence = label_info.get("confidence", 0.0)
            is_modified = label_info.get("is_modified", False)
            
            info_text = (
                f"Filename: {img_name}\n"
                f"Current class: {current_class}\n"
                f"Original class: {original_class}\n"
                f"Confidence: {confidence:.4f}\n"
                f"Modified: {'Yes' if is_modified else 'No'}\n"
                f"Selected: {'Yes' if img_name in labeler.selected_images else 'No'}"
            )
            
            return [result, info_text, *update_gallery()]
        
        def update_batch_labels(new_class):
            result = labeler.update_selected_labels(new_class)
            return [result, *update_gallery()]
        
        def clear_selection_handler():
            result = labeler.clear_selection()
            return [result, *update_gallery()]
        
        def ground_truth_handler(action, experiment=None):
            if action == "set" and experiment:
                result = labeler.set_as_ground_truth(experiment)
            elif action == "load":
                result = labeler.load_ground_truth(labeler.current_category)
                result += "\n" + labeler.apply_ground_truth_to_labels()
            elif action == "save":
                result = labeler.save_ground_truth()
            else:
                result = "Invalid action"
            
            return [result, *update_gallery()]
        
        def update_statistics():
            stats = labeler.get_statistics()
            text = (
                f"Total images: {stats['total']}\n"
                f"Modified labels: {stats['modified']} ({stats['modified']/stats['total']*100:.1f}%)\n\n"
                "Class distribution:\n"
            )
            
            for cls, count in stats['by_class'].items():
                if count > 0:
                    text += f"- {cls}: {count} ({count/stats['total']*100:.1f}%)\n"
            
            return text
        
        def export_metrics_handler():
            metrics = labeler.export_metrics()
            if isinstance(metrics, str):
                return metrics
            
            # Format metrics as text
            text = "Experiment Metrics (compared to ground truth):\n\n"
            
            for exp_name, exp_metrics in metrics.items():
                acc = exp_metrics["accuracy"] * 100
                text += f"{exp_name}: {acc:.2f}% accuracy ({exp_metrics['correct']}/{exp_metrics['total']})\n"
                
                # Class-specific accuracy
                text += "  Class accuracy:\n"
                for cls, cls_acc in exp_metrics["class_accuracy"].items():
                    if cls_acc > 0:
                        text += f"  - {cls}: {cls_acc*100:.2f}%\n"
                text += "\n"
            
            return text
        
        # Connect event handlers
        category_dropdown.change(
            fn=update_experiments,
            inputs=category_dropdown,
            outputs=experiment_dropdown
        )
        
        load_btn.click(
            fn=load_experiment_handler,
            inputs=[category_dropdown, experiment_dropdown],
            outputs=[
                load_result,
                set_gt_dropdown,
                batch_class,
                single_class,
                filter_class,
                gallery,
                page_info
            ]
        )
        
        prev_btn.click(
            fn=lambda: change_page_handler("prev"),
            outputs=[gallery, page_info]
        )
        
        next_btn.click(
            fn=lambda: change_page_handler("next"),
            outputs=[gallery, page_info]
        )
        
        filter_class.change(
            fn=lambda value: update_filter_handler("class", value),
            inputs=filter_class,
            outputs=[gallery, page_info]
        )
        
        filter_confidence.change(
            fn=lambda value: update_filter_handler("confidence", value),
            inputs=filter_confidence,
            outputs=[gallery, page_info]
        )
        
        filter_modified.change(
            fn=lambda value: update_filter_handler("is_modified", value),
            inputs=filter_modified,
            outputs=[gallery, page_info]
        )
        
        apply_filter_btn.click(
            fn=apply_filters_handler,
            outputs=[gallery, page_info]
        )
        
        gallery.select(
            fn=select_image_handler,
            outputs=[image_info, selected_image, single_class, gallery, page_info]
        )
        
        apply_single_btn.click(
            fn=update_single_label,
            inputs=[selected_image, single_class],
            outputs=[load_result, image_info, gallery, page_info]
        )
        
        apply_batch_btn.click(
            fn=update_batch_labels,
            inputs=batch_class,
            outputs=[load_result, gallery, page_info]
        )
        
        clear_selection_btn.click(
            fn=clear_selection_handler,
            outputs=[load_result, gallery, page_info]
        )
        
        set_gt_btn.click(
            fn=lambda exp: ground_truth_handler("set", exp),
            inputs=set_gt_dropdown,
            outputs=[gt_result, gallery, page_info]
        )
        
        load_gt_btn.click(
            fn=lambda: ground_truth_handler("load"),
            outputs=[gt_result, gallery, page_info]
        )
        
        save_gt_btn.click(
            fn=lambda: ground_truth_handler("save"),
            outputs=[gt_result, gallery, page_info]
        )
        
        refresh_stats_btn.click(
            fn=update_statistics,
            outputs=stats_text
        )
        
        export_metrics_btn.click(
            fn=export_metrics_handler,
            outputs=stats_text
        )
        
    return app

def main():
    app = create_ui()
    app.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main() 