from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model

# Train the model with copy_paste augmentation
model.train(
    data='/home/ml/project-agi/data/test_category/8.refine-dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    copy_paste=0.3,  # Enable copy_paste augmentation with 0.3 probability
    save_period=10,
    patience=20,
    device=0,
    project='runs/segment',
    name='augmented_train'
) 