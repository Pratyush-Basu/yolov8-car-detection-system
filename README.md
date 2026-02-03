Car Detection System — YOLOv8 Fine-Tuning & Evaluation
# Overview

This project implements a car detection and localization system using a pretrained YOLOv8 model fine-tuned on a road-camera style dataset. The objective was to design, evaluate and analyze an object detection pipeline with a focus on model performance, error analysis and post-processing improvements.

The system detects vehicles in images, draws bounding boxes and evaluates detection quality using both quantitative metrics and structured failure analysis.

# Approach
Model Selection

Used YOLOv8 Nano (yolov8n) for fast convergence and reduced overfitting on a limited dataset

Initialized with pretrained weights (COCO dataset)

Fine-tuned on custom car detection dataset

Dataset Preparation

Converted bounding boxes from pixel format (xmin, ymin, xmax, ymax) to YOLO normalized format

Organized into YOLO-compatible structure:

images/train, images/val
labels/train, labels/val


Split labeled data into 80% training and 20% validation

Training Configuration

Image size: 640

Epochs: 30

Batch size: 16

Light augmentations (flip, scale, brightness, translation)

Optimization based on validation mAP trends

# Evaluation Methodology

Model performance was evaluated using:

Precision

Recall

mAP@0.5

mAP@0.5:0.95

Loss convergence monitoring

Additionally, a custom object-level error analysis module was implemented using IoU-based matching between predictions and ground truth.

# Error Analysis Module (Add-On Contribution)

Each predicted bounding box was categorized into:

Category	Description
Correct Detection	IoU ≥ 0.5 with ground truth
Missed Detection	Ground truth with no matching prediction
Poor Localization	Prediction with IoU < 0.5
False Positive	Detection without ground truth
Results (Validation Set)
Correct detections: 114  
Missed cars: 5  
Poor localization: 4  
False positives: 3  

Key Insights

Majority of vehicles were correctly detected

Most errors occurred in small or distant vehicles

Localization drift occurred in crowded scenes

Very low false positive rate after post-processing

# Post-Processing Logic (Add-On Contribution)

Confidence threshold tuning was applied to improve bounding box quality:

Confidence	False Positives	Localization Errors
0.25	9	10
0.45	3	4
Final threshold used: 0.45

This significantly reduced noisy detections while preserving high recall.

# Training & Evaluation Enhancements

Dedicated validation split for unbiased evaluation

Continuous monitoring of loss and mAP curves to ensure training stability

Validation-driven confidence tuning for improved inference reliability

# Inference Demo

A lightweight Streamlit application allows:

Uploading an image

Running car detection

Visualizing bounding boxes

python -m streamlit run app.py

📁 Project Structure
├── notebook.ipynb        # training, evaluation, error analysis
├── app.py               # Streamlit demo
├── best.pt              # fine-tuned model
├── README.md
└── yolo_data/
    ├── images/
    └── labels/

⚠️ Limitations

Model performs best on road-camera style imagery similar to training distribution

Performance decreases on visually distinct scenes due to domain shift

Small dataset size limits extreme generalization

🔮 Potential Improvements

Multi-domain training with diverse viewpoints

Backbone freezing experiments to improve robustness

Larger labeled dataset

Temporal video-based detection

✅ Conclusion

The fine-tuned YOLOv8 model achieves strong detection accuracy on road-camera imagery with clean bounding box localization. Error analysis and post-processing enhancements significantly improved output quality while maintaining high recall.

The project demonstrates a complete ML workflow including data preparation, model fine-tuning, evaluation, failure analysis, and deployment-ready inference.