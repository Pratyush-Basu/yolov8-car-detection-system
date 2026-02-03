Car Detection System — YOLOv8 Fine-Tuning & Evaluation
# Overview

This project implements a car detection and localization system using a pretrained YOLOv8 Nano model fine-tuned on a road-camera style dataset. The objective was to design, evaluate and analyze an object detection pipeline with emphasis on performance, failure understanding and prediction quality improvements.

The system detects vehicles in images, visualizes bounding boxes and evaluates detection quality using both standard metrics and structured error analysis.

# Approach
Model Selection

YOLOv8 Nano (yolov8n) selected for efficiency and reduced overfitting on a limited dataset

Initialized with pretrained COCO weights

Fine-tuned on a custom car detection dataset

Dataset Preparation

Converted bounding boxes from (xmin, ymin, xmax, ymax) format to YOLO normalized format

Organized into YOLO-compatible structure:

images/train, images/val  
labels/train, labels/val


80% training / 20% validation split

Training Configuration

Image size: 640

Epochs: 30

Batch size: 16

Light augmentation (horizontal flip, scale jitter, brightness variation, translation)

Training monitored using validation mAP and loss curves

# Evaluation Methodology

Model performance was assessed using:

Precision

Recall

mAP@0.5

mAP@0.5:0.95

Loss convergence

Additionally, an object-level error analysis module matched predictions to ground truth using IoU thresholds.

# Add-On Contributions (Mandatory Extensions)
# Option 1 — Error Analysis Module

Each prediction was categorized as:

Category	Description
Correct Detection	IoU ≥ 0.5 with ground truth
Missed Detection	Ground truth without prediction
Poor Localization	IoU < 0.5
False Positive	Detection without ground truth
Validation Results

Correct detections: 116
Missed cars: 5
Poor localization: 6
False positives: 6

Insights

Most vehicles were correctly detected

Errors occurred primarily for small, distant or partially occluded cars

Crowded scenes produced most localization drift

# Option 3 — Post-Processing Logic

Confidence threshold tuning was applied to improve bounding box quality.

Confidence	False Positives	Localization Errors
0.25 -	9	- 9
0.35 -	6	- 6
Final threshold used: 0.35

This reduced noisy detections by ~33% with only a negligible recall decrease.

# Option 4 — Lightweight Demo

A Streamlit interface enables:

Uploading an image

Running car detection

Visualizing bounding boxes

Run with:

python -m streamlit run app.py

# Option 5 — Training & Evaluation Enhancements

Dedicated validation split for unbiased evaluation

Continuous monitoring of mAP and loss for training stability

Validation-based confidence tuning for reliable inference

# Backbone Freezing Experiment (Exploratory)

An additional experiment froze the first 10 backbone layers during fine-tuning.

Observations:

Improved robustness on visually different real-world images

Slight reduction in dataset-specific accuracy

This demonstrated a generalization–specialization trade-off.

# Project Structure
├── notebook
├   ├──car_detection_pipeline.ipynb        # training, evaluation, error analysis
├── app.py               # Streamlit demo
├── best.pt              # fine-tuned model
├── README.md
└── data/

# Mandatory Analysis
1. Assumptions

Single-class detection (car only)

Accurate bounding box annotations

Similar distribution between training and validation data

Confidence threshold of 0.35 balances precision and recall

2. Most Common Errors

Localization drift on small or distant vehicles

Missed detections in crowded scenes

3. Why These Errors Occur

Limited dataset diversity

Scale variation of vehicles

Partial occlusion in traffic frames

4. Trade-Offs Between Simplicity and Accuracy

YOLOv8 Nano was chosen for efficiency and faster training over larger YOLO variants

Increasing the confidence threshold from 0.25 to 0.35 reduced false positives at the cost of a small increase in missed detections

5. Future Improvements

Experiment with freezing backbone layers to preserve pretrained visual features and improve generalization

Train on more diverse datasets and viewpoints

Apply stronger augmentation strategies

Extend detection to video sequences

Multi-domain training

# Limitations

Best performance on road-camera style imagery

Reduced generalization on visually distinct scenes due to domain shift

Small dataset restricts extreme robustness

# Results & Visual Evaluation
Detection Performance Example

<h3>Detection Results</h3>
<img src="https://github.com/Pratyush-Basu/yolov8-car-detection-system/blob/main/results/results.png" width="700" alt="Car Detection Results">

This figure shows predicted bounding boxes on validation images after post-processing confidence tuning.

<h3>Confusion Matrix</h3>
<img src="https://github.com/Pratyush-Basu/yolov8-car-detection-system/blob/main/results/confusion_matrix.png" width="700" alt="Confusion Matrix">

This summarizes correct detections, missed vehicles, localization errors and false positives.

# Potential Improvements

Multi-domain training with diverse viewpoints

Backbone freezing experiments to improve robustness

Larger labeled dataset

Temporal video-based detection

# Conclusion

The fine-tuned YOLOv8 model achieves strong detection accuracy on road-camera imagery with clean bounding box localization. Error analysis and post-processing enhancements significantly improved output quality while maintaining high recall.

The project demonstrates a complete ML workflow including data preparation, model fine-tuning, evaluation, failure analysis, and deployment-ready inference.
