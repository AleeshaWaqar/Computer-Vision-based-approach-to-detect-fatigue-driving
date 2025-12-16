![CI Tests](https://github.com/AleeshaWaqar/Computer-Vision-based-approach-to-detect-fatigue-driving/workflows/Fatigue%20Detection%20CI/badge.svg)

# Computer-Vision-based Approach to Detect Fatigue Driving

A real-time driver fatigue detection system implemented on NVIDIA Jetson Nano using deep learning and computer vision techniques.

## Project Overview

This project implements a non-invasive fatigue detection system that monitors driver alertness through facial features analysis. The system employs Multi-Task Cascaded Convolutional Neural Networks (MTCNN) for face detection, Convolutional Neural Networks (CNN) for feature classification, and combines multiple fatigue indicators including Eye Aspect Ratio (EAR), PERCLOS, Mouth Aspect Ratio (MAR), and head pose estimation for comprehensive fatigue assessment.

### Key Features

- Real-time facial detection and landmark localization using MTCNN
- Multi-modal fatigue detection combining:
  - Eye Closure Detection (EAR & PERCLOS)
  - Yawn Detection (MAR-based)
  - Head Pose Estimation (PnP algorithm)
- Deep learning-based classifiers for eye and mouth states
- Rule-based multi-modal fusion for robust fatigue detection
- Optimized for embedded deployment on NVIDIA Jetson Nano
- Low-latency processing suitable for real-world vehicular applications

## Algorithms Used

### Core Detection Algorithms

1. **MTCNN (Multi-Task Cascaded Convolutional Neural Network)**
   - Real-time face detection and facial keypoint extraction
   - Detects eyes, nose, and mouth landmarks with high accuracy

2. **CNN – Eye State Classifier**
   - Classifies cropped eye images into open or closed states
   - Enables accurate blink detection and eye closure monitoring

3. **CNN – Yawn Detector**
   - Classifies cropped mouth images into yawn or no-yawn classes
   - Robust yawn detection independent of lighting conditions

### Fatigue Metrics

4. **EAR (Eye Aspect Ratio)**
   - Geometric ratio measuring eye openness
   - Detects blinks and prolonged eye closures

5. **PERCLOS (Percentage of Eye Closure)**
   - Temporal metric measuring percentage of time eyes remain closed
   - Primary fatigue indicator recognized by transportation safety standards 

6. **MAR (Mouth Aspect Ratio)**
   - Geometric ratio detecting mouth opening
   - Works robustly across different lighting conditions 

### Supporting Algorithms

7. **Temporal Thresholding / Frame Counting**
   - Detects prolonged eye closure or mouth opening
   - Tracks blink duration and yawn duration over consecutive frames

8. **Head Pose Estimation (PnP – Perspective-n-Point)**
   - Estimates 3D head orientation (pitch, yaw, roll)
   - Detects head nodding and abnormal head positions
   - Uses facial landmarks for 6DOF pose estimation

9. **Rule-Based Multi-Modal Fusion**
   - Combines eye metrics, mouth activity, and head pose
   - Makes final fatigue decision based on weighted criteria
   - Reduces false positives through multi-index validation

10. **Real-Time Video Processing Pipeline (OpenCV)**
    - Handles frame capture, preprocessing, and visualization
    - Optimized inference pipeline for embedded systems
    - Real-time overlay of detection results

## Team Members

| Name | Role | Email |
|------|------|-------|
| Haida Asif | Simulation & Algorithms | hasif.bee22seecs@seecs.edu.pk |
| Aleesha Waqar | Research & Development | awaqar.bee22seecs@seecs.edu.pk |
| Ayesha Nahman | Embedded Systems | anahman.bee22seecs@seecs.edu.pk |

**Course:** CS-477 Computer Vision (Fall 2025)  
**Instructor:** Dr. Tauseef ur Rehman  
**Institution:** National University of Sciences and Technology (NUST)

## System Requirements

### Hardware Requirements

- **Development Platform:** NVIDIA Jetson Nano (4GB recommended)
- **Camera:** CSI camera module or USB webcam
- **Storage:** 128GB USB drive
- **Power Supply:** 5V 4A power adapter

### Software Requirements

- **Operating System:** Ubuntu 18.04 (JetPack SDK)
- **Python Version:** 3.6+
- **CUDA:** 10.2
- **cuDNN:** 8.0
- **TensorRT:** 7.1 
```
 
