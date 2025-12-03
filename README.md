# Computer-Vision-based Approach to Detect Fatigue Driving

A real-time driver fatigue detection system implemented on NVIDIA Jetson Nano using deep learning and computer vision techniques.

## Project Overview

This project implements a non-invasive fatigue detection system that monitors driver alertness through facial features analysis. The system employs Multi-Task Cascaded Convolutional Neural Networks (MTCNN) for face detection and optimized MobileNetV2 for feature classification, with deployment on NVIDIA Jetson Nano edge computing platform.

### Key Features

- Real-time facial detection and landmark localization
- Multi-index fusion strategy combining:
  - Eye Closure Rate (ECR)
  - Mouth Opening Rate (MOR)
  - Head Non-Positive Face Rate (HNFR)
- Optimized for embedded deployment on Jetson Nano
- Low-latency processing suitable for real-world vehicular applications

## Team Members

| Name | CMS ID | Role | Email |
|------|--------|------|-------|
| Haida Asif | 411800 | Simulation & Algorithms | hasif.bee22seecs@seecs.edu.pk |
| Aleesha Waqar | 417349 | Research & Development | awaqar.bee22seecs@seecs.edu.pk |
| Ayesha Nahman | 424154 | Embedded Systems | anahman.bee22seecs@seecs.edu.pk |

**Course:** CS-477 Computer Vision (Fall 2025)  
**Instructor:** Dr. Tauseef ur Rehman  
**Institution:** National University of Sciences and Technology (NUST)

## System Requirements

### Hardware Requirements

- **Development Platform:** NVIDIA Jetson Nano (4GB recommended)
- **Camera:** CSI camera module or USB webcam
- **Storage:** Minimum 128GB microSD card
- **Power Supply:** 5V 4A power adapter

### Software Requirements

- **Operating System:** Ubuntu 18.04 (JetPack SDK)
- **Python Version:** 3.6+
- **CUDA:** 10.2
- **cuDNN:** 8.0
- **TensorRT:** 7.1

## Installation

### 1. Setup NVIDIA Jetson Nano
Install JetPack SDK on Jetson Nano
### 2. Install System Dependencies
### 3. Install Python Dependencies
# Install MTCNN for face detection
# Install Dlib for facial landmark detection
# Install MobileNetV2 dependencies

### 5. Clone Repository

## Project Structure
 fatigue-detection/  
│  
├── README.md  
├── requirements.txt  
│  
├── DOC/  
│   ├── A_Survey_on_State-of-the-Art_Drowsiness_Detection.pdf 
│   ├── Driver_Fatigue_Detection_System.pdf  
│   ├── Driver_Fatigue_State_Detection.pdf  
│   ├── Fatigue_Driving_Detection_Based.pdf  
│   ├── LITERATURE_REVIEW (1).pdf  
│   ├── cv_review1_logbook.pdf  
│   ├── fatigue-driving-detection-based.pdf  
│   ├── implementation paper (1).pdf  
│   └── survey (1).pdf  
│  
├── Data/  
│   ├── mouth_closed_1/  
│   ├── mouth_closed_2/  
│   ├── mouth_open_1/  
│   ├── mouth_open_2/  
│   ├── dataset_link_1  
│   └── dataset_link_2  
│  
├── Results/  
│  
├── Src/  
│   ├── code/  
│   │   ├── blink_detector.py 
│   │   ├── camera_test.py  
│   │   ├── fatigue_detector.py  
│   │   └── train_mouth.py  
│   │  
│   └── models/  
│       ├── eye_model.h5  
│       └── mouth_model.h5  
   
 

## Usage
### Training Mode

Train the fatigue detection model:
### Inference Mode (Desktop/Development)
### Real-time Detection (Jetson Nano)
Run real-time fatigue detection with camera:

### Camera Interfacing
With CSI camera on Jetson Nano or USB webcam

### 3. Feature Classification (MobileNetV2)
Classifies eye and mouth states using optimized MobileNetV2.

### 4. Multi-Index Fusion
Combines ECR, MOR, and HNFR for comprehensive fatigue assessment.

## Dataset

The system is trained on publicly available facial fatigue datasets:

- Driver drowsiness detection datasets
- Facial expression databases
- Custom collected video sequences

## Troubleshooting

### Common Issues

**Issue 1: CUDA Out of Memory**  
**Issue 2: Camera Not Detected**  
**Issue 3: Slow Inference Speed**  
**Issue 4: Low Detection Accuracy**  

## Model Optimization
Convert trained model to TensorRT for faster inference 

## Documentation

- **Literature Review:** `docs/literature_review.pdf`
- **Project Logbook:** `docs/cv_review1_logbook.pdf`

## License

This project is developed for academic purposes at NUST SEECS.

## Acknowledgments

- **Instructor:** Dr. Tauseef ur Rehman
- **Lab Engineer:** Ms. Tehniyat Siddiqui
- **Teaching Assistant:** Mr. Zahid
- NVIDIA for Jetson Nano platform and development resources
- Research papers and open-source implementations that guided this work

## References

1. Ramzan, M., Khan, H. U., et al. "A survey on state-of-the-art drowsiness detection techniques." IEEE Access, 2019.
2. Sikander, G., & Anwar, S. "Driver fatigue detection systems: a review." IEEE Transactions on Intelligent Transportation Systems, 2018.
3. Jia, H., Xiao, Z., & Ji, P. "Fatigue driving detection based on deep learning and multi-index fusion." IEEE Access, 2021.

## Contact

For project-related queries:

- **Haida Asif:** hasif.bee22seecs@seecs.edu.pk
- **Aleesha Waqar:** awaqar.bee22seecs@seecs.edu.pk
- **Ayesha Nahman:** anahman.bee22seecs@seecs.edu.pk

---

**Course:** CS-477 Computer Vision (Fall 2025)  
**Institution:** National University of Sciences and Technology (NUST)  
**School:** Electrical Engineering and Computer Science (SEECS)
