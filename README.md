# Plant Root Segmentation & Robotics Integration

This project was developed as part of a university DataLab client project at **Breda University of Applied Sciences**, combining **computer vision**, **deep learning**, and **robotics control (PID/RL)** to automate the detection and inoculation of primary root tips in Petri dishes.

> **Client Need**: Automate root tip inoculation using AI and robotics to increase throughput and consistency in plant science experiments.

---

## 🔧 Folder Overview

### `segmentation/`
**Purpose**: Implements traditional and deep learning segmentation workflows from scratch.

- **Tasks Covered**:
  - **Task 5**: Model training (`U-Net`)
  - **Task 6**: Instance segmentation
  - **Task 7**: RSA (Root System Architecture) extraction
  - **Task 8**: Full pipeline + Kaggle submission
- **Key Files**:
  - `001.ipynb`, `002.ipynb`, `003.ipynb`: Step-by-step pipelines
  - `predictions/`: Final mask predictions per image
  - `skeletonized_predictions/`: Individual root-level masks, RSA, root lengths
  - `submissions/root_predictions_first_submission.csv`: Kaggle submission

### `root_prediction/`
**Purpose**: Focused deep learning module for inference and model training.

- **Key Files**:
  - `training.ipynb`: Trains the final U-Net model
  - `inference.ipynb`: Takes raw input, returns predicted mask
  - `unet_model_128px.h5`: Trained model file used across other folders

### `robotics/`
**Purpose**: Integrates the CV pipeline with robotics simulation using both RL and PID controllers.

- **Tasks Covered**:
  - **Task 13**: RL-based robot control for inoculating root tips
  - **Task 15**: Final presentation materials
- **Subfolders**:
  - `cv_integration/`:
    - `final_cv_pipeline.py`: Final integrated pipeline
    - `sim_class.py`, `rl_controller.py`, `pid_controller.py`: Robot simulation & control logic
    - `model_files/`: Contains both segmentation and RL models
    - `meshes/`, `textures/`, `.urdf` files: Robot visual setup for simulation
  - `evidencing/`:
    - `RL_showcase.gif`: Shows robot executing inoculation

> **Note**: Folder structure was preserved to avoid import path issues. All dependencies assume relative paths within this hierarchy.

![Final cv pipeline](robotics/evidencing//RL_showcase.gif)

### `benchmarking/`
**Purpose**: Compare the performance of the PID vs RL controller in terms of inoculation speed and accuracy.

- **Task Covered**:
  - **Task 14**: Benchmarking
- **Key Files**:
  - `benchmark.py`: Measures controller performance
  - `performance_dashboard.png/svg`: Visual summary of benchmark results

### `data/`
**Purpose**: Sample image/mask data for quick debugging or testing model inference.

### `presentation/`
**Purpose**: Final block presentation showcasing the full integrated solution.

- **File**:
  - `final_presentation.pptx`

---

## 🧪 Core Deliverables & Their Locations

| Task     | Description                                 | Key File(s)                                                   |
|----------|---------------------------------------------|---------------------------------------------------------------|
| Task 5   | Model Training                              | `root_prediction/training.ipynb`, `unet_model_128px.h5`       |
| Task 6   | Individual Root Segmentation                | `segmentation/skeletonized_predictions/`                      |
| Task 7   | RSA Extraction                               | `segmentation/skeletonized_predictions/`, `.txt` root tips    |
| Task 8   | Full Pipeline & Kaggle Submission           | `segmentation/submissions/`, `003.ipynb`                      |
| Task 13  | RL Control + CV Integration                 | `robotics/cv_integration/final_cv_pipeline.py`, `RL_showcase.gif` |
| Task 14  | Controller Benchmarking                     | `benchmarking/benchmark.py`, `*.png` dashboards               |
| Task 15  | Final Presentation                          | `presentation/final_presentation.pptx`                        |

---

## 🚀 Highlights

- ✅ End-to-end computer vision to robotics pipeline
- 🌟 Primary root tip detection and inoculation
- 🤖 RL and PID control simulation
- 📊 Controller performance benchmarking
- 🎓 Developed for a real university client challenge

---

## 🏫 Educational Context

This project was developed during Block B of the Artificial Intelligence & Data Science program at **Breda University of Applied Sciences**, under a client-driven challenge to automate root inoculation in high-throughput plant science research.

---

## 🔄 How to Run

> Coming soon (ask if you'd like to include setup instructions, environments, or usage commands).

