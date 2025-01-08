# Boundary Detection and CIFAR10 Classification

This repository is divided into two phases, focusing on classical and modern computer vision techniques:

---

## Phase 1: Probability of Boundary (Pb) Detection

This phase implements a boundary detection pipeline to identify probable edges in images using traditional computer vision techniques.

### Key Steps:
- **Feature Extraction**: Compute feature maps for Texton, Brightness, and Color.
- **Gradient Computation**: Generate gradient maps (Tg, Bg, Cg) for the extracted features.
- **Boundary Map Combination**: Combine gradients to produce a final Probability of Boundary (Pb) map.

### How to Run:
1. Navigate to `/Phase1/Code`.
2. Run the script `Wrapper.py`.
3. View results in `/Phase1/Results`:
   - **Texton, Brightness, Color Maps**: Feature maps.
   - **Gradient Maps (Tg, Bg, Cg)**: Gradients of features.
   - **Pb_lite**: Final boundary detection maps.

---

## Phase 2: CIFAR10 Image Classification

This phase involves testing and evaluating deep learning models for image classification on the CIFAR10 dataset.

### Key Components:
- **Dataset**: CIFAR10 dataset with 10 object categories.
- **Models**: Various neural network architectures defined in `Network_torch.py`.
- **Outputs**: Accuracy plots, confusion matrices, and trained model checkpoints.

### How to Run:
1. Navigate to `/Phase2/Code`.
2. Edit `Train_torch.py` or `Test_torch.py` to select the desired model (de-comment relevant lines).
3. Run the respective script for training or testing.
4. Results are saved as performance metrics and visualizations.

---

## Repository Structure

```
.
├── Phase1
│   ├── Code
│   │   └── Wrapper.py
│   ├── Results
│       ├── Texton
│       ├── Brightness
│       ├── Color
│       ├── Tg
│       ├── Bg
│       ├── Cg
│       └── Pb_lite
├── Phase2
│   ├── CIFAR10
│   ├── Checkpoints
│   ├── Code
│   │   ├── Train_torch.py
│   │   ├── Test_torch.py
│   │   └── Network_torch.py
```

---

## Key Features
- **Phase 1**: Classical image processing for edge detection with interpretable results.
- **Phase 2**: Deep learning-based classification with modular and extensible code.

---

## Getting Started
1. Clone the repository.
2. Install required dependencies listed in `requirements.txt`.
3. Follow the instructions for each phase to generate results.

