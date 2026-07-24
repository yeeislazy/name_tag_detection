# Methodology

<img width="412" height="432" alt="name_tag_detection drawio" src="https://github.com/user-attachments/assets/9a7464ae-5450-44a0-bcb6-8a4a48458002" />

# Project Structure

```text
./
├── data/
│   ├── fake/                     # Gemini-generated reference images
│   ├── false/                    # Negative samples (without name tag)
│   ├── true/                     # Positive samples (with name tag)
│   ├── raw.csv                   # Raw dataset metadata
│   ├── train.csv                 # Training split
│   └── test.csv                  # Validation/Test split
│
├── debug/
│   └── debug_crop/               # Cropped person images for debugging
│
├── models/
│   ├── best_resnet18.pth         # Final trained ResNet18 model
│   ├── best_resnet18_v1.pth
│   ├── best_resnet18_v2.pth
│   ├── best_resnet18_v3.pth
│   ├── best_resnet18_v4.pth
│   ├── best_resnet18_v5.pth
│   ├── best_resnet18_v6.pth
│   ├── yolo11n.pt
│   └── yolo11s.pt
│
├── notebook/
│   ├── label_csv_gen.ipynb       # Generate train/test CSV labels
│   └── test.ipynb                # Experiments and testing
│
├── src/
│   ├── configuration/
│   │   └── config.py             # Project configuration
│   ├── main/
│   │   └── main.py               # Video inference pipeline
│   ├── others/
│   │   └── name_tag_img_gen.py   # Generate synthetic images using Gemini API
│   └── train/
│       └── train_cnn.py          # Train ResNet18 classifier
│
├── test_videos/
│   ├── input/
│   │   └── sample.mp4            # Input video(s)
│   └── output/
│       ├── sample.mp4            # Annotated output video
│       └── sample.csv            # Staff detection results (frame ID, track ID, bounding box coordinates)
│
├── main.py                       # Project entry point
├── Project Report.docx
├── README.md
├── pyproject.toml
└── uv.lock
```
