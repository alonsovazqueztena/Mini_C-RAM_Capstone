# AIegis Beam

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Hardware Setup](#hardware-setup)
- [Model Training](#model-training)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)
- [Citations and Acknowledgements](#citations-and-acknowledgements)

---

## Project Overview
![Poster](readme_images/poster.png)

A real-time counter-drone system using computer vision and object tracking. Detects UAVs in video streams and simulates countermeasures (target locking).

### Demo Videos
- [Alonso Vazquez Tena](https://youtube.com/shorts/53AWf_9uEzw?feature=shared)
- [Daniel Saravia](https://youtube.com/shorts/bJcraph1RIk?feature=shared)

---

## Key Features
- Real-time object detection using YOLO12
- Centroid-based object tracking
- Frame processing pipeline (1920x1080 @ 120FPS)
- Configurable detection thresholds and tracking parameters

---

## System Architecture
```
Mini_C-RAM_Capstone$ tree -I cap/
.
├── LICENSE
├── notebooks
│   └── ai_model_training.ipynb
├── readme_images
│   ├── confusion_matrix_normalized.png
│   ├── confusion_matrix.png
│   ├── F1_curve.png
│   ├── labels_correlogram.jpg
│   ├── labels.jpg
│   ├── P_curve.png
│   ├── poster.png
│   ├── PR_curve.png
│   ├── R_curve.png
│   └── results.png
├── README.md
├── requirements.txt
├── src
│   ├── ai_model_interface.py
│   ├── dmx_control.py
│   ├── DMX_frame_pipeline.py
│   ├── drone_detected.mp3
│   ├── drone_detector_12l.pt
│   ├── drone_detector_12m.pt
│   ├── drone_detector_12n.pt
│   ├── drone_detector_12s.pt
│   ├── drone_detector_12x.pt
│   ├── frame_pipeline.py
│   ├── __init__.py
│   ├── main.py
│   ├── __pycache__
│   │   ├── ai_model_interface.cpython-312.pyc
│   │   ├── detection_processor.cpython-312.pyc
│   │   ├── DMX_frame_pipeline.cpython-312.pyc
│   │   ├── frame_pipeline.cpython-312.pyc
│   │   ├── frame_processor.cpython-312.pyc
│   │   ├── tracking_system.cpython-312.pyc
│   │   └── video_stream_manager.cpython-312.pyc
│   ├── qlight_workspace.qxw
│   ├── run.py
│   ├── run.sh
│   ├── tracking_system.py
│   └── video_stream_manager.py
├── test
│   ├── __init__.py
│   └── test_ai.py
└── test_images
7 directories, 58 files
```

NOTE: drone_detector_12x.pt could not be added to the repository due to its size (exceeds 100MB). [Download](https://drive.google.com/file/d/1yzFKtHaEQzx3OuTVzEUAEHYwTYP83IO0/view?usp=drive_link)

---

## Installation

### Prerequisites
Ensure all these are installed:
- Python 3.11.9
- PIP 25.0.1
- Windows 11 or Ubuntu 20.04
- Nvidia GPU
- CUDA Toolkit 12.6
- USB Webcam or IP Camera

Follow these steps to ensure the program can be executed:
```bash
# Clone repository
git clone https://github.com/alonsovazqueztena/Mini_C-RAM_Capstone.git
cd Mini_C-RAM_Capstone

# Create virtual environment
python -m venv env
source env/bin/activate      # Linux/MacOS
source env/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Basic Operation
To only execute the object detection and tracking software, run this:
```bash
# Execute the program using the source code (must be in src folder)
python main.py
python3 main.py # Or also this.
python3 run.py if set up with hardware(Moving Headlight, Enttec Open DMX USB, DMX cables, Phone Web Cam)
```

### User Control
For keyboard control, follow these commands: 
| Key   | Description            |
|-------|------------------------|
| q     | Quit system            |
| Up    | Move system up         |
| Down  | Move system down       |
| Left  | Move system left       |
| Right | Move system right      |

To toggle between manual and automatic mode, press on the manual mode button on the graphical user interface (GUI) (the rightmost icon on the window).

For Xbox controller control, follow these commands:
| Button       | Description            |
|--------------|------------------------|
| B            | Quit system            |
| Y            | Toggle manual/automatic mode           |
| Leftmost Analog Stick | Move system freely (any direction)     |

---

## Hardware Setup
**Camera Connection**

To set up a USB phone webcam, do the following:

1. Install Iriun Webcam (can be for either Android or iPhone) for both the computer and the phone.
2. Setup and ensure Iriun Webcam is running on both the phone and computer.
3. Plug a USB cable into the phone and computer.
4. Ensure that Iriun Webcam is receiving your video frame.

### QLC+ for Moving Head Light Control

This project utilizes QLC+ to control stage lighting with a moving head fixture. The setup uses industry-standard hardware and a custom DMX configuration to deliver professional results.

### Hardware Components

- **U`King LED Moving Head Light 25W DJ Lights Stage Lighting**  
  The primary lighting fixture for dynamic stage effects.

- **Enttec Open DMX USB**  
  The DMX interface used for communication between your computer and the lighting hardware.

- **DMX Cables**  
  Standard cables to connect your DMX interface to the lighting fixture.

### Software Requirements

- **QLC+**  
  The official lighting control software running on Linux.
  
- **Custom Workspace File (`qlight_workspace.qxw`)**  
  Pre-configured workspace with DMX settings tailored to your setup.

### Installation Steps for Linux

QLC+ is distributed as a Debian package (.deb) for Linux systems. Follow these steps to install and start QLC+:

1. **Download the QLC+ Debian Package**  
   Visit the [official QLC+ website](https://qlcplus.org/) and download the appropriate Debian package (e.g., `qlcplus_x.y.x.deb`).

2. **Install QLC+**  
   Open a terminal, navigate to the directory containing the downloaded package, and run:

   ```bash
   sudo dpkg -i qlcplus_x.y.x.deb
   ```

   This command installs QLC+ on your system.

3. **Launch QLC+ with Your Custom Workspace**  
   To start QLC+ using your configuration, execute:

   ```bash
   qlcplus -w -o qlight_workspace.qxw
   ```

   - The `-w` flag launches QLC+ in windowed mode.
   - The `-o` flag specifies the custom workspace file to load.

### Verifying Your Setup

Once QLC+ is running, verify that your DMX setup is functioning by opening a web browser and navigating to:

```
http://localhost:9999/
```

If the web interface loads, your DMX configuration is active and the system is ready for control.

### Additional Configuration Tips

- **Check DMX Addressing:**  
  Ensure that the DMX addresses configured in QLC+ match those on your U`King moving head light.

- **Hardware Connections:**  
  Confirm that the Enttec Open DMX USB interface is connected correctly and that all DMX cables are securely attached.

- **Further Resources:**  
  For more detailed configuration or troubleshooting, refer to the [QLC+ Documentation](https://qlcplus.org/docs.html).

---

## Model Training
### General Steps
1. The Junyper notebook used to train the YOLO12 models is found within the notebooks folder:

```
.
├── notebooks
│   └── ai_model_training.ipynb       # AI model training notebook
```

2. Use **Google Colab** to ensure proper functionality and avoidance of dependency issues.

3. Ensure that CUDA is installed to allow for proper leveraging of the GPU if available.

```bash
!nvidia-smi # Check Nvidia GPU.
import torch # Use GPU for AI training.
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count()) 
```

4. Replace the following file paths:

```bash
!ls Insert Google Drive path here # Check drive access.
```

```bash
!rm -rf Insert Google Drive path here # Remove to ensure mounting.
```

```bash
from google.colab import drive # Import drive again.
drive.mount('Insert Google Drive path here') # Mount drive.
```

```bash
!ls "Insert images dataset path here" # Check image dataset access.
```

```bash
%load_ext tensorboard # Load Tensorboard.
%tensorboard --logdir insert/directory/to/runs/here # Execute Tensorboard.
```

```bash
tensorboard_log_dir = "Insert Tensorboard log directory path here" # Tensorboard log directory.
```

```bash
model = YOLO("Insert your YOLO model directory path here") # Load YOLO model.
```

```bash
train_results = model.train(
        data="Insert your image dataset YAML file path here",
        epochs=100, imgsz=640, device="cuda", save=True, save_period=10,
        project=tensorboard_log_dir, name=f"session(insert-name)"
        ) # Train YOLO model (YAML file, epochs, image size, GPU or CPU, allowed saving, save period, log, run name).
```

5. Execute each cell from top to bottom, one at a time.

6. To check live results of the AI model training, examine the Tensorboard server run in this cell:
```bash
%load_ext tensorboard # Load Tensorboard.
%tensorboard --logdir insert/directory/to/runs/here # Execute Tensorboard.
```

### Expected Results
- **Tensorboard logs directory**: Every run will be its own subdirectory. In it, the following will be contained:

  1. Weights folder (every epoch weight, last weight, best weight)

  2. Validation batches (predictions and labels)

  3. Train batches

  4. Metric results (Graphs and CSV): box loss (train and validation), cls loss (train and validation), dfl loss (train and validation), precision (B), recall (B), mAP50 (B), and mAP50-95 (B)

  5. Curve graphs: Precision, recall,  precision-recall, and F1

  6. Label graphs (regular and correlogram)

  7. Confusion matrix graphs (regular and normalized)

  8. Training arguments YAML file

### Drone Detection AI Model Results
The following results were achieved with the final YOLO12m model used for this project:

- **Metric Graphs:**

![Metric result graphs](readme_images/results.png)

- **Recall-Confidence Curve:**

![R curve graph](readme_images/R_curve.png)

- **Precision-Recall Curve:**

![PR curve graph](readme_images/PR_curve.png)

- **Precision-Confidence Curve:**

![P curve graph](readme_images/P_curve.png)

- **Label Graphs:**

![Labels graph](readme_images/labels.jpg)

- **Label Correlogram:**

![Labels correlogram](readme_images/labels_correlogram.jpg)

- **F1 Curve:**

![F1 curve](readme_images/F1_curve.png)

- **Confusion Matrix:**

![Confusion matrix](readme_images/confusion_matrix.png)

- **Confusion Matrix (Normalized):**

![Confusion matrix normalized](readme_images/confusion_matrix_normalized.png)

---

## Testing
### Module Testing

To test the following modules:
- **Video Stream Manager**
- **AI Model Interface**
- **Tracking System**
- **Frame Pipeline**

Run:

```bash
cd src # Ensure you are in the correct directory.
python main.py # Run all main module tests of the program.
```

### AI Model Testing

1. The AI testing script is found within the test folder:

```
.
├── test
│   └── test_ai.py                    # Detection test for AI model
```

2. Update this filepath to any of the YOLO model filepaths:
```bash
model = YOLO("..\src\drone_detector_12n.pt") # Create model instance.
```

3. Update this filepath to your test image filepath:
```bash
results = model.predict("..\\test_images\drone_real_test_10.jpg", conf=0.5, imgsz=640, show=True, save=True) # Run inference (confidence, image size, display, save prediction).
```

4. To test the AI model, run:

```bash
cd test # Ensure you are in the correct directory.

# Run a sign language detection test on an image (must be in JPG format).
python test_ai.py   # Results in runs folder
```

5. A runs folder will be created or used if present. Within it, a predict folder will be created for every prediction made on an image. The image will have the AI predictions labelled:
```
.
└── runs
    └── predict 
        └── processed_test_image.jpg  # Detection test for AI model
```

---

## Troubleshooting
### ERROR: No frames available.
Ensure that all the capture device indexes match to your capture device (0 if its an internal webcam, 1 if its an external webcam such as an Iriun webcam or GoPro).

video_stream_manager.py:
```bash
def __init__(self, capture_device=0, max_queue_size=10): # Update the capture device index.
```

### Testing Different AI Models
Ensure that the filepaths for the AI model are updated to the desired AI model's filepath:

run.py:
```bash
def main():
    # Configure logging with timestamps and log levels.
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG to show all log messages.
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        pipeline = DMXFramePipeline(
            model_path="drone_detector_12n.pt", # Update this filepath.
            confidence_threshold=0.5
        ) # Initialize DMX pipeline with model path and confidence threshold.
```
frame_pipeline.py:
```bash
class FramePipeline:
    """Pipeline: captures frames, runs AI detections, tracks objects, displays results."""

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5): # Update this filepath.
```
DMX_frame_pipeline.py:
```bash
class DMXFramePipeline(FramePipeline):
    """
    DMXFramePipeline now uses a state machine to manage mode transitions:
      - MANUAL: User-controlled via keyboard/joystick (AI detection is still shown, but DMX values are not updated automatically).
      - LOCKED: Drone is detected; the beam locks on.
      - HOLD: Detections have just dropped out; continue holding the lock.
      - SCANNING: No detection for a while; resume scanning from last offset.
    """

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5): # Update this filepath
        super().__init__(model_path, confidence_threshold) # Initialize base class
```
ai_model_interface.py:
```bash
class AIModelInterface:
    """Optimized interface for YOLO drone detection."""
    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5, audio_path="drone_detected.mp3"): # Update the first filepath.
```
test_ai.py:
```bash
model = YOLO("..\src\drone_detector_12n.pt") # Create model instance, update this filepath.
```

---

## Contributing

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/new-tracker
```
3. Add tests for new functionality
4. Submit a pull request

### Coding Standards
- Complete concise commenting
- Docstrings for all modules
- 80%+ test coverage

---

## License
MIT License - See LICENSE for details

---

## Maintainers
- Alonso Vazquez Tena: AI Engineer
- Daniel Saravia: System Integration Engineer

**Mentor**: Ryan Woodward  

---

## Citations and Acknowledgements
**YOLO12**  
```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

@software{yolo12,
  author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
  title = {YOLOv12: Attention-Centric Real-Time Object Detectors},
  year = {2025},
  url = {https://github.com/sunsmarterjie/yolov12},
  license = {AGPL-3.0}
}
```

**Image Dataset**
```bibtex
@misc{uavs-vqpqt_dataset,
  title = { UAVs Dataset },
  type = { Open Source Dataset },
  author = { UAVS },
  howpublished = { \url{ https://universe.roboflow.com/uavs-7l7kv/uavs-vqpqt } },
  url = { https://universe.roboflow.com/uavs-7l7kv/uavs-vqpqt },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2024 },
  month = { dec },
  note = { visited on 2025-03-26 },
}
```
