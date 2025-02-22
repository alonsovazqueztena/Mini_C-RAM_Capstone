# Mini C-RAM System

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Hardware Setup](#hardware-setup)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)
- [Citations and Acknowledgements](#citations-and-acknowledgements)


## Project Overview
A real-time counter-drone system using computer vision and object tracking. Detects UAVs in video streams and simulates countermeasures (laser activation).

## Key Features
- Real-time object detection using YOLOv11n (or any other compatible AI model)
- Centroid-based object tracking
- Frame processing pipeline (1920x1080 @ 120FPS)
- Hardware control interface for laser systems
- Configurable detection thresholds and tracking parameters

## System Architecture
```
.
├── images
│   ├── drone_mock_test_1.jpg         # Sample mock test image
│   ├── drone_mock_test_2.jpg         # Sample mock test image
│   ├── drone_mock_test_3.jpg         # Sample mock test image
│   ├── drone_real_test_1.jpg         # Sample real test image
│   ├── drone_real_test_2.jpg         # Sample real test image
│   ├── drone_real_test_3.jpg         # Sample real test image
│   ├── drone_real_test_4.jpg         # Sample real test image
│   ├── drone_real_test_5.jpg         # Sample real test image
│   ├── drone_real_test_6.jpg         # Sample real test image
│   ├── drone_real_test_7.jpg         # Sample real test image
│   ├── drone_real_test_8.jpg         # Sample real test image
│   ├── drone_real_test_9.jpg         # Sample real test image
│   ├── drone_real_test_10.jpg        # Sample real test image
│   ├── drone_real_test_11.jpg        # Sample real test image
│   ├── drone_real_test_12.jpg        # Sample real test image
│   ├── drone_real_test_13.jpg        # Sample real test image
│   ├── drone_real_test_14.jpg        # Sample real test image
│   └── drone_real_test_15.jpg        # Sample real test image
├── src
│   ├── ai_model_interface.py         # AI model wrapper
│   ├── control_output_manager.py     # Laser control interface (GPIO/PWM)
│   ├── detection_processor.py        # Filters/processes AI detections
│   ├── drone_detector_ai.pt          # YOLOv11n drone detector model
│   ├── frame_pipeline.py             # Main processing workflow
│   ├── frame_processor.py            # Frame resizing/normalization
│   ├── main.py                       # Program execution code
│   ├── tracking_system.py            # Object tracking implementation
│   └── video_stream_manager.py       # Camera/stream input handling
└── test
    ├── test_ai_model_interface.py    # Unit test for AI model interface
    ├── test_ai.py                    # Detection test for AI model
    ├── test_detection_processor.py   # Unit test for detection processor
    ├── test_frame_processor.py       # Unit test for frame processor
    ├── test_tracking_system.py       # Unit test for tracking system
    └── test_video_stream_manager.py  # Unit test for video stream manager
```

## Installation

### Prerequisites
- Python 3.11.1+
- Windows 11 or Ubuntu 20.04
- USB Webcam or IP Camera

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

## Usage

### Basic Operation
```bash
# Execute the program using the source code (must be in src folder)
python main.py
```

#### Keyboard Control
| Key | Description            |
|-----|------------------------|
| q   | Quit system            |
| p   | Pause processing       |
| d   | Toggle debug overlay   |

## Hardware Setup
**Camera Connection**
- **USB Webcam**: Plug into available USB port
- **IP Camera**: Set RTSP URL

**Laser Control**
- Connect laser module to GPIO 18
- Power: 5V PWM compatible laser diode

## Testing
Run validation tests in test folder:

```bash
# Run all unit tests of the program.
pytest

# Run an AI model detection test of image.

python test_ai.py   # Results in runs folder
```

## Contributing

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/new-tracker
```
3. Add tests for new functionality
4. Submit a pull request

### Coding Standards
- PEP8 compliance
- Type hints for public methods
- Docstrings for all modules
- 80%+ test coverage

## License
MIT License - See LICENSE for details

## Maintainers
- Alonso Vazquez Tena  
- Daniel Saravia  

**Mentor**: Ryan Woodward  
*University of Advanced Robotics, 2023*

## Citations and Acknowledgements
**YOLOv11n**  
> Glenn Jocher and Jing Qiu.  
> *Ultralytics YOLO11, version 11.0.0 (2024)*  
> [GitHub Repository](https://github.com/ultralytics/ultralytics)  
> ORCID: 0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069  
> License: AGPL-3.0