# Mini C-RAM Counter-Drone System

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Hardware Setup](#hardware-setup)
- [Simulation Mode](#simulation-mode)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
A real-time counter-drone system using computer vision and object tracking. Detects UAVs in video streams and simulates countermeasures (laser activation). Designed for Raspberry Pi with optional hardware integration.

## Key Features
- Real-time object detection using YOLOv5
- Centroid-based object tracking
- Frame processing pipeline (640x480 @ 15FPS)
- Hardware control interface for laser systems
- Simulation mode for development without hardware
- Configurable detection thresholds and tracking parameters

## System Architecture
```
.
├── control_output_manager.py   # Laser control interface (GPIO/PWM)
├── detection_processor.py      # Filters/processes YOLO detections
├── frame_pipeline.py           # Main processing workflow
├── frame_processor.py          # Frame resizing/normalization
├── main.py                     # System entry point
├── tracking_system.py          # Object tracking implementation
├── video_stream_manager.py     # Camera/stream input handling
├── yolo_model_interface.py     # YOLO model wrapper
└── config.yaml                 # System configuration
```

## Installation

### Prerequisites
- Python 3.8+
- Raspberry Pi OS (Bullseye) or Ubuntu 20.04
- USB Webcam or IP Camera

```bash
# Clone repository
git clone https://github.com/alonsovazqueztena/Mini_C-RAM_Capstone.git
cd Mini_C-RAM_Capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml`:

```yaml
camera:
  device: 0              # /dev/video0
  resolution: [640, 480] # Input resolution
  fps: 30                # Target FPS

detection:
  model: yolo_epoch_100.pt
  confidence: 0.65       # Minimum detection confidence
  classes: [0]           # COCO class IDs (0: person, etc.)

tracking:
  max_disappeared: 30    # Frames to keep lost objects
  max_distance: 50       # Pixel distance for ID matching

hardware:
  laser_pin: 18          # GPIO pin for laser control
  safe_mode: true        # Disable physical outputs
```

## Usage

### Basic Operation
```bash
# Start system with default config
python main.py
```

#### Keyboard Control
| Key | Description            |
|-----|------------------------|
| q   | Quit system            |
| p   | Pause processing       |
| d   | Toggle debug overlay   |

### Advanced Modes
```bash
# Use test image instead of camera
python main.py --simulate

# Specify custom configuration
python main.py --config custom_config.yaml
```

## Hardware Setup

### Hardware Diagram

**Camera Connection**
- **USB Webcam**: Plug into available USB port
- **IP Camera**: Set RTSP URL in `config.yaml`

**Laser Control**
- Connect laser module to GPIO 18
- Power: 5V PWM compatible laser diode

**Safety Measures**
- Always enable `safe_mode` during development
- Use current-limiting resistor for laser

## Simulation Mode
Test without hardware using stored images:

```python
# In video_stream_manager.py
SIMULATION_MODE = True
IMAGE_PATH = "drone_test.jpg"
```

## Testing
Run validation tests:

```bash
python -m unittest discover -s tests/

# Individual component tests
python test.py --test detection
python test.py --test tracking
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

## Citations & Acknowledgements
**YOLOv11n**  
> Glenn Jocher and Jing Qiu.  
> *Ultralytics YOLO11, version 11.0.0 (2024)*  
> [GitHub Repository](https://github.com/ultralytics/ultralytics)  
> ORCID: 0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069  
> License: AGPL-3.0