
# GPU Specialization Capstone Project

This project implements a CUDA-based pipeline for real-time video processing, specifically focusing on Canny edge detection and GPU-accelerated computations. The project is part of the Coursera GPU Specialization Capstone.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [License](#license)

---

## Overview

The **GPU Specialization Capstone Project** demonstrates the use of CUDA for optimizing video processing tasks. It showcases the implementation of key steps in the Canny edge detection algorithm using CUDA kernels, replacing traditional CPU-based computations for improved performance.

The pipeline includes:
1. **Color Conversion**: Converts input frames from BGR to Grayscale using a CUDA kernel.
2. **Edge Detection**: Applies the Sobel operator for gradient computation.
3. **Non-Maximum Suppression**: Filters out non-edge pixels.
4. **Thresholding**: Performs edge detection with user-defined thresholds.

---

## Features

- GPU-accelerated video processing using CUDA.
- Modular CUDA kernels for:
  - Color space conversion.
  - Sobel gradient computation.
  - Non-maximum suppression.
  - Thresholding.
- Real-time video frame processing with OpenCV integration.
- Highly configurable through input arguments and threshold values.

---

## Technologies

The project leverages the following technologies:

- **CUDA**: GPU programming for high-performance computing.
- **OpenCV**: For video input/output and basic image handling.
- **C++**: Core programming language for the application.

---

## Setup and Installation

### Prerequisites

1. **CUDA Toolkit**: Ensure CUDA is installed on your system and your GPU supports it.
   - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. **OpenCV**: Install OpenCV with C++ bindings.
   - On Ubuntu: `sudo apt install libopencv-dev`
3. **CMake**: Required for building the project.

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/snandasena/courseera_gpu_specilization_capstone_project.git
   cd courseera_gpu_specilization_capstone_project
   ```

2. Create a build directory and compile the project:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. The compiled executable will be available in the `bin/` directory.

---

## Usage

### Running the Application

1. Provide the video file path as input:
   ```bash
   ./bin/main.exe path/to/video.mp4
   ```

2. For real-time logging and output redirection:
   ```bash
   make run ARGS="path/to/video.mp4"
   ```

3. Output logs will be saved in `outputs.log`.

### Command Line Arguments

- **Input Video Path**: Specify the path to the video file to be processed.
- **Thresholds**: Modify `lowThreshold` and `highThreshold` values within the code for different edge detection sensitivity.

---

## Code Structure

```
.
├── CMakeLists.txt            # Build configuration
├── src/
│   ├── main.cpp              # Main program entry point
│   └── kernels.h             # Cuda Kernels
├── inputs/                   # Sample video files (placeholder)
├── logs/                     # Output directory for logs
└── README.md                 # Project documentation
```

### Key Components

- **CUDA Kernels**:
  - `bgrToGrayscaleKernel`: Converts BGR to grayscale.
  - `sobelKernel`: Computes image gradients using Sobel filters.
  - `nonMaxSuppression`: Suppresses non-maximum gradient values.
  - `applyThreshold`: Applies thresholding for edge classification.

- **Main Pipeline**:
  - Captures video frames.
  - Processes each frame using CUDA kernels.
  - Displays processed output in real time.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software, provided that you include proper attribution.

--- 

### Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements or bug fixes.

---

For any issues or questions, please open an [issue](https://github.com/snandasena/courseera_gpu_specilization_capstone_project/issues).

Enjoy coding! 🚀
