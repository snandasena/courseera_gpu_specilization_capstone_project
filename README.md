
# Sobel Edge Detection using CUDA

This repository contains a CUDA implementation of Sobel Edge Detection on video frames. The program uses CUDA to accelerate the process of converting BGR frames to grayscale, applying the Sobel operator for edge detection, and performing non-maximum suppression and thresholding.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 10.0 or higher)
- OpenCV (version 4.0 or higher)
- C++ compiler (e.g., g++ or clang++)

## Files Overview

- **kernels.cu**: Contains the CUDA kernels for grayscale conversion, Sobel edge detection, non-maximum suppression, and thresholding.
- **main.cpp**: The main program file that processes a video using CUDA acceleration for Sobel edge detection.
- **CMakeLists.txt**: CMake build configuration file.
- **README.md**: Documentation with instructions.

## Compilation

To compile the program, you can use CMake to generate the makefile for your environment.

### Steps to compile:

1. Ensure that you have the CUDA toolkit and OpenCV installed on your machine.

2. Create a build directory inside the project directory:
   ```bash
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```bash
   cmake ..
   ```

4. Compile the project using `make`:
   ```bash
   make
   ```

5. The executable will be generated as `sobel_edge_detection_cuda`.

## Running the Program

Once compiled, you can run the program by providing the path to a video file as a command-line argument. You can also specify the low and high thresholds for the Sobel operator detection.

### Example command:
```bash
./sobel_edge_detection_cuda <video_path> <low_threshold> <high_threshold>
```

### Example with custom thresholds:
```bash
./sobel_edge_detection_cuda /path/to/video.mp4 50 190
```

Where:
- `<video_path>`: Path to the input video file.
- `<low_threshold>`: Low threshold for edge detection (integer).
- `<high_threshold>`: High threshold for edge detection (integer).

If you don't provide `low_threshold` and `high_threshold`, the default values will be used (50 and 190, respectively).

### Stopping the Program
- Press `q` or `ESC` to stop the program and exit.

## Troubleshooting

- If the program fails to start, ensure that you have a CUDA-capable GPU and that CUDA is properly installed.
- Ensure that OpenCV is linked correctly with the project.
- If you encounter issues with CMake, check your CMake version and OpenCV path.

## License

This code is released under the MIT License.
