/*
 * Sobel Edge Detection with CUDA and Lens Distortion Correction
 *
 * This program processes a video file to perform Sobel edge detection using CUDA, while also applying distortion correction.
 * The output combines the original frame and the edge-detected frame vertically for display.
 *
 * Features:
 * - Camera lens distortion correction using OpenCV's `initUndistortRectifyMap`.
 * - Sobel edge detection implemented with CUDA kernels.
 * - Dynamic memory management for GPU and host data.
 * - Performance logging for various processing steps.
 *
 * Usage:
 *   ./sobel_edge_detection <video_path>
 *
 * Dependencies:
 * - OpenCV (highgui, imgproc, calib3d modules)
 * - CUDA Toolkit
 */

#include "kernels.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            return -1; \
        } \
    } while (0)

using namespace std;
using namespace cv;


/**
 * Precompute remap coordinates for undistortion.
 *
 * @param mapX Output X-coordinate map for remapping.
 * @param mapY Output Y-coordinate map for remapping.
 * @param imageSize Size of the input image (width, height).
 * @param cameraMatrix Camera intrinsic matrix.
 * @param distCoeffs Distortion coefficients.
 */
void precomputeRemap(cv::Mat &mapX, cv::Mat &mapY, cv::Size imageSize, const cv::Mat &cameraMatrix,
                     const cv::Mat &distCoeffs)
{
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
                                cameraMatrix, imageSize, CV_32FC1, mapX, mapY);
}

/**
 * Log the elapsed time for a specific operation.
 *
 * @param message Description of the operation.
 * @param start Start time point.
 */
void logTime(const string &message, const chrono::steady_clock::time_point &start)
{
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << message << " took " << duration << " ms" << endl;
}

// CMD parser
std::tuple<int, int> parseThresholds(int argc, char *argv[]) {
    // Default threshold values
    int lowThreshold = 50;
    int highThreshold = 190;

    // If thresholds are provided in the command-line arguments, use them
    if (argc >= 4) {
        try {
            // Parse thresholds from the arguments
            lowThreshold = std::stoi(argv[2]);
            highThreshold = std::stoi(argv[3]);

            // Ensure the thresholds are within valid ranges
            if (lowThreshold < 0 || lowThreshold > 255 || highThreshold < 0 || highThreshold > 255) {
                std::cerr << "Error: Thresholds must be between 0 and 255." << std::endl;
                return std::make_tuple(-1, -1);  // Return error values
            }

            // Ensure highThreshold is greater than lowThreshold
            if (highThreshold <= lowThreshold) {
                std::cerr << "Error: highThreshold must be greater than lowThreshold." << std::endl;
                return std::make_tuple(-1, -1);  // Return error values
            }
        }
        catch (const std::invalid_argument &e) {
            std::cerr << "Error: Invalid threshold values provided. Please provide valid integers." << std::endl;
            return std::make_tuple(-1, -1);  // Return error values
        }
        catch (const std::out_of_range &e) {
            std::cerr << "Error: Threshold values are out of range." << std::endl;
            return std::make_tuple(-1, -1);  // Return error values
        }
    }

    // Return the thresholds as a tuple
    return std::make_tuple(lowThreshold, highThreshold);
}



// Main function to process video using CUDA
int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> [lowThreshold] [highThreshold]" << std::endl;
        return -1;
    }

    int lowThreshold;   // Low threshold for Sobel
    int highThreshold ; // High threshold for Sobel

    std::tie(lowThreshold, highThreshold) = parseThresholds(argc, argv);


    string videoPath = argv[1];
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened())
    {
        cerr << "Error: Couldn't open video file: " << videoPath << endl;
        return -1;
    }


    cout << "Processing video: " << videoPath << endl;

    const string window_name = "Sobel Edge Detection with CUDA";

    cv::namedWindow(window_name, cv::WINDOW_NORMAL); // Allow resizing
    cv::resizeWindow(window_name, 1280, 720); // Set initial size
    cv::moveWindow(window_name, 100, 100); // Move to position (100, 100)


    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));


    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 535.4, 0, 320.1,
            0, 539.2, 247.6,
            0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.2469, 0.1092, 0, 0, -0.002);

    // Precompute remapping coordinates
    cv::Mat mapX, mapY;
    precomputeRemap(mapX, mapY, cv::Size(frameWidth, frameHeight), cameraMatrix, distCoeffs);

    // Allocate GPU memory for remap coordinates
    float *d_mapX, *d_mapY;
    CUDA_CHECK(cudaMalloc(&d_mapX, mapX.total() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mapY, mapY.total() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_mapX, mapX.ptr<float>(), mapX.total() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mapY, mapY.ptr<float>(), mapY.total() * sizeof(float), cudaMemcpyHostToDevice));


    size_t pitch_bgr, pitch_gray;
    unsigned char *d_bgr = nullptr, *d_gray = nullptr, *d_output = nullptr, *d_result = nullptr;
    unsigned char *h_result = new unsigned char[frameWidth * frameHeight];

    // Logging memory allocation
    auto start = chrono::steady_clock::now();
    cout << "Allocating GPU memory..." << endl;

    CUDA_CHECK(cudaMallocPitch(&d_bgr, &pitch_bgr, 3 * frameWidth * sizeof(unsigned char), frameHeight));
    CUDA_CHECK(cudaMallocPitch(&d_gray, &pitch_gray, frameWidth * sizeof(unsigned char), frameHeight));
    CUDA_CHECK(cudaMallocPitch(&d_output, &pitch_gray, frameWidth * sizeof(unsigned char), frameHeight));
    CUDA_CHECK(cudaMallocPitch(&d_result, &pitch_gray, frameWidth * sizeof(unsigned char), frameHeight));

    logTime("Memory allocation", start);

    // Define CUDA block and grid sizes
    dim3 block(16, 16);
    dim3 grid((frameWidth + block.x - 1) / block.x, (frameHeight + block.y - 1) / block.y);


    cv::Mat frame;
    while (true)
    {
        start = chrono::steady_clock::now();

        cap >> frame;
        if (frame.empty()) break;

        cout << "Processing frame..." << endl;


        // Copy input frame to GPU
        CUDA_CHECK(cudaMemcpy2D(d_bgr, pitch_bgr, frame.data, frame.step, 3 * frameWidth, frameHeight,
                                cudaMemcpyHostToDevice));

        // Apply undistortion kernel
        undistortKernel<<<grid, block>>>(d_bgr, d_gray, d_mapX, d_mapY, frameWidth, frameHeight, pitch_bgr, pitch_gray);
        CUDA_CHECK(cudaDeviceSynchronize());



        // Copy the BGR frame to GPU memory
        CUDA_CHECK(cudaMemcpy2D(d_bgr, pitch_bgr, frame.data, frame.step, 3 * frameWidth, frameHeight,
                                cudaMemcpyHostToDevice));
        logTime("Copying frame to GPU", start);

        start = chrono::steady_clock::now();
        // Step 1: Convert BGR to grayscale using CUDA kernel
        bgrToGrayscaleKernel<<<grid, block>>>(d_bgr, d_gray, frameWidth, frameHeight, pitch_bgr, pitch_gray);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        logTime("Grayscale conversion", start);

        start = chrono::steady_clock::now();
        // Step 2: Sobel edge detection kernel (gradient calculation)
        sobelKernel<<<grid, block>>>(d_gray, d_output, frameWidth, frameHeight, pitch_gray);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Non-maximum suppression kernel
        nonMaxSuppression<<<grid, block>>>(d_output, d_result, frameWidth, frameHeight, pitch_gray);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 4: Apply thresholds to get final edge detection result
        applyThreshold<<<grid, block>>>(d_result, d_output, frameWidth, frameHeight, pitch_gray, lowThreshold,
                                        highThreshold);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        logTime("Sobel kernel", start);

        start = chrono::steady_clock::now();
        // Copy the result back to the host
        CUDA_CHECK(cudaMemcpy2D(h_result, frameWidth, d_output, pitch_gray, frameWidth, frameHeight,
                                cudaMemcpyDeviceToHost));
        logTime("Copying result to CPU", start);

        // Create a Mat object to display the result
        cv::Mat outputFrame(frameHeight, frameWidth, CV_8UC1, h_result);

        // Convert grayscale output to BGR for concatenation
        cv::Mat frameColor;
        cv::cvtColor(frame, frameColor, cv::COLOR_RGB2GRAY);

        // Stack frames vertically (original on top, edge-detected below)
        cv::Mat combinedFrame;
        cv::vconcat(frameColor, outputFrame, combinedFrame);

        // Show the combined frame
        cv::imshow(window_name, combinedFrame);


        auto key = (char) waitKey(30);
        if (key == 'q' || key == 27)
        {
            cout << "Processing stopped by user. Exiting..." << endl;
            break;
        }
    }


    // Cleanup
    CUDA_CHECK(cudaFree(d_bgr));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_result));
    delete[] h_result;

    cout << "Cleanup completed. Goodbye!" << endl;

    return 0;
}