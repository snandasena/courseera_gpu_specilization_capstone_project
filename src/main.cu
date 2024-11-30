#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <cuda_runtime.h>
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

// Kernel for converting BGR to Grayscale
__global__ void bgrToGrayscaleKernel(const unsigned char *d_bgr, unsigned char *d_gray,
                                     int width, int height, int pitch_bgr,
                                     int pitch_gray)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // Calculate the BGR pixel position
        int bgrIdx = y * pitch_bgr + 3 * x;
        unsigned char b = d_bgr[bgrIdx];
        unsigned char g = d_bgr[bgrIdx + 1];
        unsigned char r = d_bgr[bgrIdx + 2];

        // Calculate the grayscale value
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        // Write to the grayscale image
        d_gray[y * pitch_gray + x] = gray;
    }
}


// Kernel for Sobel operator to calculate gradients
__global__ void sobelKernel(const unsigned char *d_input, unsigned char *d_output, int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        // Sobel kernels for edge detection in the x and y direction
        int gx = -1 * d_input[(y - 1) * pitch + (x - 1)] + 1 * d_input[(y - 1) * pitch + (x + 1)] +
                 -2 * d_input[y * pitch + (x - 1)] + 2 * d_input[y * pitch + (x + 1)] +
                 -1 * d_input[(y + 1) * pitch + (x - 1)] + 1 * d_input[(y + 1) * pitch + (x + 1)];

        int gy = -1 * d_input[(y - 1) * pitch + (x - 1)] + -2 * d_input[(y - 1) * pitch + x] +
                 -1 * d_input[(y - 1) * pitch + (x + 1)] +
                 1 * d_input[(y + 1) * pitch + (x - 1)] + 2 * d_input[(y + 1) * pitch + x] +
                 1 * d_input[(y + 1) * pitch + (x + 1)];

        int magnitude = sqrtf(gx * gx + gy * gy);
        magnitude = min(max(magnitude, 0), 255); // Clamp to [0, 255]
        d_output[y * pitch + x] = static_cast<unsigned char>(magnitude);
    }
}

// Kernel for non-maximum suppression (NMS)
__global__ void
nonMaxSuppression(const unsigned char *d_input, unsigned char *d_output, int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        // Compare the pixel value to its neighbors (8-connectivity)
        if (d_input[y * pitch + x] > d_input[(y - 1) * pitch + x] &&
            d_input[y * pitch + x] > d_input[(y + 1) * pitch + x] &&
            d_input[y * pitch + x] > d_input[y * pitch + (x - 1)] &&
            d_input[y * pitch + x] > d_input[y * pitch + (x + 1)])
        {
            d_output[y * pitch + x] = d_input[y * pitch + x];  // Retain the pixel value
        }
        else
        {
            d_output[y * pitch + x] = 0;  // Set to 0 if it’s not a local maximum
        }
    }
}

// Kernel for applying thresholding
__global__ void applyThreshold(const unsigned char *d_input, unsigned char *d_output, int width, int height, int pitch,
                               int lowThreshold, int highThreshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int value = d_input[y * pitch + x];
        if (value >= highThreshold)
        {
            d_output[y * pitch + x] = 255;  // Strong edge
        }
        else if (value >= lowThreshold)
        {
            d_output[y * pitch + x] = 125;  // Weak edge
        }
        else
        {
            d_output[y * pitch + x] = 0;  // Non-edge
        }
    }
}

// Add logging for time measurement
void logTime(const string &message, const chrono::steady_clock::time_point &start)
{
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << message << " took " << duration << " ms" << endl;
}



// Main function to process video using CUDA
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <video_path>" << endl;

        return -1;
    }

    string videoPath = argv[1];
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened())
    {
        cerr << "Error: Couldn't open video file: " << videoPath << endl;
        return -1;
    }

    cout << "Processing video: " << videoPath << endl;

    const string window_name = "Sobel Edge Detection with CUDA";
    cv::namedWindow(window_name);

    int lowThreshold = 50;   // Low threshold for Canny
    int highThreshold = 150; // High threshold for Canny
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

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
        cv::imshow(window_name, outputFrame);

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