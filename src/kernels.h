#include <cuda_runtime.h>


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
