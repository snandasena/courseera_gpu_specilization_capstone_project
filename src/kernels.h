#include <cuda_runtime.h>

/**
 * Kernel for converting BGR to Grayscale.
 *
 * @param d_bgr       Input BGR image on the device.
 * @param d_gray      Output grayscale image on the device.
 * @param width       Width of the image.
 * @param height      Height of the image.
 * @param pitch_bgr   Pitch of the BGR image.
 * @param pitch_gray  Pitch of the grayscale image.
 */
__global__ void bgrToGrayscaleKernel(const unsigned char *d_bgr, unsigned char *d_gray,
                                     int width, int height, int pitch_bgr,
                                     int pitch_gray)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int bgrIdx = y * pitch_bgr + 3 * x;
        unsigned char b = d_bgr[bgrIdx];
        unsigned char g = d_bgr[bgrIdx + 1];
        unsigned char r = d_bgr[bgrIdx + 2];
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        d_gray[y * pitch_gray + x] = gray;
    }
}

/**
 * Kernel for applying the Sobel operator to compute edge gradients.
 *
 * @param d_input     Input grayscale image on the device.
 * @param d_output    Output image containing edge gradients.
 * @param width       Width of the image.
 * @param height      Height of the image.
 * @param pitch       Pitch of the input/output images.
 */
__global__ void sobelKernel(const unsigned char *d_input, unsigned char *d_output, int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int gx = -1 * d_input[(y - 1) * pitch + (x - 1)] + 1 * d_input[(y - 1) * pitch + (x + 1)] +
                 -2 * d_input[y * pitch + (x - 1)] + 2 * d_input[y * pitch + (x + 1)] +
                 -1 * d_input[(y + 1) * pitch + (x - 1)] + 1 * d_input[(y + 1) * pitch + (x + 1)];

        int gy = -1 * d_input[(y - 1) * pitch + (x - 1)] + -2 * d_input[(y - 1) * pitch + x] +
                 -1 * d_input[(y - 1) * pitch + (x + 1)] +
                 1 * d_input[(y + 1) * pitch + (x - 1)] + 2 * d_input[(y + 1) * pitch + x] +
                 1 * d_input[(y + 1) * pitch + (x + 1)];

        int magnitude = sqrtf(gx * gx + gy * gy);
        magnitude = min(max(magnitude, 0), 255);
        d_output[y * pitch + x] = static_cast<unsigned char>(magnitude);
    }
}

/**
 * Kernel for non-maximum suppression (NMS) in edge detection.
 *
 * @param d_input     Input edge gradient image on the device.
 * @param d_output    Output image after suppression.
 * @param width       Width of the image.
 * @param height      Height of the image.
 * @param pitch       Pitch of the input/output images.
 */
__global__ void nonMaxSuppression(const unsigned char *d_input, unsigned char *d_output, int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        if (d_input[y * pitch + x] > d_input[(y - 1) * pitch + x] &&
            d_input[y * pitch + x] > d_input[(y + 1) * pitch + x] &&
            d_input[y * pitch + x] > d_input[y * pitch + (x - 1)] &&
            d_input[y * pitch + x] > d_input[y * pitch + (x + 1)])
        {
            d_output[y * pitch + x] = d_input[y * pitch + x];
        }
        else
        {
            d_output[y * pitch + x] = 0;
        }
    }
}

/**
 * Kernel for applying thresholding to edge gradients.
 *
 * @param d_input       Input image containing edge gradients.
 * @param d_output      Output binary image after thresholding.
 * @param width         Width of the image.
 * @param height        Height of the image.
 * @param pitch         Pitch of the input/output images.
 * @param lowThreshold  Lower threshold for edge detection.
 * @param highThreshold Higher threshold for strong edges.
 */
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
            d_output[y * pitch + x] = 255;
        }
        else if (value >= lowThreshold)
        {
            d_output[y * pitch + x] = 100;
        }
        else
        {
            d_output[y * pitch + x] = 0;
        }
    }
}

/**
 * Kernel for undistorting an image based on precomputed maps.
 *
 * @param input         Input image in BGR format.
 * @param output        Output undistorted image in BGR format.
 * @param mapX          Precomputed X-coordinate remapping map.
 * @param mapY          Precomputed Y-coordinate remapping map.
 * @param width         Width of the image.
 * @param height        Height of the image.
 * @param pitch_input   Pitch of the input image.
 * @param pitch_output  Pitch of the output image.
 */
__global__ void undistortKernel(const unsigned char* input, unsigned char* output,
                                float* mapX, float* mapY, int width, int height, size_t pitch_input, size_t pitch_output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch_output + x * 3;
        int srcX = mapX[y * width + x];
        int srcY = mapY[y * width + x];

        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
            int srcIdx = srcY * pitch_input + srcX * 3;
            output[idx]     = input[srcIdx];
            output[idx + 1] = input[srcIdx + 1];
            output[idx + 2] = input[srcIdx + 2];
        }
    }
}
