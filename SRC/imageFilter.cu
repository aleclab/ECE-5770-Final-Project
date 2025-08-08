#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "wb.h"

#include <stdio.h>

/* cuda standard libraries */
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
    bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
        if (abort)
            exit(code);
    }
}

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define MASK_WIDTH (5)
#define RADIUS     (MASK_WIDTH / 2)
#define TILE_WIDTH (16)


#define w (TILE_WIDTH + MASK_WIDTH - 1)
#define clampf(x) (fminf(fmaxf((x), 0.0), 1.0))

// ======= config & helpers =======
#ifndef TILE_X
#define TILE_X 32
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif

// one extra element of shared stride to avoid bank conflicts
#ifndef SMEM_SKEW
#define SMEM_SKEW 1
#endif

// clamp to [0,1] if your data are normalized floats
__device__ __forceinline__ float clamp01(float v) {
    return fminf(1.f, fmaxf(0.f, v));
}

// ld with read-only cache hint where available
template<typename T>
__device__ __forceinline__ T ro(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}
#ifndef TILE_X
#define TILE_X 32
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif

// Add 1 to shared-memory row stride to reduce bank conflicts
#ifndef SMEM_SKEW
#define SMEM_SKEW 1
#endif

__device__ __forceinline__ float clamp01(float v) {
    return fminf(1.0f, fmaxf(0.0f, v));
}

// Read-only cache hint where available
template<typename T>
__device__ __forceinline__ T ro(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// ============================================================================
// Single-mask 2D convolution (sharpen, laplacian, emboss, etc.)
// - Input/output: interleaved float, length = width*height*CHANNELS
// - Mask passed as pointer (length K*K)
// ============================================================================
template<int K, int CHANNELS>
__global__ void conv2d_single_mask(
    const float* __restrict__ d_input,
    float* __restrict__       d_output,
    int width, int height,
    int in_row_stride_elems,   // usually width*CHANNELS for tightly packed
    int out_row_stride_elems,  // usually width*CHANNELS for tightly packed
    const float* __restrict__  d_mask       // length K*K
)
{
    constexpr int R = K / 2;
    const int sW = TILE_X + K - 1 + SMEM_SKEW;
    const int sH = TILE_Y + K - 1;

    extern __shared__ float smem[]; // layout: CHANNELS planes, each sW*sH
    float* sPlane[CHANNELS];
    {
        int planeSize = sW * sH;
        for (int c = 0; c < CHANNELS; ++c) sPlane[c] = smem + c * planeSize;
    }

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;

    // Load tile + halo with clamped coordinates
    for (int sy = threadIdx.y; sy < sH; sy += blockDim.y) {
        int iy = min(max(by + sy - R, 0), height - 1);
        for (int sx = threadIdx.x; sx < sW - SMEM_SKEW; sx += blockDim.x) {
            int ix = min(max(bx + sx - R, 0), width - 1);

            int g = iy * in_row_stride_elems + ix * CHANNELS;
            int s = sy * sW + sx;

#pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                sPlane[c][s] = ro(&d_input[g + c]);
            }
        }
    }
    __syncthreads();

    // Compute output
    int ox = bx + threadIdx.x;
    int oy = by + threadIdx.y;
    if (ox < width && oy < height) {
        const int s0 = threadIdx.y * sW + threadIdx.x;
        int o = oy * out_row_stride_elems + ox * CHANNELS;

#pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            float acc = 0.0f;

#pragma unroll
            for (int ky = 0; ky < K; ++ky) {
#pragma unroll
                for (int kx = 0; kx < K; ++kx) {
                    float m = d_mask[ky * K + kx];
                    acc += m * sPlane[c][s0 + ky * sW + kx];
                }
            }
            d_output[o + c] = clamp01(acc);
        }
    }
}

// ============================================================================
// Two-mask fused 2D convolution for Sobel (compute Gx and Gy in one pass)
// - Input/output: interleaved float
// - d_mask1: length K*K (e.g., Sobel Kx)
// - d_mask2: length K*K (e.g., Sobel Ky)
// - COMBINE_L1 true => |Gx| + |Gy|, false => sqrt(Gx^2 + Gy^2)
// ============================================================================
template<int K, int CHANNELS, bool COMBINE_L1 = true>
__global__ void conv2d_two_masks_fused(
    const float* __restrict__ d_input,
    float* __restrict__       d_output,
    int width, int height,
    int row_stride_elems,      // usually width*CHANNELS for tightly packed
    const float* __restrict__  d_mask1,
    const float* __restrict__  d_mask2
)
{
    static_assert(K % 2 == 1, "K must be odd");
    constexpr int R = K / 2;
    const int sW = TILE_X + K - 1 + SMEM_SKEW;
    const int sH = TILE_Y + K - 1;

    extern __shared__ float smem[]; // layout: CHANNELS planes, each sW*sH
    float* sPlane[CHANNELS];
    {
        int planeSize = sW * sH;
        for (int c = 0; c < CHANNELS; ++c) sPlane[c] = smem + c * planeSize;
    }

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;

    // Load tile + halo with clamped coordinates
    for (int sy = threadIdx.y; sy < sH; sy += blockDim.y) {
        int iy = min(max(by + sy - R, 0), height - 1);
        for (int sx = threadIdx.x; sx < sW - SMEM_SKEW; sx += blockDim.x) {
            int ix = min(max(bx + sx - R, 0), width - 1);

            int g = iy * row_stride_elems + ix * CHANNELS;
            int s = sy * sW + sx;

#pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                sPlane[c][s] = ro(&d_input[g + c]);
            }
        }
    }
    __syncthreads();

    // Compute output
    int ox = bx + threadIdx.x;
    int oy = by + threadIdx.y;
    if (ox < width && oy < height) {
        const int s0 = threadIdx.y * sW + threadIdx.x;
        int o = oy * row_stride_elems + ox * CHANNELS;

#pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            float gx = 0.0f, gy = 0.0f;

#pragma unroll
            for (int ky = 0; ky < K; ++ky) {
#pragma unroll
                for (int kx = 0; kx < K; ++kx) {
                    float p = sPlane[c][s0 + ky * sW + kx];
                    gx += d_mask1[ky * K + kx] * p;
                    gy += d_mask2[ky * K + kx] * p;
                }
            }

            float mag = COMBINE_L1 ? (fabsf(gx) + fabsf(gy))
                : sqrtf(gx * gx + gy * gy);
            d_output[o + c] = clamp01(mag);
        }
    }
}

// ============================================================================
// Example host-side usage (interleaved RGB floats in [0,1])
// ============================================================================
/*
    // Assume width,height, d_input,d_output exist; row stride equals width*3
    const int C = 3;
    dim3 block(TILE_X, TILE_Y);
    dim3 grid((width  + TILE_X - 1) / TILE_X,
              (height + TILE_Y - 1) / TILE_Y);

    // Shared memory size: (sW*sH) floats per channel
    // sW = TILE_X + K - 1 + SMEM_SKEW
    // sH = TILE_Y + K - 1
    auto smem_bytes = [&](int K){
        int sW = TILE_X + K - 1 + SMEM_SKEW;
        int sH = TILE_Y + K - 1;
        return static_cast<size_t>(sW) * sH * C * sizeof(float);
    };

    // 1) Sharpen (3x3)
    {
        constexpr int K = 3;
        float h_sharp[K*K] = {
            -1, -1, -1,
            -1,  9, -1,
            -1, -1, -1
        };
        conv2d_single_mask<K, C><<<grid, block, smem_bytes(K)>>>(
            d_input, d_output, width, height,
            width*C, width*C,
            h_sharp);
    }

    // 2) Sobel fused (3x3): output magnitude into d_output
    {
        constexpr int K = 3;
        float h_sobelX[K*K] = { -1, 0, 1,  -2, 0, 2,  -1, 0, 1 };
        float h_sobelY[K*K] = {  1, 2, 1,   0, 0, 0,  -1,-2,-1 };

        conv2d_two_masks_fused<K, C, true><<<grid, block, smem_bytes(K)>>>(
            d_input, d_output, width, height,
            width*C,
            h_sobelX, h_sobelY);
    }

    // Note: Gaussian should be handled separately with two 1-D passes for speed.
*/

// ============================================================================
// Notes:
// - Keep CHANNELS as 1 for grayscale or 3 for RGB. The interface uses floats.
// - For images stored tightly packed: row_stride_elems = width * CHANNELS.
// - If your data are uint8_t, convert to float on upload or add a fast path.
// - Consider moving small masks to __constant__ and unrolling K loops.
// - Build example: nvcc -O3 -arch=sm_80 yourfile.cu
// ============================================================================


#define CHANNELS 3
//@@ INSERT CODE HERE
__global__
void convolution(
    const float* __restrict__ d_input,   // length = width*height*3
    float* __restrict__ d_output,  // length = width*height*3
    int                         width,
    int                         height,
    const float* __restrict__   d_mask   // length = MASK_WIDTH*MASK_WIDTH
)
{
    extern __shared__ float s_tile[];    // size = (TILE_WIDTH+MASK_WIDTH-1)^2 * 3
    const int sW = TILE_WIDTH + MASK_WIDTH - 1;

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * TILE_WIDTH, by = blockIdx.y * TILE_WIDTH;

    // 1) load tile + halo into shared memory
    for (int y = ty; y < sW; y += blockDim.y) {
        for (int x = tx; x < sW; x += blockDim.x) {
            int ix = bx + x - RADIUS;
            int iy = by + y - RADIUS;
            for (int c = 0; c < 3; ++c) {
                float pix = 0.0f;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    pix = d_input[(iy * width + ix) * 3 + c];
                }
                s_tile[(y * sW + x) * 3 + c] = pix;

            }
        }
    }
    __syncthreads();

    // 2) compute convolution for our output pixel
    int ox = bx + tx, oy = by + ty;
    if (ox < width && oy < height) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
        for (int i = 0; i < MASK_WIDTH; ++i) {
            for (int j = 0; j < MASK_WIDTH; ++j) {
                float m = d_mask[i * MASK_WIDTH + j];
                int base = ((ty + i) * sW + (tx + j)) * 3;
                sum0 += m * s_tile[base + 0];
                sum1 += m * s_tile[base + 1];
                sum2 += m * s_tile[base + 2];
            }
        }
        // clamp to [0,1]
        sum0 = clampf(sum0);
        sum1 = clampf(sum1);
        sum2 = clampf(sum2);

        int outIdx = (oy * width + ox) * 3;
        d_output[outIdx + 0] = sum0;
        d_output[outIdx + 1] = sum1;
        d_output[outIdx + 2] = sum2;
    }
}

/* 


*/


int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char* inputImageFile;
    char* inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* hostMaskData;
    float* deviceInputImageData;
    float* deviceOutputImageData;
    float* deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float*)wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    //@@ INSERT CODE HERE
    wbCheck(cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceMaskData, maskRows * maskColumns * sizeof(float)));


    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    //@@ INSERT CODE HERE

    wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceOutputImageData, hostOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice));


    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    // number of “pixels” in tile+halo:
    int sW = TILE_WIDTH + MASK_WIDTH - 1;
    int numPixels = sW * sW;
    // number of floats = pixels × channels (3 for RGB)
    int numFloats = numPixels * 3;
    // total bytes = floats × sizeof(float)
    size_t shMemBytes = numFloats * sizeof(float);


    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(
        (imageWidth + TILE_WIDTH - 1) / TILE_WIDTH,
        (imageHeight + TILE_WIDTH - 1) / TILE_WIDTH
    );

    convolution << <gridDim, blockDim, shMemBytes >> > (deviceInputImageData, deviceOutputImageData,
        imageWidth, imageHeight, deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    //@@ INSERT CODE HERE
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
        imageWidth * imageHeight * imageChannels * sizeof(float),
        cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    //@@ Insert code here
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
