#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "wb.h"

#include <stdio.h>

/* cuda standard libraries */
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <math.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

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

// ------------------------------- Tiling --------------------------------------
#ifndef TILE_X
#define TILE_X 32
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif
#ifndef SMEM_SKEW
#define SMEM_SKEW 1    // small skew to reduce shared mem bank conflicts
#endif

// We assume RGB images for this assignment. Change to 1 if running grayscale datasets.
#define CHANNELS 3

// ----------------------------- Device helpers --------------------------------

// OpenCV BORDER_REFLECT_101 index mapping: mirror around edges without repeating the edge pixel.
// Produces index in [0, n-1].
__device__ __forceinline__ int reflect101(int p, int n) {
    if (n <= 1) return 0;
    int period = 2 * n - 2;
    int t = p % period;
    if (t < 0) t += period;
    if (t >= n) t = period - t;
    return t;
}

// Read-only cached load when available
template <typename T>
__device__ __forceinline__ T ro(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// Clamp to 0..255 float
__device__ __forceinline__ float clamp255(float v) {
    if (v < 0.0f)   return 0.0f;
    if (v > 255.0f) return 255.0f;
    return v;
}

// In-place scaler for buffers
__global__ void scale_inplace(float* data, size_t n, float factor) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

// ----------------------- Single-mask 2D convolution --------------------------
// Generic per-channel KxK convolution with REFLECT_101 borders.
// Use this for Sharpen (3x3) or any other single-mask filter.

template<int K, int CH>
__global__ void conv2d_single_mask_reflect101(
    const float* __restrict__ d_input,   // interleaved RGB or Gray, in [0,255]
    float* __restrict__       d_output,  // interleaved, will clamp to [0,255]
    int width, int height,
    int in_row_stride_elems,
    int out_row_stride_elems,
    const float* __restrict__ d_mask     // K*K
) {
    static_assert(K % 2 == 1, "K must be odd");
    constexpr int R = K / 2;

    const int sW = TILE_X + K - 1 + SMEM_SKEW; // halo both sides in X
    const int sH = TILE_Y + K - 1;            // halo both sides in Y

    extern __shared__ float smem[];           // CH planes, each sW*sH
    float* sPlane[CH];
    {
        int planeSize = sW * sH;
        for (int c = 0; c < CH; ++c) sPlane[c] = smem + c * planeSize;
    }

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;

    // Load tile + halo with REFLECT_101
    for (int sy = threadIdx.y; sy < sH; sy += blockDim.y) {
        int iy = reflect101(by + sy - R, height);
        for (int sx = threadIdx.x; sx < sW - SMEM_SKEW; sx += blockDim.x) {
            int ix = reflect101(bx + sx - R, width);
            int g = iy * in_row_stride_elems + ix * CH;
            int s = sy * sW + sx;
#pragma unroll
            for (int c = 0; c < CH; ++c) {
                sPlane[c][s] = ro(&d_input[g + c]);
            }
        }
    }
    __syncthreads();

    // Convolution per output pixel
    int ox = bx + threadIdx.x;
    int oy = by + threadIdx.y;
    if (ox < width && oy < height) {
        const int s0 = threadIdx.y * sW + threadIdx.x;
        int o = oy * out_row_stride_elems + ox * CH;

#pragma unroll
        for (int c = 0; c < CH; ++c) {
            float acc = 0.0f;
#pragma unroll
            for (int ky = 0; ky < K; ++ky) {
#pragma unroll
                for (int kx = 0; kx < K; ++kx) {
                    float m = d_mask[ky * K + kx];
                    acc += m * sPlane[c][s0 + ky * sW + kx];
                }
            }
            d_output[o + c] = clamp255(acc);
        }
    }
}

// ----------------------------- Fused Sobel -----------------------------------
// Sobel Gx and Gy on luminance (from RGB), L2 magnitude, replicate to CH channels.
// REFLECT_101 borders in tile load. Double accumulators to mimic CV_64F.

template<int K, int CH>
__global__ void conv2d_sobel_fused_reflect101_rgb(
    const float* __restrict__ d_input,   // interleaved RGB or Gray, in [0,255]
    float* __restrict__       d_output,  // interleaved, gray replicated to CH
    int width, int height,
    int row_stride_elems,
    const float* __restrict__ d_kx,      // K*K Sobel X
    const float* __restrict__ d_ky       // K*K Sobel Y
) {
    static_assert(K % 2 == 1, "K must be odd");
    constexpr int R = K / 2;

    const int sW = TILE_X + K - 1 + SMEM_SKEW;
    const int sH = TILE_Y + K - 1;

    extern __shared__ float smem[];
    float* sPlane[CH];
    {
        int planeSize = sW * sH;
        for (int c = 0; c < CH; ++c) sPlane[c] = smem + c * planeSize;
    }

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;

    // Load tile + halo with REFLECT_101
    for (int sy = threadIdx.y; sy < sH; sy += blockDim.y) {
        int iy = reflect101(by + sy - R, height);
        for (int sx = threadIdx.x; sx < sW - SMEM_SKEW; sx += blockDim.x) {
            int ix = reflect101(bx + sx - R, width);
            int g = iy * row_stride_elems + ix * CH;
            int s = sy * sW + sx;
#pragma unroll
            for (int c = 0; c < CH; ++c) {
                sPlane[c][s] = ro(&d_input[g + c]);
            }
        }
    }
    __syncthreads();

    int ox = bx + threadIdx.x;
    int oy = by + threadIdx.y;
    if (ox < width && oy < height) {
        const int s0 = threadIdx.y * sW + threadIdx.x;

        // double accumulators like CV_64F
        double gx = 0.0, gy = 0.0;

#pragma unroll
        for (int ky = 0; ky < K; ++ky) {
#pragma unroll
            for (int kx = 0; kx < K; ++kx) {
                float y;  // luminance
                int idx = s0 + ky * sW + kx;

                if (CH == 1) {
                    y = sPlane[0][idx];
                }
                else {
                    // RGB -> luminance (OpenCV RGB weights)
                    float r = sPlane[0][idx];
                    float g = sPlane[1][idx];
                    float b = sPlane[2][idx];
                    y = 0.299f * r + 0.587f * g + 0.114f * b;
                }

                float kxv = d_kx[ky * K + kx];
                float kyv = d_ky[ky * K + kx];
                gx += (double)kxv * (double)y;
                gy += (double)kyv * (double)y;
            }
        }

        double mag = sqrt(gx * gx + gy * gy);
        float outf = clamp255((float)mag);

        int o = oy * row_stride_elems + ox * CH;
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            d_output[o + c] = outf; // replicate gray
        }
    }
}

// -------------------------- Separable Gaussian -------------------------------
// Horizontal and vertical passes, 1D mask length K, per-channel, REFLECT_101.

template<int K, int CH>
__global__ void gauss1d_horiz_reflect101(
    const float* __restrict__ d_input,
    float* __restrict__       d_output,
    int width, int height,
    int row_stride_elems,               // width * CH
    const float* __restrict__ d_mask1d  // length K
) {
    static_assert(K % 2 == 1, "K must be odd");
    constexpr int R = K / 2;

    const int sW = TILE_X + K - 1 + SMEM_SKEW; // halo in X
    const int sH = TILE_Y;                     // no halo in Y
    extern __shared__ float smem[];
    float* sPlane[CH];
    {
        int planeSize = sW * sH;
        for (int c = 0; c < CH; ++c) sPlane[c] = smem + c * planeSize;
    }

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;

    // Load tile + X halo
    for (int sy = threadIdx.y; sy < sH; sy += blockDim.y) {
        int iy = reflect101(by + sy, height);
        for (int sx = threadIdx.x; sx < sW - SMEM_SKEW; sx += blockDim.x) {
            int ix = reflect101(bx + sx - R, width);
            int g = iy * row_stride_elems + ix * CH;
            int s = sy * sW + sx;
#pragma unroll
            for (int c = 0; c < CH; ++c) {
                sPlane[c][s] = ro(&d_input[g + c]);
            }
        }
    }
    __syncthreads();

    int ox = bx + threadIdx.x;
    int oy = by + threadIdx.y;
    if (ox < width && oy < height) {
        int s0 = threadIdx.y * sW + threadIdx.x;
        int o = oy * row_stride_elems + ox * CH;

#pragma unroll
        for (int c = 0; c < CH; ++c) {
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < K; ++k) {
                acc += d_mask1d[k] * sPlane[c][s0 + k];
            }
            d_output[o + c] = clamp255(acc);
        }
    }
}

template<int K, int CH>
__global__ void gauss1d_vert_reflect101(
    const float* __restrict__ d_input,
    float* __restrict__       d_output,
    int width, int height,
    int row_stride_elems,               // width * CH
    const float* __restrict__ d_mask1d  // length K
) {
    static_assert(K % 2 == 1, "K must be odd");
    constexpr int R = K / 2;

    const int sW = TILE_X + SMEM_SKEW;        // no halo in X
    const int sH = TILE_Y + K - 1;            // halo in Y
    extern __shared__ float smem[];
    float* sPlane[CH];
    {
        int planeSize = sW * sH;
        for (int c = 0; c < CH; ++c) sPlane[c] = smem + c * planeSize;
    }

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;

    // Load tile + Y halo
    for (int sy = threadIdx.y; sy < sH; sy += blockDim.y) {
        int iy = reflect101(by + sy - R, height);
        for (int sx = threadIdx.x; sx < sW - SMEM_SKEW; sx += blockDim.x) {
            int ix = reflect101(bx + sx, width);
            int g = iy * row_stride_elems + ix * CH;
            int s = sy * sW + sx;
#pragma unroll
            for (int c = 0; c < CH; ++c) {
                sPlane[c][s] = ro(&d_input[g + c]);
            }
        }
    }
    __syncthreads();

    int ox = bx + threadIdx.x;
    int oy = by + threadIdx.y;
    if (ox < width && oy < height) {
        int s0 = threadIdx.y * sW + threadIdx.x;
        int o = oy * row_stride_elems + ox * CH;

#pragma unroll
        for (int c = 0; c < CH; ++c) {
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < K; ++k) {
                acc += d_mask1d[k] * sPlane[c][s0 + k * sW];
            }
            d_output[o + c] = clamp255(acc);
        }
    }
}

// --------------------------- Host utilities ----------------------------------

// Build normalized 1-D Gaussian weights. If sigma<=0, use OpenCV's rule for sigma=0.
static inline void make_gaussian_kernel_1d(int K, double sigma, std::vector<float>& out) {
    if (K <= 0 || (K % 2) == 0) throw std::runtime_error("Gaussian K must be odd and >0");
    if (sigma <= 0.0) {
        // OpenCV: sigma = 0.3*((K-1)*0.5 - 1) + 0.8
        sigma = 0.3 * ((K - 1) * 0.5 - 1.0) + 0.8;
    }
    int R = K / 2;
    out.resize(K);
    double sum = 0.0;
    double s2 = 2.0 * sigma * sigma;
    for (int i = -R, j = 0; j < K; ++j, ++i) {
        double v = std::exp(-(i * i) / s2);
        out[j] = static_cast<float>(v);
        sum += v;
    }
    float inv = static_cast<float>(1.0 / sum);
    for (int j = 0; j < K; ++j) out[j] *= inv;
}

// Simple CLI parse: --op gaussian|sobel|sharpen (default sobel)
enum Op { OP_SHARPEN, OP_SOBEL, OP_GAUSSIAN };
static Op parse_op(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--op") || !strcmp(argv[i], "-op")) {
            if (i + 1 < argc) {
                if (!strcmp(argv[i + 1], "sharpen"))  return OP_SHARPEN;
                if (!strcmp(argv[i + 1], "sobel"))    return OP_SOBEL;
                if (!strcmp(argv[i + 1], "gaussian")) return OP_GAUSSIAN;
            }
        }
        if (!strncmp(argv[i], "--op=", 5)) {
            const char* v = argv[i] + 5;
            if (!strcmp(v, "sharpen"))  return OP_SHARPEN;
            if (!strcmp(v, "sobel"))    return OP_SOBEL;
            if (!strcmp(v, "gaussian")) return OP_GAUSSIAN;
        }
    }
    return OP_SOBEL;
}

// ------------------------------- main ----------------------------------------

int main(int argc, char* argv[]) {
    wbArg_t arg = wbArg_read(argc, argv);
    Op op = parse_op(argc, argv);

    // Input via wb (floats in [0,1])
    char* inputImagePath = wbArg_getInputFile(arg, 0);
    wbImage_t inputImage = wbImport(inputImagePath);
    int width = wbImage_getWidth(inputImage);
    int height = wbImage_getHeight(inputImage);
    int chans = wbImage_getChannels(inputImage);

    if (chans != CHANNELS) {
        fprintf(stderr, "Expected %d channels, got %d. Rebuild with CHANNELS=%d if needed.\n",
            CHANNELS, chans, chans);
        return 1;
    }

    wbImage_t outputImage = wbImage_new(width, height, chans);
    float* h_in = wbImage_getData(inputImage);   // [0,1]
    float* h_out = wbImage_getData(outputImage);  // will receive [0,1]

    const size_t nElems = (size_t)width * height * CHANNELS;

    // Device buffers
    float* d_in = nullptr, * d_out = nullptr;
    // Sharpen and Sobel masks (3x3)
    float* d_sharp = nullptr, * d_kx = nullptr, * d_ky = nullptr;
    // Gaussian
    float* d_gmask = nullptr; // 1-D length K
    float* d_tmp = nullptr; // temp buffer for separable passes

    // Host masks
    static const float h_sharpen3[9] = {
        -1.f, -1.f, -1.f,
        -1.f,  9.f, -1.f,
        -1.f, -1.f, -1.f
    };
    static const float h_sobelX[9] = { -1.f, 0.f, 1.f,  -2.f, 0.f, 2.f,  -1.f, 0.f, 1.f };
    static const float h_sobelY[9] = { 1.f, 2.f, 1.f,   0.f, 0.f, 0.f,  -1.f,-2.f,-1.f };

    wbTime_start(GPU, "GPU total");

    // Allocations
    wbTime_start(GPU, "cudaMalloc");
    wbCheck(cudaMalloc(&d_in, nElems * sizeof(float)));
    wbCheck(cudaMalloc(&d_out, nElems * sizeof(float)));
    if (op == OP_SHARPEN) {
        wbCheck(cudaMalloc(&d_sharp, 9 * sizeof(float)));
    }
    else if (op == OP_SOBEL) {
        wbCheck(cudaMalloc(&d_kx, 9 * sizeof(float)));
        wbCheck(cudaMalloc(&d_ky, 9 * sizeof(float)));
    }
    else { // OP_GAUSSIAN
        wbCheck(cudaMalloc(&d_gmask, 5 * sizeof(float)));       // K=5
        wbCheck(cudaMalloc(&d_tmp, nElems * sizeof(float)));  // intermediate
    }
    wbTime_stop(GPU, "cudaMalloc");

    // H2D
    wbTime_start(Copy, "H2D");
    wbCheck(cudaMemcpy(d_in, h_in, nElems * sizeof(float), cudaMemcpyHostToDevice));
    if (op == OP_SHARPEN) {
        wbCheck(cudaMemcpy(d_sharp, h_sharpen3, 9 * sizeof(float), cudaMemcpyHostToDevice));
    }
    else if (op == OP_SOBEL) {
        wbCheck(cudaMemcpy(d_kx, h_sobelX, 9 * sizeof(float), cudaMemcpyHostToDevice));
        wbCheck(cudaMemcpy(d_ky, h_sobelY, 9 * sizeof(float), cudaMemcpyHostToDevice));
    }
    else { // OP_GAUSSIAN
        std::vector<float> g5;
        make_gaussian_kernel_1d(5, 0.0, g5); // sigma=0 => OpenCV rule
        wbCheck(cudaMemcpy(d_gmask, g5.data(), 5 * sizeof(float), cudaMemcpyHostToDevice));
    }
    wbTime_stop(Copy, "H2D");

    // Scale input [0,1] -> [0,255] to match OpenCV-like numerics
    {
        wbTime_start(Compute, "Scale input 0..1 -> 0..255");
        int tpb = 256;
        int blks = (int)((nElems + tpb - 1) / tpb);
        scale_inplace << <blks, tpb >> > (d_in, nElems, 255.0f);
        wbCheck(cudaGetLastError());
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Scale input 0..1 -> 0..255");
    }

    // Launch config
    dim3 block(TILE_X, TILE_Y);
    dim3 grid((width + TILE_X - 1) / TILE_X,
        (height + TILE_Y - 1) / TILE_Y);

    auto smem_bytes_conv = [&](int K)->size_t {
        int sW = TILE_X + K - 1 + SMEM_SKEW;
        int sH = TILE_Y + K - 1;
        return (size_t)CHANNELS * sW * sH * sizeof(float);
        };
    auto smem_bytes_gauss_h = [&](int K)->size_t {
        int sW = TILE_X + K - 1 + SMEM_SKEW;
        int sH = TILE_Y;
        return (size_t)CHANNELS * sW * sH * sizeof(float);
        };
    auto smem_bytes_gauss_v = [&](int K)->size_t {
        int sW = TILE_X + SMEM_SKEW;
        int sH = TILE_Y + K - 1;
        return (size_t)CHANNELS * sW * sH * sizeof(float);
        };

    // Dispatch
    if (op == OP_SHARPEN) {
        wbTime_start(Compute, "Sharpen kernel");
        conv2d_single_mask_reflect101<3, CHANNELS> << <grid, block, smem_bytes_conv(3) >> > (
            d_in, d_out, width, height,
            width * CHANNELS, width * CHANNELS,
            d_sharp
            );
        wbCheck(cudaGetLastError());
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Sharpen kernel");

    }
    else if (op == OP_SOBEL) {
        wbTime_start(Compute, "Sobel fused kernel");
        conv2d_sobel_fused_reflect101_rgb<3, CHANNELS> << <grid, block, smem_bytes_conv(3) >> > (
            d_in, d_out, width, height,
            width * CHANNELS,
            d_kx, d_ky
            );
        wbCheck(cudaGetLastError());
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Sobel fused kernel");

    }
    else { // OP_GAUSSIAN
        // Horizontal pass: d_in -> d_tmp
        wbTime_start(Compute, "Gaussian horiz");
        gauss1d_horiz_reflect101<5, CHANNELS> << <grid, block, smem_bytes_gauss_h(5) >> > (
            d_in, d_tmp, width, height,
            width * CHANNELS,
            d_gmask
            );
        wbCheck(cudaGetLastError());
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Gaussian horiz");

        // Vertical pass: d_tmp -> d_out
        wbTime_start(Compute, "Gaussian vert");
        gauss1d_vert_reflect101<5, CHANNELS> << <grid, block, smem_bytes_gauss_v(5) >> > (
            d_tmp, d_out, width, height,
            width * CHANNELS,
            d_gmask
            );
        wbCheck(cudaGetLastError());
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Gaussian vert");
    }

    // Scale output back to [0,1] for wbSolution
    {
        wbTime_start(Compute, "Scale output 0..255 -> 0..1");
        int tpb = 256;
        int blks = (int)((nElems + tpb - 1) / tpb);
        scale_inplace << <blks, tpb >> > (d_out, nElems, 1.0f / 255.0f);
        wbCheck(cudaGetLastError());
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Scale output 0..255 -> 0..1");
    }

    // D2H and emit
    wbTime_start(Copy, "D2H");
    wbCheck(cudaMemcpy(h_out, d_out, nElems * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "D2H");

    wbTime_stop(GPU, "GPU total");

    wbSolution(arg, outputImage);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    if (d_sharp) cudaFree(d_sharp);
    if (d_kx)    cudaFree(d_kx);
    if (d_ky)    cudaFree(d_ky);
    if (d_gmask) cudaFree(d_gmask);
    if (d_tmp)   cudaFree(d_tmp);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    return 0;
}