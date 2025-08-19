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
#include <cstring>
#include <stdexcept>
#include <math.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <fstream>

#include <npp.h>


#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <cmath>



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

// Read-only cached load when available
template <typename T>
__device__ __forceinline__ T ro(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}


// In-place scaler for buffers
__global__ void scale_inplace(float* data, size_t n, float factor) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

/* Utility kernel to ensure comparable outputs to python CV2 */
/* Mirror image at the border without repeating edge pixel (symettric neighborhood) */
// OpenCV BORDER_REFLECT_101 index mapping: mirror around edges without repeating the edge pixel.
// Produces index in [0, n-1].
// Host+device reflect101 so both CPU paths and CUDA kernels can use it.
__host__ __device__ __forceinline__
int reflect101(int p, int n) {
    if (n <= 1) return 0;
    int period = 2 * n - 2;
    int t = p % period;
    if (t < 0) t += period;
    if (t >= n) t = period - t;
    return t;
}

// Device-side clamp (used in kernels)
__device__ __forceinline__
float clamp255(float v) {
    if (v < 0.0f)   return 0.0f;
    if (v > 255.0f) return 255.0f;
    return v;
}

// Host-side clamp (used in CPU code)
__host__ __forceinline__
float clamp255_h(float v) {
    if (v < 0.0f)   return 0.0f;
    if (v > 255.0f) return 255.0f;
    return v;
}


// ===================== State of the Art open source reference implementation  =========================
// uses npp lib

/* Our kernels operate on normalied floats with full dynamic range. */
/* 32f variatns of open source kernels is most comparable */


void npp_gauss5x5_f32_default(const Npp32f* d_src, int srcPitch,
    Npp32f* d_dst, int dstPitch, int w, int h);

void npp_linear3x3_f32_default(const Npp32f* d_src, int srcPitch,
    Npp32f* d_dst, int dstPitch, int w, int h,
    const Npp32f k3x3[3 * 3]);

void npp_sobel3x3_dx_32f(const Npp32f* d_src, int srcPitch,
    Npp32f* d_dst, int dstPitch, int w, int h);


// dx
static const Npp32f SOBEL_X_3x3[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};
// dy
static const Npp32f SOBEL_Y_3x3[9] = {
    -1,-2,-1,
     0, 0, 0,
     1, 2, 1
};
void npp_gauss5x5_f32_default(const Npp32f* d_src, int srcPitch,
    Npp32f* d_dst, int dstPitch,
    int w, int h)
{
    const NppiSize srcSize{ w, h };
    const NppiPoint srcOfs{ 0, 0 };
    const NppiSize roi{ w, h };

    NppStatus st = nppiFilterGaussBorder_32f_C1R(
        d_src, srcPitch, srcSize, srcOfs,
        d_dst, dstPitch, roi,
        NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE);
    if (st != NPP_SUCCESS) throw std::runtime_error("nppiFilterGaussBorder_32f_C1R failed");
}

void npp_linear3x3_f32_default(const Npp32f* d_src, int srcPitch,
    Npp32f* d_dst, int dstPitch,
    int w, int h, const Npp32f k3x3[3 * 3])
{
    const NppiSize srcSize{ w, h };
    const NppiPoint srcOfs{ 0, 0 };
    const NppiSize roi{ w, h };
    const NppiSize kSize{ 3,3 };
    const NppiPoint anchor{ 1,1 };

    NppStatus st = nppiFilterBorder_32f_C1R(
        d_src, srcPitch, srcSize, srcOfs,
        d_dst, dstPitch, roi,
        k3x3, kSize, anchor, NPP_BORDER_REPLICATE);
    if (st != NPP_SUCCESS) throw std::runtime_error("nppiFilterBorder_32f_C1R failed");
}

void npp_gauss5x5_32f(const Npp32f* d_src, int srcPitch, Npp32f* d_dst, int dstPitch,
    int w, int h) {
    NppiSize srcSize{ w, h };
    NppiPoint srcOfs{ 0, 0 };
    NppiSize roi{ w, h };
    auto st = nppiFilterGaussBorder_32f_C1R(
        d_src, srcPitch, srcSize, srcOfs,
        d_dst, dstPitch, roi,
        NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE);
    if (st != NPP_SUCCESS) throw std::runtime_error("nppiFilterGaussBorder_32f failed");
}

void npp_sobel3x3_dx_32f(const Npp32f* d_src, int srcPitch, Npp32f* d_dst, int dstPitch,
    int w, int h) {
    NppiSize roi{ w, h };
    auto st = nppiFilterSobelHoriz_32f_C1R(d_src, srcPitch, d_dst, dstPitch, roi);
    if (st != NPP_SUCCESS) throw std::runtime_error("nppiFilterSobelHoriz_32f failed");
}

// pointwise magnitude (float32 in/out). If your app wants 0..1, this stays in float domain.
__global__ void mag32f_kernel(const float* __restrict__ dx,
    const float* __restrict__ dy,
    float* __restrict__ mag,
    int width, int height, int pitchElems)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int idx = y * pitchElems + x;
    float gx = dx[idx], gy = dy[idx];
    mag[idx] = sqrtf(gx * gx + gy * gy);
}

void npp_sobel_mag3x3_f32(const Npp32f* d_src, int srcPitchBytes,
    Npp32f* d_dst, int dstPitchBytes, int w, int h)
{
    // reuse linear3x3 to get dx/dy, then compute magnitude
    size_t fPitchB = 0;
    Npp32f* d_dx = nullptr, * d_dy = nullptr;
    cudaMallocPitch((void**)&d_dx, &fPitchB, w * sizeof(Npp32f), h);
    cudaMallocPitch((void**)&d_dy, &fPitchB, w * sizeof(Npp32f), h);

    // dx, dy
    npp_linear3x3_f32_default(d_src, srcPitchBytes, d_dx, (int)fPitchB, w, h, SOBEL_X_3x3);
    npp_linear3x3_f32_default(d_src, srcPitchBytes, d_dy, (int)fPitchB, w, h, SOBEL_Y_3x3);

    // mag = hypot(dx,dy)
    dim3 block(32, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    mag32f_kernel << <grid, block >> > (d_dx, d_dy, d_dst, w, h, (int)(fPitchB / sizeof(float)));
    cudaDeviceSynchronize();

    cudaFree(d_dx);
    cudaFree(d_dy);
}

// C1: float per pixel
__global__ void mag_c1(const float* __restrict__ dx,
    const float* __restrict__ dy,
    float* __restrict__ out,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gx = dx[i], gy = dy[i];
        out[i] = sqrtf(gx * gx + gy * gy);
    }
}

// C3: interleaved RGB float per pixel
__global__ void mag_c3(const float* __restrict__ dx,
    const float* __restrict__ dy,
    float* __restrict__ out,
    int nPixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nPixels) {
        int base = 3 * i;
        float gx0 = dx[base + 0], gy0 = dy[base + 0];
        float gx1 = dx[base + 1], gy1 = dy[base + 1];
        float gx2 = dx[base + 2], gy2 = dy[base + 2];
        out[base + 0] = sqrtf(gx0 * gx0 + gy0 * gy0);
        out[base + 1] = sqrtf(gx1 * gx1 + gy1 * gy1);
        out[base + 2] = sqrtf(gx2 * gx2 + gy2 * gy2);
    }
}

// ===================== End =========================

// ===================== CPU reference implementations =====================
// All CPU functions accept floats in [0,255] and write floats in [0,255].
// Layout: interleaved CH channels, row_stride_elems = width*CH.
// Threads: if built with OpenMP, set threads>1 to use multi-threading;
//          otherwise they run single-threaded even if threads>1.
/*********************Multi-threaded sequential Start*****************************/

// Single-mask KxK convolution (e.g., Sharpen 3x3)
template<int K, int CH>
static void conv2d_single_mask_reflect101_cpu(
    const float* in, float* out,
    int width, int height, int row_stride_elems,
    const float* mask,
    int threads // 1 for ST, >1 for MT
) {
    const int R = K / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int o = y * row_stride_elems + x * CH;
            for (int c = 0; c < CH; ++c) {
                float acc = 0.0f;
                for (int ky = -R; ky <= R; ++ky) {
                    int iy = reflect101(y + ky, height);
                    for (int kx = -R; kx <= R; ++kx) {
                        int ix = reflect101(x + kx, width);
                        acc += mask[(ky + R) * K + (kx + R)]
                            * in[iy * row_stride_elems + ix * CH + c];
                    }
                }
                out[o + c] = clamp255_h(acc);
            }
        }
    }
}

// Fused Sobel: luminance -> L2 magnitude, replicate to CH channels
template<int K, int CH>
static void sobel_fused_reflect101_rgb_cpu(
    const float* in, float* out,
    int width, int height, int row_stride_elems,
    const float* kx, const float* ky,
    int threads
) {
    const int R = K / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double gx = 0.0, gy = 0.0;
            for (int ky_i = -R; ky_i <= R; ++ky_i) {
                int iy = reflect101(y + ky_i, height);
                for (int kx_i = -R; kx_i <= R; ++kx_i) {
                    int ix = reflect101(x + kx_i, width);
                    float ypix;
                    if (CH == 1) {
                        ypix = in[iy * row_stride_elems + ix * CH + 0];
                    }
                    else {
                        const float r = in[iy * row_stride_elems + ix * CH + 0];
                        const float g = in[iy * row_stride_elems + ix * CH + 1];
                        const float b = in[iy * row_stride_elems + ix * CH + 2];
                        ypix = 0.299f * r + 0.587f * g + 0.114f * b;
                    }
                    gx += (double)kx[(ky_i + R) * K + (kx_i + R)] * (double)ypix;
                    gy += (double)ky[(ky_i + R) * K + (kx_i + R)] * (double)ypix;
                }
            }
            float mag = clamp255_h((float)std::sqrt(gx * gx + gy * gy));
            int o = y * row_stride_elems + x * CH;
            for (int c = 0; c < CH; ++c) out[o + c] = mag;
        }
    }
}


enum Backend { BK_CUDA, BK_CPU, BK_MT };

static Backend parse_backend(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            if (!strcmp(argv[i + 1], "cuda")) return BK_CUDA;
            if (!strcmp(argv[i + 1], "cpu"))  return BK_CPU;
            if (!strcmp(argv[i + 1], "mt"))   return BK_MT;
        }
        else if (!strncmp(argv[i], "--backend=", 10)) {
            const char* v = argv[i] + 10;
            if (!strcmp(v, "cuda")) return BK_CUDA;
            if (!strcmp(v, "cpu"))  return BK_CPU;
            if (!strcmp(v, "mt"))   return BK_MT;
        }
        // convenience aliases
        if (!strcmp(argv[i], "--cpu_sobel") || !strcmp(argv[i], "--cpu_sharpen") || !strcmp(argv[i], "--cpu_gaussian"))
            return BK_CPU;
        if (!strcmp(argv[i], "--multithread_sobel") || !strcmp(argv[i], "--multithread_sharpen") || !strcmp(argv[i], "--multithread_gaussian"))
            return BK_MT;
    }
    return BK_CUDA;
}

static int parse_threads(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) return atoi(argv[i + 1]); // can be 0
        if (!strncmp(argv[i], "--threads=", 10)) return atoi(argv[i] + 10);           // can be 0
    }
#ifdef _OPENMP
    return 0;  // 0 => auto (use omp_get_max_threads later)
#else
    return 1;
#endif
}


/*********************Multi-threaded sequential End*****************************/

/*********************************************************************************************/
/*SOBEL exploration. Kernel implemenations with various optimization techniques NOT employed. */
__constant__ float c_sobelX3x3[9];
__constant__ float c_sobelY3x3[9];

/* 
   Uses __restrict__ - allows compiler to assume input pointers are read only, resulst in better instruction scheduling
       and register allocation
   
   no warp divergence on if (CH) statement, not exactly an optimization though
   proper bounds checking
   coalesced write pattern. neighboring threads write neighboring pixels
   read pattern leverages caches well, small 3x3 tile size 

*/
__global__ void sobel_v0(
    const float* __restrict__ d_in,
    float* __restrict__       d_out,
    int width, int height, int row_stride_elems, int CH,
    const float* __restrict__ d_kx,   // 3x3
    const float* __restrict__ d_ky    // 3x3
) {
    // map output pixels (1 per thread) to 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // prevent indexing out of bounds
    if (x >= width || y >= height) return;

    //gx and gy are accumulators for horizontal and vertical gradients 
    //double used to produce identical output to CV2 reference api
    double gx = 0.0, gy = 0.0;
    // accumulate on luminance computed on the fly

    //3x3 grid centered at x,y
    //if a portion of the neighborhood would be out of bounds, 
    //   map it back in bounds with border reflect (mirror)
    for (int ky = -1; ky <= 1; ++ky) {
        int iy = reflect101(y + ky, height);
        for (int kx = -1; kx <= 1; ++kx) {
            int ix = reflect101(x + kx, width);
            //g base index of input for pixel ix,iy
            //row_stride_elems = width * channels
            int g = iy * row_stride_elems + ix * CH;
            float ypix;
            //Luminance conversion from input pixel
            //used to match OpenCV reference API outputs.
            if (CH == 1) {
                ypix = d_in[g];
            }
            else {
                float r = d_in[g + 0];
                float gch = d_in[g + 1];
                float b = d_in[g + 2];
                ypix = 0.299f * r + 0.587f * gch + 0.114f * b;
            }

            //apply sobel masks and accumulate 
            float kxv = d_kx[(ky + 1) * 3 + (kx + 1)];
            float kyv = d_ky[(ky + 1) * 3 + (kx + 1)];
            gx += (double)kxv * (double)ypix;
            gy += (double)kyv * (double)ypix;
        }
    }
    //convert gradient to output magnitude 
    float mag = clamp255((float)sqrt(gx * gx + gy * gy));
    int o = y * row_stride_elems + x * CH;
    for (int c = 0; c < CH; ++c) d_out[o + c] = mag;
}

__global__ void sobel_v1_ro_const(
    const float* __restrict__ d_in,
    float* __restrict__       d_out,
    int width, int height, int row_stride_elems, int CH
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    double gx = 0.0, gy = 0.0;

    for (int ky = -1; ky <= 1; ++ky) {
        int iy = reflect101(y + ky, height);
        for (int kx = -1; kx <= 1; ++kx) {
            int ix = reflect101(x + kx, width);
            int g = iy * row_stride_elems + ix * CH;

            float ypix;
            if (CH == 1) {
#if __CUDA_ARCH__ >= 350
                float p0 = __ldg(&d_in[g]);
#else
                float p0 = d_in[g];
#endif
                ypix = p0;
            }
            else {
#if __CUDA_ARCH__ >= 350
                float r = __ldg(&d_in[g + 0]);
                float gc = __ldg(&d_in[g + 1]);
                float b = __ldg(&d_in[g + 2]);
#else
                float r = d_in[g + 0];
                float gc = d_in[g + 1];
                float b = d_in[g + 2];
#endif
                ypix = 0.299f * r + 0.587f * gc + 0.114f * b;
            }

            float kxv = c_sobelX3x3[(ky + 1) * 3 + (kx + 1)];
            float kyv = c_sobelY3x3[(ky + 1) * 3 + (kx + 1)];
            gx += (double)kxv * (double)ypix;
            gy += (double)kyv * (double)ypix;
        }
    }

    float mag = clamp255((float)sqrt(gx * gx + gy * gy));
    int o = y * row_stride_elems + x * CH;
    for (int c = 0; c < CH; ++c) d_out[o + c] = mag;
}


// ----------------------------- Fused Sobel -----------------------------------
// ------------------------------Sobel "V2" ------------------------------------
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

template<int CH>
__global__ void sobel_v3_tiled_y(
    const float* __restrict__ d_in,
    float* __restrict__       d_out,
    int width, int height, int row_stride_elems
) {
    // K=3 => R=1
    const int R = 1;
    const int sW = TILE_X + 3 - 1 + SMEM_SKEW; // halo in X + skew
    const int sH = TILE_Y + 3 - 1;             // halo in Y

    extern __shared__ float sY[]; // single luminance plane size = sW*sH

    const int bx = blockIdx.x * TILE_X;
    const int by = blockIdx.y * TILE_Y;

    // Load tile + halo into sY
    for (int sy = threadIdx.y; sy < sH; sy += blockDim.y) {
        int iy = reflect101(by + sy - R, height);
        for (int sx = threadIdx.x; sx < sW - SMEM_SKEW; sx += blockDim.x) {
            int ix = reflect101(bx + sx - R, width);
            int g = iy * row_stride_elems + ix * CH;

#if __CUDA_ARCH__ >= 350
            if (CH == 1) {
                sY[sy * sW + sx] = __ldg(&d_in[g]);
            }
            else {
                float r = __ldg(&d_in[g + 0]);
                float gc = __ldg(&d_in[g + 1]);
                float b = __ldg(&d_in[g + 2]);
                sY[sy * sW + sx] = 0.299f * r + 0.587f * gc + 0.114f * b;
            }
#else
            if (CH == 1) {
                sY[sy * sW + sx] = d_in[g];
            }
            else {
                float r = d_in[g + 0];
                float gc = d_in[g + 1];
                float b = d_in[g + 2];
                sY[sy * sW + sx] = 0.299f * r + 0.587f * gc + 0.114f * b;
            }
#endif
        }
    }
    __syncthreads();

    int ox = bx + threadIdx.x;
    int oy = by + threadIdx.y;
    if (ox < width && oy < height) {
        int s0 = threadIdx.y * sW + threadIdx.x;

        // double accumulators
        double gx = 0.0, gy = 0.0;

#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                float yv = sY[s0 + ky * sW + kx];
                float kxv = c_sobelX3x3[ky * 3 + kx];
                float kyv = c_sobelY3x3[ky * 3 + kx];
                gx += (double)kxv * (double)yv;
                gy += (double)kyv * (double)yv;
            }
        }

        float mag = clamp255((float)sqrt(gx * gx + gy * gy));
        int o = oy * row_stride_elems + ox * CH;
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            d_out[o + c] = mag;
        }
    }
}


/***************************Sobel exploration end ********************************************/
/*********************************************************************************************/

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

static int parse_cuda_variant(int argc, char** argv) {
    // 0,1,2,3 (default 2 =  tiled RGB)
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--cuda-variant") && i + 1 < argc) return atoi(argv[i + 1]);
        if (!strncmp(argv[i], "--cuda-variant=", 15)) return atoi(argv[i] + 15);
    }
    return 2;
}


int main(int argc, char* argv[]) {
    // --- detect if the user asked for NPP explicitly (parse_backend may not know "npp")
    auto is_backend_str = [&](const char* key)->bool {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
                return std::strcmp(argv[i + 1], key) == 0;
            }
        }
        return false;
        };
    const bool backendIsNPP = is_backend_str("npp");

    wbArg_t arg = wbArg_read(argc, argv);
    Op      op = parse_op(argc, argv);
    Backend backend = parse_backend(argc, argv);
    int     threads = parse_threads(argc, argv);  // may be 0/None => use all cores
    int     sobelVar = parse_cuda_variant(argc, argv);

    int nth = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);
    if (backend == BK_MT) {
        nth = (threads > 0) ? threads : omp_get_max_threads();
    }
    else {
        nth = 1;
    }
    omp_set_num_threads(nth);
    printf("CPU backend: %s, threads=%d (omp_get_max_threads=%d)\n",
        (backend == BK_MT ? "MT" : "ST"), nth, omp_get_max_threads());
#else
    printf("CPU backend: %s, threads=%d (OpenMP not enabled)\n",
        (backend == BK_MT ? "MT" : "ST"), (backend == BK_MT ? threads : 1));
#endif

    // ---- Load input via wb (floats in [0,1]) ----
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
    float* h_out = wbImage_getData(outputImage);  // [0,1] result

    const size_t nElems = (size_t)width * height * CHANNELS;
    const int    row_stride = width * CHANNELS;

    // Host masks (shared by CPU/CUDA/NPP)
    static const float h_sharpen3[9] = {
        -1.f, -1.f, -1.f,
        -1.f,  9.f, -1.f,
        -1.f, -1.f, -1.f
    };
    static const float h_sobelX[9] = { -1.f, 0.f, 1.f,  -2.f, 0.f, 2.f,  -1.f, 0.f, 1.f };
    static const float h_sobelY[9] = { 1.f, 2.f, 1.f,   0.f, 0.f, 0.f,  -1.f,-2.f,-1.f };

    // ========================= CPU / CPU-MT BACKEND =========================
    // (Unchanged)  only runs when NOT CUDA and NOT NPP
    if ((backend != BK_CUDA) && !backendIsNPP) {
        std::vector<float> in(nElems), out(nElems);

        // Scale input [0,1] -> [0,255]
        wbTime_start(Compute, "Scale input 0..1 -> 0..255 (CPU)");
        for (size_t i = 0; i < nElems; ++i) in[i] = h_in[i] * 255.0f;
        wbTime_stop(Compute, "Scale input 0..1 -> 0..255 (CPU)");

        const bool isMT = (backend == BK_MT);
        const char* tag = isMT ? " (CPU MT)" : " (CPU)";

#ifdef _OPENMP
        if (isMT) { omp_set_dynamic(0); omp_set_num_threads(nth); }
#endif
        if (op == OP_SHARPEN) {
            std::string lbl = std::string("Sharpen kernel") + tag;
            wbTime_start(Compute, lbl.c_str());
            conv2d_single_mask_reflect101_cpu<3, CHANNELS>(
                in.data(), out.data(), width, height, row_stride, h_sharpen3, nth);
            wbTime_stop(Compute, lbl.c_str());
        }
        else if (op == OP_SOBEL) {
            std::string lbl = std::string("Sobel fused kernel") + tag;
            wbTime_start(Compute, lbl.c_str());
            sobel_fused_reflect101_rgb_cpu<3, CHANNELS>(
                in.data(), out.data(), width, height, row_stride, h_sobelX, h_sobelY, nth);
            wbTime_stop(Compute, lbl.c_str());
        }
        else { // OP_GAUSSIAN (5x5 separable)
            std::vector<float> g5; make_gaussian_kernel_1d(5, 0.0, g5);

            // Horizontal pass: in -> tmp
            std::vector<float> tmp(nElems);
            {
                std::string lbl = std::string("Gaussian horiz") + tag;
                wbTime_start(Compute, lbl.c_str());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if(nth>1)
#endif
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        for (int c = 0; c < CHANNELS; ++c) {
                            float acc = 0.0f;
                            for (int k = 0; k < 5; ++k) {
                                int ix = reflect101(x + k - 2, width);
                                acc += g5[k] * in[y * row_stride + ix * CHANNELS + c];
                            }
                            tmp[y * row_stride + x * CHANNELS + c] = clamp255_h(acc);
                        }
                    }
                }
                wbTime_stop(Compute, lbl.c_str());
            }
            // Vertical pass: tmp -> out
            {
                std::string lbl = std::string("Gaussian vert") + tag;
                wbTime_start(Compute, lbl.c_str());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if(nth>1)
#endif
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        for (int c = 0; c < CHANNELS; ++c) {
                            float acc = 0.0f;
                            for (int k = 0; k < 5; ++k) {
                                int iy = reflect101(y + k - 2, height);
                                acc += g5[k] * tmp[iy * row_stride + x * CHANNELS + c];
                            }
                            out[y * row_stride + x * CHANNELS + c] = clamp255_h(acc);
                        }
                    }
                }
                wbTime_stop(Compute, lbl.c_str());
            }
        }

        // Scale output back to [0,1]
        wbTime_start(Compute, "Scale output 0..255 -> 0..1 (CPU)");
        for (size_t i = 0; i < nElems; ++i) h_out[i] = out[i] * (1.0f / 255.0f);
        wbTime_stop(Compute, "Scale output 0..255 -> 0..1 (CPU)");

        wbSolution(arg, outputImage);
        wbImage_delete(outputImage);
        wbImage_delete(inputImage);
        return 0;
    }

    // ============================== NPP BACKEND ==============================
    if (backendIsNPP) {
        wbTime_start(GPU, "GPU total");

        float* d_in = nullptr, * d_out = nullptr;
        wbTime_start(GPU, "cudaMalloc");
        wbCheck(cudaMalloc(&d_in, nElems * sizeof(float)));
        wbCheck(cudaMalloc(&d_out, nElems * sizeof(float)));
        wbTime_stop(GPU, "cudaMalloc");

        // H2D
        wbTime_start(Copy, "H2D");
        wbCheck(cudaMemcpy(d_in, h_in, nElems * sizeof(float), cudaMemcpyHostToDevice));
        wbTime_stop(Copy, "H2D");

        // Scale input [0,1] -> [0,255] (on GPU, same labels as CUDA path)
        {
            wbTime_start(Compute, "Scale input 0..1 -> 0..255");
            int tpb = 256;
            int blks = (int)((nElems + tpb - 1) / tpb);
            scale_inplace << <blks, tpb >> > (d_in, nElems, 255.0f);
            wbCheck(cudaGetLastError());
            wbCheck(cudaDeviceSynchronize());
            wbTime_stop(Compute, "Scale input 0..1 -> 0..255");
        }

        const int stepBytesC1 = width * sizeof(float);
        const int stepBytesC3 = width * CHANNELS * sizeof(float);
        const NppiSize roi{ width, height };

        if (op == OP_SHARPEN) {
            // NPP 3x3 linear filter in float32
            wbTime_start(Compute, "Sharpen kernel");
            if (CHANNELS == 1) {
                NppiSize ksz{ 3,3 }; NppiPoint anchor{ 1,1 };
                NppStatus st = nppiFilterBorder_32f_C1R(
                    d_in, stepBytesC1, NppiSize{ width,height }, NppiPoint{ 0,0 },
                    d_out, stepBytesC1, roi,
                    h_sharpen3, ksz, anchor, NPP_BORDER_REPLICATE);
                if (st != NPP_SUCCESS) fprintf(stderr, "nppiFilterBorder_32f_C1R (sharpen) st=%d\n", st);
            }
            else if (CHANNELS == 3) {
                NppiSize ksz{ 3,3 }; NppiPoint anchor{ 1,1 };
                NppStatus st = nppiFilterBorder_32f_C3R(
                    d_in, stepBytesC3, NppiSize{ width,height }, NppiPoint{ 0,0 },
                    d_out, stepBytesC3, roi,
                    h_sharpen3, ksz, anchor, NPP_BORDER_REPLICATE);
                if (st != NPP_SUCCESS) fprintf(stderr, "nppiFilterBorder_32f_C3R (sharpen) st=%d\n", st);
            }
            else {
                fprintf(stderr, "[npp] unsupported CHANNELS=%d\n", CHANNELS);
                cudaFree(d_in); cudaFree(d_out);
                wbImage_delete(outputImage); wbImage_delete(inputImage);
                return 1;
            }
            wbTime_stop(Compute, "Sharpen kernel");

        }
        else if (op == OP_SOBEL) {
            wbTime_start(Compute, "Sobel fused kernel");

            float* d_dx = nullptr, * d_dy = nullptr;
            wbCheck(cudaMalloc(&d_dx, nElems * sizeof(float)));
            wbCheck(cudaMalloc(&d_dy, nElems * sizeof(float)));

            NppiSize roi{ width, height };
            NppiSize ksz{ 3,3 }; NppiPoint anchor{ 1,1 };

            if (CHANNELS == 1) {
                // dx, dy via NPP 3x3 conv on C1
                NppStatus st = nppiFilter_32f_C1R(d_in, width * sizeof(float),
                    d_dx, width * sizeof(float), roi,
                    h_sobelX, ksz, anchor);
                if (st != NPP_SUCCESS) fprintf(stderr, "nppiFilter_32f_C1R(dx) st=%d\n", st);
                st = nppiFilter_32f_C1R(d_in, width * sizeof(float),
                    d_dy, width * sizeof(float), roi,
                    h_sobelY, ksz, anchor);
                if (st != NPP_SUCCESS) fprintf(stderr, "nppiFilter_32f_C1R(dy) st=%d\n", st);

                // magnitude on CUDA kernel (avoids nppiSqr/Add/Sqrt)
                int n = width * height, tpb = 256, blks = (n + tpb - 1) / tpb;
                mag_c1 << <blks, tpb >> > (d_dx, d_dy, d_out, n);
                wbCheck(cudaGetLastError());
                wbCheck(cudaDeviceSynchronize());

            }
            else if (CHANNELS == 3) {
                // dx, dy via NPP 3x3 conv on C3
                const int stepBytesC3 = width * CHANNELS * sizeof(float);
                NppStatus st = nppiFilter_32f_C3R(d_in, stepBytesC3,
                    d_dx, stepBytesC3, roi,
                    h_sobelX, ksz, anchor);
                if (st != NPP_SUCCESS) fprintf(stderr, "nppiFilter_32f_C3R(dx) st=%d\n", st);
                st = nppiFilter_32f_C3R(d_in, stepBytesC3,
                    d_dy, stepBytesC3, roi,
                    h_sobelY, ksz, anchor);
                if (st != NPP_SUCCESS) fprintf(stderr, "nppiFilter_32f_C3R(dy) st=%d\n", st);

                // magnitude on CUDA kernel (per-channel)
                int nPix = width * height, tpb = 256, blks = (nPix + tpb - 1) / tpb;
                mag_c3 << <blks, tpb >> > (d_dx, d_dy, d_out, nPix);
                wbCheck(cudaGetLastError());
                wbCheck(cudaDeviceSynchronize());

            }
            else {
                fprintf(stderr, "[npp] unsupported CHANNELS=%d\n", CHANNELS);
                cudaFree(d_dx); cudaFree(d_dy);
                cudaFree(d_in); cudaFree(d_out);
                wbImage_delete(outputImage); wbImage_delete(inputImage);
                return 1;
            }

            cudaFree(d_dx);
            cudaFree(d_dy);
            wbTime_stop(Compute, "Sobel fused kernel");

        }
        else { // OP_GAUSSIAN (single-call)
            wbTime_start(Compute, "Gaussian kernel");
            if (CHANNELS == 1) {
                NppStatus st = nppiFilterGaussBorder_32f_C1R(
                    d_in, stepBytesC1, NppiSize{ width,height }, NppiPoint{ 0,0 },
                    d_out, stepBytesC1, roi,
                    NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE);
                if (st != NPP_SUCCESS) fprintf(stderr, "nppiFilterGaussBorder_32f_C1R st=%d\n", st);
            }
            else if (CHANNELS == 3) {
                NppStatus st = nppiFilterGaussBorder_32f_C3R(
                    d_in, stepBytesC3, NppiSize{ width,height }, NppiPoint{ 0,0 },
                    d_out, stepBytesC3, roi,
                    NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE);
                if (st != NPP_SUCCESS) fprintf(stderr, "nppiFilterGaussBorder_32f_C3R st=%d\n", st);
            }
            else {
                fprintf(stderr, "[npp] unsupported CHANNELS=%d\n", CHANNELS);
                cudaFree(d_in); cudaFree(d_out);
                wbImage_delete(outputImage); wbImage_delete(inputImage);
                return 1;
            }
            wbTime_stop(Compute, "Gaussian kernel");
        }

        // Scale output back to [0,1] on GPU (keep labels identical to CUDA path)
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

        wbImage_delete(outputImage);
        wbImage_delete(inputImage);
        return 0;
    }

    // ============================= CUDA BACKEND =============================
    wbTime_start(GPU, "GPU total");

    float* d_in = nullptr, * d_out = nullptr;
    float* d_sharp = nullptr, * d_kx = nullptr, * d_ky = nullptr;
    float* d_gmask = nullptr, * d_tmp = nullptr;

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
        wbCheck(cudaMalloc(&d_gmask, 5 * sizeof(float)));
        wbCheck(cudaMalloc(&d_tmp, nElems * sizeof(float)));
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
    else {
        std::vector<float> g5;
        make_gaussian_kernel_1d(5, 0.0, g5);
        wbCheck(cudaMemcpy(d_gmask, g5.data(), 5 * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (op == OP_SOBEL && (sobelVar == 1 || sobelVar == 3)) {
        wbCheck(cudaMemcpyToSymbol(c_sobelX3x3, h_sobelX, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
        wbCheck(cudaMemcpyToSymbol(c_sobelY3x3, h_sobelY, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));
    }
    wbTime_stop(Copy, "H2D");

    // Scale input [0,1] -> [0,255]
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
        if (sobelVar == 0) {
            sobel_v0 << <grid, block >> > (
                d_in, d_out, width, height, width * CHANNELS, CHANNELS,
                d_kx, d_ky
                );
        }
        else if (sobelVar == 1) {
            sobel_v1_ro_const << <grid, block >> > (
                d_in, d_out, width, height, width * CHANNELS, CHANNELS
                );
        }
        else if (sobelVar == 2) {
            conv2d_sobel_fused_reflect101_rgb<3, CHANNELS> << <grid, block, smem_bytes_conv(3) >> > (
                d_in, d_out, width, height,
                width * CHANNELS,
                d_kx, d_ky
                );
        }
        else { // 3
            size_t smem = (size_t)(TILE_X + 3 - 1 + SMEM_SKEW)
                * (TILE_Y + 3 - 1)
                * sizeof(float);
            sobel_v3_tiled_y<CHANNELS> << <grid, block, smem >> > (
                d_in, d_out, width, height, width * CHANNELS
                );
        }
        wbCheck(cudaGetLastError());
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Sobel fused kernel");

    }
    else { // OP_GAUSSIAN
        wbTime_start(Compute, "Gaussian horiz");
        gauss1d_horiz_reflect101<5, CHANNELS> << <grid, block, smem_bytes_gauss_h(5) >> > (
            d_in, d_tmp, width, height,
            width * CHANNELS,
            d_gmask
            );
        wbCheck(cudaGetLastError());
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Gaussian horiz");

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

    // Scale output back to [0,1]
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
