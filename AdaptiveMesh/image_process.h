#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "helper_math.h"
#include "io_helper.h"
#include "lodepng.h"
#include "device_launch_parameters.h"

template <class T>
inline __host__ __device__ uint ___rgba(const T val)
{
    return 4294902015u; // default
}
template<>
inline __host__ __device__ uint ___rgba(const float4 val)
{
    if (isnan(val.x) || isnan(val.y) || isnan(val.z) || isnan(val.w))
        return 4294902015u;
    const uint4 v = make_uint4(clamp(val, 0.f, 1.f) * 255.f);
    return (v.w << 24) | (v.z << 16) | (v.y << 8) | v.x;
}
template<>
inline __host__ __device__ uint ___rgba(const float3 val)
{
    if (isnan(val.x) || isnan(val.y) || isnan(val.z))
        return 4294902015u;
    const uint3 v = make_uint3(clamp(val, 0.f, 1.f) * 255.f);
    return (255u << 24) | (v.z << 16) | (v.y << 8) | v.x;
}
template<>
inline __host__ __device__ uint ___rgba(const float val)
{
    if (isnan(val))
        return 4294902015u;
    const float temp = clamp(val, 0.f, 1.f) * 255.f;
    const uint4 v = make_uint4(temp, temp, temp, 255u);
    return (v.w << 24) | (v.z << 16) | (v.y << 8) | v.x;
}
#include "tensors.h"
template<>
inline __host__ __device__ uint ___rgba(const compressed_float3 val)
{
    return ___rgba((float3)val);
}

template <class T>
__global__ void __encode_img(uint* pixels, const T* image, const uint width, const uint height)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height)
        return;
    uint idx2 = idx.x + idx.y * width;
    pixels[idx2] = ___rgba(image[idx2]);
}
template <class T>
void save_image(smart_gpu_cpu_buffer<uint>& temp, const smart_gpu_buffer<T>& image, const uint width, const uint height, const char* filename)
{
    const dim3 threads(min_uint(width, 16), min_uint(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    __encode_img << <blocks, threads >> > (temp.gpu_buffer_ptr, image.gpu_buffer_ptr, width, height);
    temp.copy_to_cpu(); cuda_sync(); lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), width, height);
}

__global__ void __decode_img(const uint* pixels, float4* image, const uint width, const uint height)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height)
        return;
    uint idx2 = idx.x + idx.y * width;
    uint pixel_val = pixels[idx2];
    image[idx2] = make_float4(pixel_val & 255u, (pixel_val >> 8u) & 255u, (pixel_val >> 16u) & 255u, (pixel_val >> 24u) & 255u) / 255u;
}
/// <summary>
/// Loads an image from disk and converts it into a float4 buffer. temp_data contains width of image
/// </summary>
/// <param name="filename">Filepath to the image</param>
/// <returns></returns>
smart_gpu_buffer<float4> load_image(const char* filepath)
{
    uint width, height; unsigned char* output;
    if (lodepng_decode32_file(&output, &width, &height, filepath) != 0)
        throw std::exception("Error encountered with file loading.");

    smart_gpu_buffer<uint> temp(width * height);
    cudaMemcpy(temp.gpu_buffer_ptr, output, temp.total_size(), cudaMemcpyHostToDevice);
    smart_gpu_buffer<float4> image(width * height); image.temp_data = width;

    const dim3 threads(min_uint(width, 16), min_uint(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    __decode_img << <blocks, threads >> > (temp.gpu_buffer_ptr, image.gpu_buffer_ptr, width, height);
    cuda_sync(); temp.destroy(); free(output); return image;
}

__global__ void __init_pattern(float4* image, uint width, uint height)
{
    const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    const uint y = idx / width, x = idx - y * width;
    if (y >= height) { return; }

    float xx = x / (float)width;
    float yy = y / (float)height;
    image[idx] = make_float4(xx, yy, 0.f, 1.f);
}
void init_default_image(smart_gpu_buffer<float4>& image, uint width, uint height)
{
    dim3 threads(min_uint(width * height, 512u));
    dim3 blocks(ceilf((width * height) / (float)threads.x));
    __init_pattern<<<blocks, threads>>>(image.gpu_buffer_ptr, width, height);
}

template <class T>
__global__ void __translate_img_half_width(T* inout, const uint width, const uint height)
{
    const uint3 idx = threadIdx + blockDim * blockIdx;
    if (idx.y >= height || (idx.x << 1u) >= width) { return; }

    const uint read_idx_1 = idx.x + idx.y * width;
    const uint read_idx_2 = read_idx_1 + (width >> 1u);
    const T a = inout[read_idx_1];
    inout[read_idx_1] = inout[read_idx_2];
    inout[read_idx_2] = a;
}
template <class T>
__global__ void __translate_img_half_height(T* inout, const uint width, const uint height)
{
    const uint3 idx = threadIdx + blockDim * blockIdx;
    if ((idx.y << 1u) >= height || idx.x >= width) { return; }

    const uint read_idx_1 = idx.x + idx.y * width;
    const uint read_idx_2 = read_idx_1 + width * (height >> 1u);
    const T a = inout[read_idx_1];
    inout[read_idx_1] = inout[read_idx_2];
    inout[read_idx_2] = a;
}

template <class T>
void translate_image_half_width(smart_gpu_buffer<T>& image, const uint width, const uint height)
{
    const dim3 threads(min_uint(width >> 1u, 32), min_uint(height, 16));
    const dim3 blocks((uint)ceilf((width >> 1u) / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    __translate_img_half_width<<<blocks,threads>>>(image.gpu_buffer_ptr, width, height);
}
template <class T>
void translate_image_half_height(smart_gpu_buffer<T>& image, const uint width, const uint height)
{
    const dim3 threads(min_uint(width, 32), min_uint(height >> 1u, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf((height >> 1u) / (float)threads.y));
    __translate_img_half_height<<<blocks,threads>>>(image.gpu_buffer_ptr, width, height);
}


template <class T>
__global__ void __downscale_2x(const T* __restrict__ idata, T* __restrict__ odata, const uint width, const uint height)
{
    uint2 global_idx = make_uint2(threadIdx + blockDim * blockIdx);
    if ((global_idx.x << 1u) < width && (global_idx.y << 1u) < height)
    {
        odata[global_idx.x + global_idx.y * (width >> 1u)] = (idata[(global_idx.x + global_idx.y * width) * 2u] +
                                                              idata[(global_idx.x + global_idx.y * width) * 2u + 1u] +
                                                              idata[(global_idx.x + global_idx.y * width) * 2u + width] +
                                                              idata[(global_idx.x + global_idx.y * width) * 2u + width + 1u]) * .25f;
    }
}
template <class T>
void downscale_image_2x(const smart_gpu_buffer<T>& input, smart_gpu_buffer<T>& output, uint width, uint height)
{
    dim3 threads(min_uint(width, 32u), min_uint(height, 16u));
    dim3 blocks(ceilf(width / ((float)threads.x * 2.f)), ceilf(height / ((float)threads.y * 2.f)));
    __downscale_2x<<<blocks, threads>>>(input.gpu_buffer_ptr, output.gpu_buffer_ptr, width, height);
}

#endif
