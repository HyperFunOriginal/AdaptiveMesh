
#ifndef CUDA_MEM_H
#define CUDA_MEM_H

#if defined(__CUDACC__) // NVCC
#define ALIGN_BYTES(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define ALIGN_BYTES(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN_BYTES(n) __declspec(align(n))
#else
#error "Please provide a definition for ALIGN_BYTES macro for your host compiler!"
#endif

#include "cuda_runtime.h"
#include <string>
#include <stdexcept>

template <class T>
static cudaError_t cuda_alloc_buffer(T** buffer_ptr, const size_t& buffer_len)
{
    cudaError_t cudaStatus = cudaMalloc((void**)buffer_ptr, buffer_len * sizeof(T));;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, ("Allocation of " + std::to_string(buffer_len) + "length buffer failed!\n").c_str());
    }
    return cudaStatus;
}

template <class T>
static cudaError_t cuda_copytogpu_buffer(const T* cpuB, T* gpuB, const size_t& buffer_len)
{
    cudaError_t cudaStatus = cudaMemcpy(gpuB, cpuB, buffer_len * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy to GPU memory.\n");
    }
    return cudaStatus;
}

template <class T>
static cudaError_t cuda_copyfromgpu_buffer(T* cpuB, const T* gpuB, const size_t& buffer_len)
{
    cudaError_t cudaStatus = cudaMemcpy(cpuB, gpuB, buffer_len * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy to CPU memory.\n");
    }
    return cudaStatus;
}

template <class T>
struct smart_cpu_buffer
{
    unsigned int temp_data;
    size_t dedicated_len;
    T* cpu_buffer_ptr;
    bool created;
    bool reference;

    T& operator[](const size_t index)
    {
        if (!created)
            throw std::exception("Buffer unavailable.");
        return cpu_buffer_ptr[min(index, dedicated_len)];
    }
    operator T* ()
    {
        if (!created)
            throw std::exception("Buffer already freed.");
        return cpu_buffer_ptr;
    }
    smart_cpu_buffer() : dedicated_len(0), temp_data(0), created(false), cpu_buffer_ptr(nullptr), reference(false)
    {

    }
    smart_cpu_buffer(size_t max_size) : dedicated_len(max_size), created(true), temp_data(0), reference(false)
    {
        cpu_buffer_ptr = new T[max_size];
    }
    size_t total_size() const { return dedicated_len * sizeof(T); }

    void destroy()
    {
        if (!created || reference) { return; }
        delete[] cpu_buffer_ptr;
        temp_data = 0;
        created = false;
    }
};

template <class T>
struct smart_gpu_buffer
{
    unsigned int temp_data;
    size_t dedicated_len;
    T* gpu_buffer_ptr;
    bool created;
    bool reference;

    size_t total_size() const { return dedicated_len * sizeof(T); }
    void swap_pointers(smart_gpu_buffer<T>& other)
    {
        if (!dedicated_len == other.dedicated_len)
            throw std::exception("Cannot swap buffers of differing lengths!");

        T* temp = gpu_buffer_ptr;
        gpu_buffer_ptr = other.gpu_buffer_ptr;
        other.gpu_buffer_ptr = temp;
    }

    operator T* ()
    {
        if (!created)
            throw std::exception("Buffer already freed or badly allocated.");
        return gpu_buffer_ptr;
    }
    smart_gpu_buffer() : dedicated_len(0), temp_data(0), created(false), gpu_buffer_ptr(nullptr), reference(false)
    {

    }
    smart_gpu_buffer(size_t max_size) : dedicated_len(max_size), temp_data(0), reference(false)
    {
        gpu_buffer_ptr = 0;
        if (cuda_alloc_buffer(&gpu_buffer_ptr, dedicated_len) != cudaSuccess)
            cudaFree(gpu_buffer_ptr);
        else
            created = true;
    }
    virtual void destroy()
    {
        if (reference) { return; }
        if (created)
            cudaFree(gpu_buffer_ptr);
        created = false;
    }
};

template <class T>
struct smart_gpu_cpu_buffer : smart_gpu_buffer<T>
{
    T* cpu_buffer_ptr;

    smart_gpu_cpu_buffer(smart_cpu_buffer<T>& cpu_buffer, bool destroy_old_success, bool destroy_old_failure) : smart_gpu_buffer<T>(cpu_buffer.dedicated_len)
    {
        if (created)
        {
            cpu_buffer_ptr = cpu_buffer.cpu_buffer_ptr; 

            if (destroy_old_success)
            {
                cpu_buffer.destroy();
            }
        }
        else if (destroy_old_failure)
        {
            cpu_buffer.destroy();
        }
    }

    smart_gpu_cpu_buffer() : smart_gpu_buffer<T>()
    {

    }
    void swap_gpu_pointers(smart_gpu_buffer<T>& other)
    {
        if (!dedicated_len == other.dedicated_len)
            throw std::exception("Cannot swap buffers of differing lengths!");

        T* temp = gpu_buffer_ptr;
        gpu_buffer_ptr = other.gpu_buffer_ptr;
        other.gpu_buffer_ptr = temp;
    }
    smart_gpu_cpu_buffer(size_t max_size) : smart_gpu_buffer<T>(max_size)
    {
        if (created)
            cpu_buffer_ptr = new T[max_size];
    }

    cudaError_t copy_to_cpu()
    {
        return cuda_copyfromgpu_buffer<T>(cpu_buffer_ptr, gpu_buffer_ptr, dedicated_len);
    }
    cudaError_t copy_to_gpu()
    {
        return cuda_copytogpu_buffer<T>(cpu_buffer_ptr, gpu_buffer_ptr, dedicated_len);
    }

    void destroy() override
    {
        if (reference) { return; }
        if (created)
        {
            cudaFree(gpu_buffer_ptr);
            delete[] cpu_buffer_ptr;
        }
        created = false;
    }
};

template <class T>
cudaError_t copy_to_cpu(const smart_gpu_buffer<T>& a, smart_cpu_buffer<T>& b, const size_t copy_count = ~0ull)
{
    size_t copy_len_true = a.dedicated_len < copy_count ? a.dedicated_len : copy_count;
    copy_len_true = copy_len_true < b.dedicated_len ? copy_len_true : b.dedicated_len;
    return cuda_copyfromgpu_buffer<T>(b.cpu_buffer_ptr, a.gpu_buffer_ptr, copy_len_true);
}
template <class T>
cudaError_t copy_to_gpu(smart_gpu_buffer<T>& a, const smart_cpu_buffer<T>& b, const size_t copy_count = ~0ull)
{
    size_t copy_len_true = a.dedicated_len < copy_count ? a.dedicated_len : copy_count;
    copy_len_true = copy_len_true < b.dedicated_len ? copy_len_true : b.dedicated_len;
    return cuda_copytogpu_buffer<T>(b.cpu_buffer_ptr, a.gpu_buffer_ptr, a.dedicated_len);
}

static cudaError_t cuda_sync()
{
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernels!\n", cudaStatus);
    }
    return cudaStatus;
}

#pragma message("NOTE: -fno-strict-aliasing command required for compilation.")
template <class T1, class T2>
static smart_cpu_buffer<T2> reinterpret_cast_buffer_by_reference(const smart_cpu_buffer<T1>& buffer)
{
    smart_cpu_buffer<T2> result;
    result.cpu_buffer_ptr = (T2*)buffer.cpu_buffer_ptr;
    result.dedicated_len = (buffer.dedicated_len * sizeof(T1)) / sizeof(T2);
    result.temp_data = buffer.temp_data;
    result.created = buffer.created;
    result.reference = true;
    return result;
}
template <class T1, class T2>
static smart_gpu_buffer<T2> reinterpret_cast_buffer_by_reference(const smart_gpu_buffer<T1>& buffer)
{
    smart_gpu_buffer<T2> result;
    result.gpu_buffer_ptr = (T2*)buffer.gpu_buffer_ptr;
    result.dedicated_len = (buffer.dedicated_len * sizeof(T1)) / sizeof(T2);
    result.temp_data = buffer.temp_data;
    result.created = buffer.created;
    result.reference = true;
    return result;
}
template <class T1, class T2>
static smart_gpu_cpu_buffer<T2> reinterpret_cast_buffer_by_reference(const smart_gpu_cpu_buffer<T1>& buffer)
{
    smart_gpu_cpu_buffer<T2> result;
    result.gpu_buffer_ptr = (T2*)buffer.gpu_buffer_ptr;
    result.cpu_buffer_ptr = (T2*)buffer.cpu_buffer_ptr;
    result.dedicated_len = (buffer.dedicated_len * sizeof(T1)) / sizeof(T2);
    result.temp_data = buffer.temp_data;
    result.created = buffer.created;
    result.reference = true;
    return result;
}

template <class T1, class T2>
__global__ void ___cast(const T1* buff1, T2* buff2, const size_t length)
{
    size_t idx = threadIdx.x + size_t(blockDim.x) * blockIdx.x;
    if (idx < length) { buff2[idx] = (T2)buff1[idx]; }
}
template <class T1, class T2>
static void cast_buffer_copy(const smart_gpu_buffer<T1>& src, smart_gpu_buffer<T2>& dest)
{
    size_t len = src.dedicated_len < dest.dedicated_len ? src.dedicated_len : dest.dedicated_len;
    unsigned int threads = len < 512u ? len : 512u, blocks = ceilf(len / (float)threads);
    ___cast<<<blocks, threads>>>(src.gpu_buffer_ptr, dest.gpu_buffer_ptr, len);
}

template <class T>
__global__ void ___memset_generic(T* buff, const T data, const size_t length)
{
    size_t idx = threadIdx.x + size_t(blockDim.x) * blockIdx.x;
    if (idx < length) { buff[idx] = data; }
}
template <class T1, class T2>
static void memset_gpu_typepun(smart_gpu_buffer<T1>& dest, const T2 data, const size_t set_count_og = ~0ull, const size_t offset_count_og = 0ull)
{
    T2* typepunned = (T2*)(dest.gpu_buffer_ptr + offset_count_og);
    size_t len = (min((unsigned int)dest.dedicated_len - offset_count_og, set_count_og) * sizeof(T1)) / sizeof(T2);
    unsigned int threads = len < 512u ? len : 512u, blocks = ceilf(len / float(threads));
    ___memset_generic<<<blocks, threads>>>(typepunned, data, len);
}

template <class T>
__global__ void __multiply_buffer(T* multiplicand, const T multplier, const size_t length)
{
    size_t idx = threadIdx.x + size_t(blockDim.x) * blockIdx.x;
    if (idx >= length) { return; }
    multiplicand[idx] = multiplicand[idx] * multplier;
}
template <class T>
static void multiply_gpu_buffer(smart_gpu_buffer<T>& buffer, const T multiplier, const size_t set_count_og = ~0ull, const size_t offset_count_og = 0ull)
{
    size_t len = min((unsigned int)buffer.dedicated_len - offset_count_og, set_count_og);
    unsigned int threads = len < 512u ? len : 512u, blocks = ceilf(len / float(threads));
    __multiply_buffer<T><<<blocks, threads>>>(buffer.gpu_buffer_ptr + offset_count_og, multiplier, len);
}

template <class T>
__global__ void __divide_buffer(T* dividend, const T divider, const size_t length)
{
    size_t idx = threadIdx.x + size_t(blockDim.x) * blockIdx.x;
    if (idx >= length) { return; }
    dividend[idx] = dividend[idx] / divider;
}
template <class T>
static void divide_gpu_buffer(smart_gpu_buffer<T>& buffer, const T multiplier, const size_t set_count_og = ~0ull, const size_t offset_count_og = 0ull)
{
    size_t len = min((unsigned int)buffer.dedicated_len - offset_count_og, set_count_og);
    unsigned int threads = len < 512u ? len : 512u, blocks = ceilf(len / float(threads));
    __divide_buffer<T><<<blocks, threads>>>(buffer.gpu_buffer_ptr + offset_count_og, multiplier, len);
}

#endif