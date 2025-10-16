
#include "AMR.h"
#include "image_process.h"
#include <stdio.h>
#include <chrono>
#include "tensors.h"

__global__ void __init_temp(float* __restrict__ old_d, float* __restrict__ new_d, const octree_abs_pos* positions)
{
    uint3 idx = threadIdx + blockDim * blockIdx;
    uint node_idx = idx.z / total_size_domain; 
    idx.z -= node_idx * total_size_domain;

    const int depth = positions[node_idx].depth(); if (depth == -1) { return; }
    float3 true_position = positions[node_idx].absolute_central_position();
    true_position += (make_float3(idx) + .5f - (total_size_domain * .5f)) * (outer_size / size_domain) / (1u << depth);
    const uint pos = (node_idx * cells_domain) + (idx.z * total_size_domain + idx.y) * total_size_domain + idx.x;
    
    old_d[pos] = 1.f / length(true_position);
    new_d[pos] = 1.f / length(true_position);
}
void init_temp(smart_gpu_buffer<float>& old_d, smart_gpu_buffer<float>& new_d, AMR<blank_AMR_data>& amr)
{
    dim3 threads(32u, 8u, 4u);
    dim3 blocks(1u, 4u, amr.curr_used_slots() * 8u);
    
    amr.copy_to_gpu();
    __init_temp<<<blocks, threads>>>(old_d.gpu_buffer_ptr, new_d.gpu_buffer_ptr, amr.positions.gpu_buffer_ptr);
}

int main()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        _sleep(10000u);
        return 1;
    }

    AMR<blank_AMR_data> amr = AMR<blank_AMR_data>(128u);
    int children[8];
    
    for (uint c = 0; c < 8; c++)
    {
        children[c] = amr.read_first_free_slot();
        amr.add_node(0, c);
    }
    
    for (uint d = 1; d < 6; d++)
        for (uint c = 0; c < 8; c++)
        {
            uint new_idx = amr.read_first_free_slot();
            amr.add_node(children[c], (~c)&7u);
            children[c] = new_idx;
        }

    round_robin_threads threads;
    std::cout << amr.to_string_debug();
    smart_gpu_buffer<float> old_d(amr.max_slots * cells_domain), new_d(amr.max_slots * cells_domain);
    smart_gpu_buffer<compressed_float3> deriv(amr.max_slots * size_domain * size_domain * size_domain);
    init_temp(old_d, new_d, amr);

    cuda_sync();
    copy_boundaries(old_d, new_d, amr, false); // too inaccurate
    cuda_sync();
    copy_new_to_old(old_d, new_d, amr);
    cuda_sync();
    mixed_partials(old_d, deriv, amr, threads.yield_stream());
    cuda_sync();
    smart_gpu_cpu_buffer<uint> dat(deriv.dedicated_len);
    save_image(dat, deriv, size_domain * size_domain, deriv.dedicated_len / (size_domain * size_domain), "test.png");

    while (true)
        _sleep(1000u);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        _sleep(10000u);
        return 1;
    }

    return 0;
}
