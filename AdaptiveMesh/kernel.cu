
#include "SimpleWave.h"
#include "BSSN.h"
#include "image_process.h"
#include <stdio.h>
#include <chrono>
#include "tensors.h"

__global__ void __init_temp(float* __restrict__ old_d, float* __restrict__ new_d, const octree_abs_pos* positions, float mul)
{
    uint3 idx = threadIdx + blockDim * blockIdx;
    uint node_idx = idx.z / total_size_domain; 
    idx.z -= node_idx * total_size_domain;

    const int depth = positions[node_idx].depth(); if (depth == -1) { return; }
    if (depth > 0) {
        if (idx.x < padding_domain || idx.x >= size_domain + padding_domain) { return; }
        if (idx.y < padding_domain || idx.y >= size_domain + padding_domain) { return; }
        if (idx.z < padding_domain || idx.z >= size_domain + padding_domain) { return; }
    }
    float3 true_position = positions[node_idx].absolute_central_position();
    true_position += (make_float3(idx) + .5f - (total_size_domain * .5f)) * (outer_size / size_domain) / (1u << depth);
    const uint pos = (node_idx * cells_domain) + (idx.z * total_size_domain + idx.y) * total_size_domain + idx.x;
    
    //old_d[pos] = mul / length(true_position);
    //new_d[pos] = mul / length(true_position);

    old_d[pos] = mul / (dot(true_position,true_position) + 1.f);
    new_d[pos] = mul / (dot(true_position,true_position) + 1.f);
    
    //old_d[pos] = mul + sinf(length(true_position)) * .33f;
    //new_d[pos] = mul + sinf(length(true_position)) * .33f;
}
template <class T>
void init_temp(smart_gpu_buffer<float>& old_d, smart_gpu_buffer<float>& new_d, AMR<T>& amr, float mul)
{
    dim3 threads(32u, 8u, 4u);
    dim3 blocks(1u, 4u, amr.curr_used_slots() * 8u);
    
    amr.copy_to_gpu();
    __init_temp<<<blocks, threads>>>(old_d.gpu_buffer_ptr, new_d.gpu_buffer_ptr, amr.positions.gpu_buffer_ptr, mul);
}

void wave_test()
{
    AMR<wave_AMR_data> amr = AMR<wave_AMR_data>(128u);
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
            amr.add_node(children[c], (~c) & 7u);
            children[c] = new_idx;
        }

    round_robin_threads threads; fast_prng rng;
    std::cout << amr.to_string_debug() + "\n\n";

    init_temp(amr.sim_data.old_field, amr.sim_data.new_field, amr, 1.f);
    init_temp(amr.sim_data.old_dt, amr.sim_data.new_dt, amr, 5.f);

    // dumb init for testing
    smart_gpu_cpu_buffer<uint> temp(cells_domain * 49u);
    save_image(temp, amr.sim_data.old_field, total_size_domain * total_size_domain, temp.dedicated_len / (total_size_domain * total_size_domain), "test_0.png");
    
    for (uint i = 1; i < 200; i++)
    {
        amr.sim_data.timestep();
        save_image(temp, amr.sim_data.old_field, total_size_domain * total_size_domain, temp.dedicated_len / (total_size_domain * total_size_domain), ("test_" + std::to_string(i) + ".png").c_str());
    }
}

void BSSN_test()
{
    AMR<BSSN_AMR_data> amr = AMR<BSSN_AMR_data>(128u);
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
            amr.add_node(children[c], (~c) & 7u);
            children[c] = new_idx;
        }

    round_robin_threads threads; fast_prng rng;
    std::cout << amr.to_string_debug() + "\n\n";

    // dumb init for testing
    init_temp(amr.sim_data.cyij_old.xx, amr.sim_data.cyij_new.xx, amr, 1.f);
    init_temp(amr.sim_data.cyij_old.yy, amr.sim_data.cyij_new.yy, amr, 1.f);
    init_temp(amr.sim_data.cyij_old.zz, amr.sim_data.cyij_new.zz, amr, 1.f);
    init_temp(amr.sim_data.cyij_old.xy, amr.sim_data.cyij_new.xy, amr, 0.f);
    init_temp(amr.sim_data.cyij_old.xz, amr.sim_data.cyij_new.xz, amr, 0.f);
    init_temp(amr.sim_data.cyij_old.yz, amr.sim_data.cyij_new.yz, amr, 0.f);

    amr.sim_data.timestep();
    std::chrono::steady_clock clock;
    cuda_sync();
    long long now = clock.now().time_since_epoch().count();
    amr.sim_data.timestep();
    cuda_sync();
    long long now2 = clock.now().time_since_epoch().count();
    std::cout << (now2 - now) * 1E-9;

    smart_gpu_cpu_buffer<uint> temp(amr.sim_data.cGijk.xx.dedicated_len);
    save_image(temp, amr.sim_data.cGijk.xx, size_domain * size_domain, temp.dedicated_len / (size_domain * size_domain), "test_xx.png");
    save_image(temp, amr.sim_data.cGijk.xy, size_domain * size_domain, temp.dedicated_len / (size_domain * size_domain), "test_xy.png");
    save_image(temp, amr.sim_data.cGijk.xz, size_domain * size_domain, temp.dedicated_len / (size_domain * size_domain), "test_xz.png");
    save_image(temp, amr.sim_data.cGijk.yy, size_domain * size_domain, temp.dedicated_len / (size_domain * size_domain), "test_yy.png");
    save_image(temp, amr.sim_data.cGijk.yz, size_domain * size_domain, temp.dedicated_len / (size_domain * size_domain), "test_yz.png");
    save_image(temp, amr.sim_data.cGijk.zz, size_domain * size_domain, temp.dedicated_len / (size_domain * size_domain), "test_zz.png");
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

    wave_test();
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
