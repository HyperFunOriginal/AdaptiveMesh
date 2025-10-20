#ifndef WAVETEST_H
#define WAVETEST_H

#include "AMR.h"

__global__ void __predictor_step(const float* old_field, const float* old_dt, 
	const compressed_float3* hess_diag, float* new_field, float* new_dt,
	const octree_abs_pos* data, const uint substep_index, const uint max_depth) {

	uint3 idx = threadIdx + blockDim * blockIdx;
	const uint node_idx = idx.z / size_domain;
	idx.z -= node_idx * size_domain;

	const int depth = data[node_idx].depth(); if (depth == -1) { return; }
	if (!active_depth(substep_index, depth, max_depth)) { return; }
	int read_write_idx = ((node_idx * size_domain + idx.z) * size_domain + idx.y) * size_domain + idx.x;

	float laplacian = dot((float3)hess_diag[read_write_idx], make_float3(1.f));
	read_write_idx = ((node_idx * total_size_domain + idx.z) * total_size_domain + idx.y) * total_size_domain
		+ idx.x + padding_domain * (1u + total_size_domain + total_size_domain * total_size_domain);
	
	float old_dt_val = old_dt[read_write_idx];
	float new_dt_val = old_dt_val + laplacian * outer_dt / (1u << depth);
	float new_val = old_field[read_write_idx] + (old_dt_val + new_dt_val) * outer_dt / (2u << depth);

	new_dt[read_write_idx] = new_dt_val;
	new_field[read_write_idx] = new_val;
}

__global__ void __corrector_step(const float* old_field, const float* old_dt,
	const compressed_float3* hess_diag, float* new_field, float* new_dt,
	const octree_abs_pos* data, const uint substep_index, const uint max_depth) {

	uint3 idx = threadIdx + blockDim * blockIdx;
	const uint node_idx = idx.z / size_domain;
	idx.z -= node_idx * size_domain;

	const int depth = data[node_idx].depth(); if (depth == -1) { return; }
	if (!active_depth(substep_index, depth, max_depth)) { return; }
	int read_write_idx = ((node_idx * size_domain + idx.z) * size_domain + idx.y) * size_domain + idx.x;

	float laplacian = dot((float3)hess_diag[read_write_idx], make_float3(1.f));
	read_write_idx = ((node_idx * total_size_domain + idx.z) * total_size_domain + idx.y) * total_size_domain
		+ idx.x + padding_domain * (1u + total_size_domain + total_size_domain * total_size_domain);

	float old_dt_val = old_dt[read_write_idx];
	float new_dt_val = old_dt_val + laplacian * outer_dt / (1u << depth);
	float new_val = old_field[read_write_idx] + (old_dt_val + new_dt_val) * outer_dt / (2u << depth);

	new_dt[read_write_idx] = (new_dt[read_write_idx] + new_dt_val) * .5f;
	new_field[read_write_idx] = (new_field[read_write_idx] + new_val) * .5f;
}

template <class AMR_data>
void wave_equation_predictor(const smart_gpu_buffer<float>& old_field, const smart_gpu_buffer<float>& old_dt,
	smart_gpu_buffer<float>& new_field, smart_gpu_buffer<float>& new_dt, smart_gpu_buffer<compressed_float3>& hess_diag,
	AMR<AMR_data>& amr, cudaStream_t& stream)
{
	dim3 threads(threadsA3_v, threadsB3_v, threadsD3_v);
	dim3 blocks(size_domain / threads.x, size_domain / threads.y,
		amr.curr_used_slots() * size_domain / threads.z);
	__predictor_step<<<blocks, threads, 0, stream>>>(old_field.gpu_buffer_ptr, old_dt.gpu_buffer_ptr,
		hess_diag.gpu_buffer_ptr, new_field.gpu_buffer_ptr, new_dt.gpu_buffer_ptr,
		amr.positions.gpu_buffer_ptr, amr.timer_helper, amr.read_max_depth());
}
template <class AMR_data>
void wave_equation_corrector(const smart_gpu_buffer<float>& old_field, const smart_gpu_buffer<float>& old_dt,
	smart_gpu_buffer<float>& new_field, smart_gpu_buffer<float>& new_dt, smart_gpu_buffer<compressed_float3>& hess_diag,
	AMR<AMR_data>& amr, cudaStream_t& stream)
{
	dim3 threads(threadsA3_v, threadsB3_v, threadsD3_v);
	dim3 blocks(size_domain / threads.x, size_domain / threads.y,
		amr.curr_used_slots() * size_domain / threads.z);
	__corrector_step<<<blocks, threads, 0, stream>>>(old_field.gpu_buffer_ptr, old_dt.gpu_buffer_ptr,
		hess_diag.gpu_buffer_ptr, new_field.gpu_buffer_ptr, new_dt.gpu_buffer_ptr,
		amr.positions.gpu_buffer_ptr, amr.timer_helper, amr.read_max_depth());
}

struct wave_AMR_data
{
	fast_prng rng;
	AMR<wave_AMR_data>& parent;
	round_robin_threads streams;

	// fields
	smart_gpu_buffer<float> old_field;
	smart_gpu_buffer<float> new_field;
	smart_gpu_buffer<float> old_dt;
	smart_gpu_buffer<float> new_dt;

	// derivatives
	smart_gpu_buffer<compressed_float3> first_derivs;
	smart_gpu_buffer<compressed_float3> hessian_diag;

	wave_AMR_data(const uint node_slots, AMR<wave_AMR_data>& parent) : parent(parent), streams(),
		old_field(node_slots * cells_domain), new_field(node_slots* cells_domain),
		old_dt(node_slots* cells_domain), new_dt(node_slots* cells_domain),
		first_derivs(node_slots * inner_cells_domain), rng(),
		hessian_diag(node_slots * inner_cells_domain) {}

	void copy_back()
	{
		cuda_sync();
		copy_new_to_old(old_field, new_field, parent, streams.yield_stream());
		copy_new_to_old(old_dt, new_dt, parent, streams.yield_stream());
	}
	void node_ctor(const int node_idx) {
		streams.set_stream_idx(0u); // required for correct synchronization
		copy_to_child(old_field, parent.positions.cpu_buffer_ptr[node_idx].final_offset(),
			node_idx, parent.parent_idx_b.cpu_buffer_ptr[node_idx], streams.yield_stream());
		copy_to_child(old_dt, parent.positions.cpu_buffer_ptr[node_idx].final_offset(),
			node_idx, parent.parent_idx_b.cpu_buffer_ptr[node_idx], streams.yield_stream());
		copy_to_child(new_field, parent.positions.cpu_buffer_ptr[node_idx].final_offset(),
			node_idx, parent.parent_idx_b.cpu_buffer_ptr[node_idx], streams.yield_stream());
		copy_to_child(new_dt, parent.positions.cpu_buffer_ptr[node_idx].final_offset(),
			node_idx, parent.parent_idx_b.cpu_buffer_ptr[node_idx], streams.yield_stream());
	}
	void node_dtor(const int node_idx) {}
	void modify_nodes()
	{

	}
	void predictor()
	{
		cuda_sync();
		cudaStream_t curr = streams.yield_stream();
		differentiate(old_field, first_derivs, hessian_diag, parent, curr, rng);
		wave_equation_predictor(old_field, old_dt, new_field, new_dt, hessian_diag, parent, curr);
	}
	void copy_bounds()
	{
		cuda_sync();
		copy_boundaries(old_field, new_field, parent, streams.yield_stream());
		copy_boundaries(old_dt, new_dt, parent, streams.yield_stream());
	}
	void corrector()
	{
		cuda_sync();
		cudaStream_t curr = streams.yield_stream();
		differentiate(new_field, first_derivs, hessian_diag, parent, curr, rng);
		wave_equation_corrector(old_field, old_dt, new_field, new_dt, hessian_diag, parent, curr);
	}
	void timestep() { // Basic form of timestep loop; must be in this exact order.
		do {
			copy_back();
			modify_nodes();
			predictor();
			copy_bounds();
			corrector();
			parent.increment_timer();
		} while (parent.timer_helper != 0u);
		cuda_sync();
	}
};

#endif // !WAVETEST_H
