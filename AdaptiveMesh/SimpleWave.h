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

__device__ constexpr uint threadsE()
{
	for (uint i = 32u; i > 0u; i--)
		if (size_domain % i == 0)
			return i;
	return 1u;
}
__device__ constexpr uint threadsF()
{
	for (uint i = 16u; i > 0u; i--)
		if (size_domain % i == 0)
			return i;
	return 1u;
}
__device__ constexpr uint threadsE_v = threadsE();
__device__ constexpr uint threadsF_v = threadsF();

__global__ void __amr_criterion_1(const compressed_float3* derivs, const compressed_float3* hess_diag,
	float* amr_criterion, const octree_abs_pos* data, const uint substep_index, const uint max_depth) {

	uint3 idx = threadIdx + blockDim * blockIdx;
	const int depth = data[idx.z / 2u].depth();
	if (depth == -1 || !active_depth(substep_index, depth, max_depth)) {
		amr_criterion[idx.x + (idx.y + idx.z * (size_domain >> 1u)) * size_domain] = NAN;
		return; 
	}

	uint read_offset = idx.x + (idx.y + ((idx.y >= (size_domain >> 2u))
		? (size_domain >> 2u) : 0u) + idx.z * size_domain * (size_domain >> 1u)) * size_domain;

	float criterion = 0.f;
	for (uint i = 0; i < size_domain >> 1u; i++)
	{
		compressed_float3 d = derivs[read_offset];
		criterion = fmaxf(criterion, length((float3)d) * outer_dx / (1u << depth));
		d = hess_diag[read_offset];
		criterion = fmaxf(criterion, length((float3)d) * outer_dx * outer_dx / (1u << (depth * 2u)));
		
		d = derivs[read_offset + (size_domain >> 2u) * size_domain];
		criterion = fmaxf(criterion, length((float3)d) * outer_dx / (1u << depth));
		d = hess_diag[read_offset + (size_domain >> 2u) * size_domain];
		criterion = fmaxf(criterion, length((float3)d) * outer_dx * outer_dx / (1u << (depth * 2u)));

		read_offset += size_domain * size_domain;
	}
	amr_criterion[idx.x + (idx.y + idx.z * (size_domain >> 1u)) * size_domain] = criterion;
}
// bad strided access, but not performance critical anyway
__global__ void __amr_criterion_2(float* amr_criterion) {

	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	uint x = idx & 1u;
	uint y = (idx >> 1u) & 1u;
	uint z = (idx >> 2u) & 1u;
	uint node_idx = idx >> 3u;

	const uint read_write_offset = node_idx * size_domain * size_domain + z * (size_domain * size_domain >> 1u)
		+ y * (size_domain * size_domain >> 2u) + x * (size_domain >> 1u);

	float crit = NAN;
	for (y = 0; y < (size_domain >> 2u); y++)
		for (x = 0; x < (size_domain >> 1u); x++)
			crit = fmaxf(crit, amr_criterion[read_write_offset + x + y * size_domain]);
	amr_criterion[read_write_offset] = crit;
}

// amr_crit must have length max_nodes * size_domain * size_domain
template <class AMR_data>
float2 amr_criterion(const smart_gpu_buffer<compressed_float3>& first_deriv, const smart_gpu_buffer<compressed_float3>& hess_diag,
	smart_gpu_cpu_buffer<float>& amr_crit, AMR<AMR_data>& amr)
{
	dim3 threads(threadsE_v, threadsF_v, 2u);
	dim3 blocks(size_domain / threads.x, (size_domain >> 1u) / threads.y, amr.curr_used_slots());
	__amr_criterion_1<<<blocks, threads>>>(first_deriv.gpu_buffer_ptr, hess_diag.gpu_buffer_ptr,
		amr_crit.gpu_buffer_ptr, amr.positions.gpu_buffer_ptr, amr.timer_helper, amr.read_max_depth());
	__amr_criterion_2<<<amr.curr_used_slots(), 8u>>>(amr_crit.gpu_buffer_ptr);
	amr_crit.copy_to_cpu();

	float avg_pressure_1 = 0.f, avg_pressure_2 = 0.f; uint active_cells = 0;
	for (uint i = 0, s = amr.curr_used_slots(); i < s; i++)
	{
		amr_crit.cpu_buffer_ptr[i * 8u] = amr_crit.cpu_buffer_ptr[i * size_domain * size_domain];
		amr_crit.cpu_buffer_ptr[i * 8u + 1u] = amr_crit.cpu_buffer_ptr[i * size_domain * size_domain + (size_domain >> 1u)];
		amr_crit.cpu_buffer_ptr[i * 8u + 2u] = amr_crit.cpu_buffer_ptr[i * size_domain * size_domain + (size_domain * size_domain >> 2u)];
		amr_crit.cpu_buffer_ptr[i * 8u + 3u] = amr_crit.cpu_buffer_ptr[i * size_domain * size_domain + ((size_domain >> 1u) + (size_domain * size_domain >> 2u))];
		amr_crit.cpu_buffer_ptr[i * 8u + 4u] = amr_crit.cpu_buffer_ptr[i * size_domain * size_domain + (size_domain * size_domain >> 1u)];
		amr_crit.cpu_buffer_ptr[i * 8u + 5u] = amr_crit.cpu_buffer_ptr[i * size_domain * size_domain + ((size_domain >> 1u) + (size_domain * size_domain >> 1u))];
		amr_crit.cpu_buffer_ptr[i * 8u + 6u] = amr_crit.cpu_buffer_ptr[i * size_domain * size_domain + ((size_domain * size_domain >> 2u) + (size_domain * size_domain >> 1u))];
		amr_crit.cpu_buffer_ptr[i * 8u + 7u] = amr_crit.cpu_buffer_ptr[i * size_domain * size_domain + ((size_domain >> 1u) + (size_domain * size_domain >> 2u) + (size_domain * size_domain >> 1u))];
		for (int j = 0; j < 8; j++)
			if (!isnan(amr_crit.cpu_buffer_ptr[i * 8u + j]))
				avg_pressure_2 += amr_crit.cpu_buffer_ptr[i * 8u + j];
	}
	for (uint i = 0, s = amr.curr_used_slots(); i < s; i++)
	{
		float temp = amr_crit.cpu_buffer_ptr[i * 8u];
		temp = fmaxf(temp, amr_crit.cpu_buffer_ptr[i * 8u + 1u]);
		temp = fmaxf(temp, amr_crit.cpu_buffer_ptr[i * 8u + 2u]);
		temp = fmaxf(temp, amr_crit.cpu_buffer_ptr[i * 8u + 3u]);
		temp = fmaxf(temp, amr_crit.cpu_buffer_ptr[i * 8u + 4u]);
		temp = fmaxf(temp, amr_crit.cpu_buffer_ptr[i * 8u + 5u]);
		temp = fmaxf(temp, amr_crit.cpu_buffer_ptr[i * 8u + 6u]);
		temp = fmaxf(temp, amr_crit.cpu_buffer_ptr[i * 8u + 7u]);
		amr_crit.cpu_buffer_ptr[i + s * 8u] = temp;
		
		if (!isnan(temp))
		{
			avg_pressure_1 += temp;
			active_cells++;
		}
	}
	return make_float2(avg_pressure_1, avg_pressure_2 * .125f) / active_cells;
}

// AMR criterion of child must always be no greater than that for the parent

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

	// AMR criterion
	smart_gpu_cpu_buffer<float> criterion;

	// derivatives
	smart_gpu_buffer<compressed_float3> first_derivs;
	smart_gpu_buffer<compressed_float3> hessian_diag;

	wave_AMR_data(const uint node_slots, AMR<wave_AMR_data>& parent) : parent(parent), streams(),
		old_field(node_slots * cells_domain), new_field(node_slots* cells_domain),
		old_dt(node_slots* cells_domain), new_dt(node_slots* cells_domain),
		criterion(node_slots* size_domain * size_domain),
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
	void node_dtor(const int node_idx) {
	}
	void modify_nodes()
	{
		float2 avg_crit = amr_criterion(first_derivs, hessian_diag, criterion, parent); // computes all refinement pressure and yields average
		if (parent.curr_used_slots() == 1u) { avg_crit.x = avg_crit.y = .13f; }
																						
		// perform triage
		float urgency_pressure = float(parent.curr_used_slots() - 1) / parent.max_slots;
		for (int i = 1, s = parent.curr_used_slots(); i < s; i++) // root cannot be removed
		{
			float depth_ratio = float(parent.positions.cpu_buffer_ptr[i].depth()) / parent.read_max_depth(); if (depth_ratio <= 0.f) { continue; }
			float weight = clamp(parent.lifetime.cpu_buffer_ptr[i] * .1f - .2f, 0.f, 1.f) * urgency_pressure * urgency_pressure * avg_crit.x * sqrtf(depth_ratio); // must run at least 2 timesteps to copy back down info, else domain is wasted.
			if (criterion.cpu_buffer_ptr[s * 8u + i] < weight) // ignores if nan, i.e. no domain
			{
				std::cout << "Removed node " + std::to_string(i) + " with factor " + std::to_string(criterion.cpu_buffer_ptr[s * 8u + i] / weight) + ".\n";
				parent.remove_node(i);
			}
		}
		for (uint i = 0, s = parent.curr_used_slots() * 8u; i < s; i++)
		{
			float weight = avg_crit.y * 3.5f / (1.f - urgency_pressure);
			if (criterion.cpu_buffer_ptr[i] > weight && (parent.children_idx_b.cpu_buffer_ptr[i] == -1))
			{
				std::cout << "Created child with parent " + std::to_string(i>>3u) + ", child " + std::to_string(i & 7) + " with factor " + std::to_string(criterion.cpu_buffer_ptr[i] / weight) + ".\n";
				parent.add_node(i >> 3u, i & 7u);
			}
		}
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
