#ifndef BSSN_H	
#define BSSN_H

#include "AMR.h"

// Numerical Relativity
__global__ void __deriv_trace(const comp_tensor2_sym_ptrs hessian, const tensor2_sym_ptrs metric_tensor,
	float* deriv_trace, const octree_abs_pos* data, const uint substep_index, const uint max_depth) {

	uint3 idx = threadIdx + blockDim * blockIdx;
	const uint node_idx = idx.z / size_domain;
	idx.z -= node_idx * size_domain;

	const int depth = data[node_idx].depth(); if (depth == -1) { return; }
	if (!active_depth(substep_index, depth, max_depth)) { return; }
	int read_write_idx = ((node_idx * total_size_domain + idx.z) * total_size_domain + idx.y) * total_size_domain
		+ idx.x + padding_domain * (1u + total_size_domain + total_size_domain * total_size_domain);

	const float3x3_sym inv_metric = metric_tensor.tens(read_write_idx).inverse();
	read_write_idx = ((node_idx * size_domain + idx.z) * size_domain + idx.y) * size_domain + idx.x;
	deriv_trace[read_write_idx] = trace_contra(hessian.tens(read_write_idx), inv_metric);
}

template <class AMR_data>
void scalar_laplacian_noncovariant(const smart_gpu_buffer<float>& field, const tensor2_sym_field& metric_tensor, smart_gpu_buffer<compressed_float3>& gradient_out,
	comp_tensor2_sym_field& hessian_intermediate, smart_gpu_buffer<float>& laplacian, AMR<AMR_data>& amr, cudaStream_t& stream, fast_prng& rng)
{
	differentiate(field, gradient_out, hessian_intermediate.diag, amr, stream, rng);
	mixed_partials(field, hessian_intermediate.off_diag, amr, stream, rng);

	dim3 threads(threadsA3_v, threadsB3_v, threadsD3_v);
	dim3 blocks(size_domain / threads.x, size_domain / threads.y,
		amr.curr_used_slots() * size_domain / threads.z);
	__deriv_trace<<<blocks, threads, 0, stream>>>(hessian_intermediate.ptrs(), metric_tensor.ptrs(),
		laplacian.gpu_buffer_ptr, amr.positions.gpu_buffer_ptr, amr.timer_helper, amr.read_max_depth());
}

template <class AMR_data>
void metric_derivatives_noncovariant(const tensor2_sym_field& metric_tensor, comp_tensor3_sym_field& metric_derivative,
	comp_tensor2_sym_field& hessian_int1, comp_tensor2_sym_field& hessian_int2, tensor2_sym_field& metric_laplacian,
	AMR<AMR_data>& amr, round_robin_threads& streams, fast_prng& rng) {

	cudaStream_t& stream_1 = streams.yield_stream();
	cudaStream_t& stream_2 = streams.yield_stream();

	scalar_laplacian_noncovariant<AMR_data>(metric_tensor.xx, metric_tensor, metric_derivative.xx, hessian_int1, metric_laplacian.xx, amr, stream_1, rng);
	scalar_laplacian_noncovariant<AMR_data>(metric_tensor.xy, metric_tensor, metric_derivative.xy, hessian_int2, metric_laplacian.xy, amr, stream_2, rng);
	scalar_laplacian_noncovariant<AMR_data>(metric_tensor.xz, metric_tensor, metric_derivative.xz, hessian_int1, metric_laplacian.xz, amr, stream_1, rng);
	scalar_laplacian_noncovariant<AMR_data>(metric_tensor.yy, metric_tensor, metric_derivative.yy, hessian_int2, metric_laplacian.yy, amr, stream_2, rng);
	scalar_laplacian_noncovariant<AMR_data>(metric_tensor.yz, metric_tensor, metric_derivative.yz, hessian_int1, metric_laplacian.yz, amr, stream_1, rng);
	scalar_laplacian_noncovariant<AMR_data>(metric_tensor.zz, metric_tensor, metric_derivative.zz, hessian_int2, metric_laplacian.zz, amr, stream_2, rng);
	cuda_sync();
}

__global__ void __compute_raised_christoffel_symbols(const tensor2_sym_ptrs metric, const comp_tensor3_sym_ptrs metric_derivs,
	const int global_seed, comp_tensor3_sym_ptrs christoffel, const octree_abs_pos* data, const uint substep_index, const uint max_depth) {

	uint3 idx = threadIdx + blockDim * blockIdx;
	const uint node_idx = idx.z / size_domain;

	const int depth = data[node_idx].depth(); if (depth == -1) { return; }
	if (!active_depth(substep_index, depth, max_depth)) { return; }

	fast_prng rng = fast_prng(global_seed + idx.x * 2178614);
	rng.seed ^= rng.generate_int() * idx.y;
	rng.seed ^= rng.generate_int() * idx.z;
	idx.z -= node_idx * size_domain;

	int read_write_idx = ((node_idx * size_domain + idx.z) * size_domain + idx.y) * size_domain + idx.x;
	compressed_float3 xx, xy, xz, yy, yz, zz;

	xx = metric_derivs.xx[read_write_idx]; // cYxx,i
	xy = metric_derivs.xy[read_write_idx]; // cYxy,i
	xz = metric_derivs.xz[read_write_idx]; // ...
	yy = metric_derivs.yy[read_write_idx];
	yz = metric_derivs.yz[read_write_idx];
	zz = metric_derivs.zz[read_write_idx];

	float3x3_sym x = float3x3_sym((float3)xx, (float3)xy, (float3)xz); // automatic symmetrize, cYx(j,i)
	float3x3_sym y = float3x3_sym((float3)xy, (float3)yy, (float3)yz); // automatic symmetrize, cYy(j,i)
	float3x3_sym z = float3x3_sym((float3)xz, (float3)yz, (float3)zz); // automatic symmetrize, cYz(j,i)
	float3x3_sym inv_met = metric.tens(((node_idx * total_size_domain + idx.z) * total_size_domain + idx.y) * total_size_domain
		+ idx.x + padding_domain * (1u + total_size_domain + total_size_domain * total_size_domain));

	christoffel.xx[read_write_idx] = compressed_float3(inv_met * (make_float3(x.xx, y.xx, z.xx) - ((float3)xx) * .5f), rng);
	christoffel.xy[read_write_idx] = compressed_float3(inv_met * (make_float3(x.xy, y.xy, z.xy) - ((float3)xy) * .5f), rng);
	christoffel.xz[read_write_idx] = compressed_float3(inv_met * (make_float3(x.xz, y.xz, z.xz) - ((float3)xz) * .5f), rng);
	christoffel.yy[read_write_idx] = compressed_float3(inv_met * (make_float3(x.yy, y.yy, z.yy) - ((float3)yy) * .5f), rng);
	christoffel.yz[read_write_idx] = compressed_float3(inv_met * (make_float3(x.yz, y.yz, z.yz) - ((float3)yz) * .5f), rng);
	christoffel.zz[read_write_idx] = compressed_float3(inv_met * (make_float3(x.zz, y.zz, z.zz) - ((float3)zz) * .5f), rng);
}

template <class AMR_data>
void compute_raised_christoffel_symbols(const tensor2_sym_field& metric, const comp_tensor3_sym_field& metric_derivs,
	comp_tensor3_sym_field& christoffel, AMR<AMR_data>& amr, cudaStream_t& stream, fast_prng& rng)
{
	dim3 threads(threadsA3_v, threadsB3_v, threadsD3_v);
	dim3 blocks(size_domain / threads.x, size_domain / threads.y,
		amr.curr_used_slots() * size_domain / threads.z);
	__compute_raised_christoffel_symbols<<<blocks, threads, 0, stream>>>(metric.ptrs(), metric_derivs.ptrs(), rng.generate_int(),
		christoffel.ptrs(), amr.positions.gpu_buffer_ptr, amr.timer_helper, amr.read_max_depth());
}

struct BSSN_AMR_data
{
	fast_prng rng;
	AMR<BSSN_AMR_data>& parent;
	round_robin_threads streams;

	// fields
	tensor2_sym_field cyij_old;
	tensor2_sym_field cyij_new;

	// derivatives
	comp_tensor3_sym_field cyij_k;
	comp_tensor3_sym_field cGijk;
	tensor2_sym_field cRij;
	comp_tensor2_sym_field didjA; // used as intermediate to start
	comp_tensor2_sym_field didjW; // used as intermediate to start

	BSSN_AMR_data(const uint node_slots, AMR<BSSN_AMR_data>& parent) : parent(parent), streams(),
		cyij_old(node_slots), cyij_new(node_slots), cyij_k(node_slots, inner_cells_domain), rng(),
		cRij(node_slots, inner_cells_domain), didjA(node_slots, inner_cells_domain),
		didjW(node_slots, inner_cells_domain), cGijk(node_slots, inner_cells_domain) {}

	void copy_back()
	{
		cuda_sync();
		copy_new_to_old(cyij_old, cyij_new, parent, streams);
	}
	void node_ctor(const int node_idx) {
		streams.set_stream_idx(0u); // required for correct synchronization
		copy_to_child(cyij_old, parent.positions.cpu_buffer_ptr[node_idx].final_offset(),
			node_idx, parent.parent_idx_b.cpu_buffer_ptr[node_idx], streams);
		copy_to_child(cyij_new, parent.positions.cpu_buffer_ptr[node_idx].final_offset(),
			node_idx, parent.parent_idx_b.cpu_buffer_ptr[node_idx], streams);
	}
	void node_dtor(const int node_idx) {}
	void modify_nodes()
	{

	}
	void predictor()
	{
		cuda_sync();
		metric_derivatives_noncovariant(cyij_old, cyij_k, didjA, didjW, cRij, parent, streams, rng);
		compute_raised_christoffel_symbols(cyij_old, cyij_k, cGijk, parent, streams.yield_stream(), rng);
	}
	void copy_bounds()
	{
		cuda_sync();
		copy_boundaries<BSSN_AMR_data>(cyij_old, cyij_new, parent, streams);
	}
	void corrector()
	{
		cuda_sync();
		metric_derivatives_noncovariant(cyij_new, cyij_k, didjA, didjW, cRij, parent, streams, rng);
		compute_raised_christoffel_symbols(cyij_new, cyij_k, cGijk, parent, streams.yield_stream(), rng);
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


#endif