#ifndef AMR_H
#define AMR_H

#include "image_process.h"
#include <stdio.h>
#include "tensors.h"

__device__ constexpr uint depth_limit = 8u; // hard limit from boundary struct handling
__device__ constexpr float outer_size = 100.f;
__device__ constexpr uint size_domain = 28u;
__device__ constexpr uint padding_domain = 2u;
__device__ constexpr uint total_size_domain = size_domain + 2u * padding_domain;
__device__ constexpr uint cells_domain = total_size_domain * total_size_domain * total_size_domain;
__device__ constexpr float outer_dx = outer_size / size_domain;
__device__ constexpr float outer_dt = outer_dx * .4f; // CFL 0.4

__device__ constexpr float smallest_dx = outer_dx / (1u << depth_limit);
__device__ constexpr float smallest_dt = outer_dt / (1u << depth_limit);

// LSB, ddddd|zyxzyx...zyxzyxzyx
struct ALIGN_BYTES(4) octree_abs_pos
{
	uint data;
	__inline__ constexpr __device__ __host__ octree_abs_pos() : data(~0u) { }
	__inline__ constexpr __device__ __host__ int depth() const {
		return (~data) ? (data >> 27u) : (-1);
	}
	__inline__ constexpr __device__ __host__ int final_offset() const {
		return (data >> ((depth() - 1u) * 3u)) & 7u;
	}
	__inline__ __device__ __host__ float3 absolute_central_position() const {
		float3 result = float3();
		for (uint i = 0, s = (data & 134217727u),
			d = depth(); i < d; i++, s >>= 3)
		{
			result.x += ((s & 1u) ? .25f : -.25f) / (1 << i);
			result.y += ((s & 2u) ? .25f : -.25f) / (1 << i);
			result.z += ((s & 4u) ? .25f : -.25f) / (1 << i);
		}
		return result * outer_size;
	}

	__inline__ __device__ __host__ void set_depth(int depth)
	{
		if (depth == -1) { data = ~0u; return; }
		data = (data & 134217727u) | (depth << 27u);
	}
	__inline__ __device__ __host__ void set_child(const octree_abs_pos& parent, const char offset)
	{
		data = parent.data + (1u << 27u);
		data |= ((uint)offset) << (parent.depth() * 3u);
	}
};
struct ALIGN_BYTES(8) domain_boundary
{
private:
	uint node_data;
	compressed_float3 rel_pos;

public:
	__inline__ __device__ __host__ domain_boundary() : node_data(~0u), rel_pos() { }
	__inline__ __device__ __host__ domain_boundary(const octree_abs_pos& this_d, 
		const octree_abs_pos& other_d, const int other_idx)
		: node_data((uint(other_idx) & 16777215u) 
			| (((this_d.depth() - other_d.depth())&15u) << 24u)
			| ((this_d.depth() & 15u) << 28u)) {

		float3 result = float3(); float mult = .25f * (1u << other_d.depth());
		for (uint i = 0, s = (this_d.data & 134217727u),
			d = this_d.depth(); i < d; i++, s >>= 3)
		{
			result.x += ((s & 1u) ? mult : -mult) / (1 << i);
			result.y += ((s & 2u) ? mult : -mult) / (1 << i);
			result.z += ((s & 4u) ? mult : -mult) / (1 << i);
		}
		for (uint i = 0, s = (other_d.data & 134217727u),
			d = other_d.depth(); i < d; i++, s >>= 3)
		{
			result.x -= ((s & 1u) ? mult : -mult) / (1 << i);
			result.y -= ((s & 2u) ? mult : -mult) / (1 << i);
			result.z -= ((s & 4u) ? mult : -mult) / (1 << i);
		}
		rel_pos = compressed_float3(result);
	}
	__inline__ __device__ __host__ int target_idx() const {
		return (node_data & 16777215u) | ((node_data & (1u << 23u)) ? 4278190080u : 0u);
	}
	__inline__ __device__ __host__ float3 rel_pos_t() const { return (float3)rel_pos; }
	__inline__ __device__ __host__ float rel_scale() const {
		return 1.f / (1u << ((node_data >> 24u) & 15u));
	}
	__inline__ __device__ __host__ uint depth_diff() const {
		return (node_data >> 24u) & 15u;
	}
	__inline__ __device__ __host__ int curr_depth() const {
		return ((node_data >> 28u) == 15u) ? (-1) : (node_data >> 28u);
	}
};

constexpr uint yield_sum_coordinates(uint dat_1, uint dat_2, bool overflow = true)
{
	uint z = ((dat_1 | 115043766u) + (dat_2 & 19173961u)) & 153391689u;
	uint y = ((dat_1 | (115043766u << 1u)) + (dat_2 & (19173961u << 1u))) & (153391689u << 1u);
	uint x = ((dat_1 | (115043766u << 2u)) + (dat_2 & (19173961u << 2u))) & (153391689u << 2u); x |= y | z;
	return ((x & (7u << 27u)) && overflow) ? (~0u) : (x & ((1u << 27u) - 1u)); // if overflow, return ~0u
}
constexpr uint yield_diff_coordinates(uint dat_1, uint dat_2, bool overflow = true) 
{
	uint z = ((dat_1 & 19173961u) - (dat_2 & 19173961u)) & 153391689u;
	uint y = ((dat_1 & (19173961u << 1u)) - (dat_2 & (19173961u << 1u))) & (153391689u << 1u);
	uint x = ((dat_1 & (19173961u << 2u)) - (dat_2 & (19173961u << 2u))) & (153391689u << 2u); x |= y | z;
	return ((x & (7u << 27u)) && overflow) ? (~0u) : (x & ((1u << 27u) - 1u)); // if overflow, return ~0u
}
constexpr uint directions[3] = { 4u, 2u, 1u }; // x y z when reversed

template <class AMR_data>
struct AMR
{
	AMR_data sim_data;
	smart_gpu_cpu_buffer<int> parent_idx_b;
	smart_gpu_cpu_buffer<int> children_idx_b;
	smart_gpu_cpu_buffer<octree_abs_pos> positions;
	smart_gpu_cpu_buffer<domain_boundary> boundaries;

private:
	bool dirty;
	int max_depth;
	uint first_free_slot;
	uint final_used_slot;

	void remove_node_recursive(const int node_idx)
	{
		if (node_idx == -1 || positions.cpu_buffer_ptr[node_idx].depth() == -1) { return; }

		sim_data.node_dtor(node_idx);
		for (uint i = 0; i < 8u; i++)
			remove_node_recursive(children_idx_b.cpu_buffer_ptr[node_idx * 8u + i]);

		children_idx_b.cpu_buffer_ptr[parent_idx_b.cpu_buffer_ptr[node_idx]
			* 8u + positions.cpu_buffer_ptr[node_idx].final_offset()] = -1;
		parent_idx_b.cpu_buffer_ptr[node_idx] = positions.cpu_buffer_ptr[node_idx].data = -1;
		first_free_slot = min_uint(first_free_slot, node_idx);
	}
	std::string print_node_and_children(const uint node_idx) const
	{
		int depth = positions.cpu_buffer_ptr[node_idx].depth();
		std::string pad = "";
		for (uint i = 0; i < depth; i++)
			pad += "  ";

		std::string result = pad + "Node index: " + std::to_string(node_idx) + "\n";
		result += pad + "Parent index: " + std::to_string(parent_idx_b.cpu_buffer_ptr[node_idx]) + "\n";
		int offset = positions.cpu_buffer_ptr[node_idx].final_offset();
		if (depth > 0)
			result += pad + "Offset: " + ((offset & 1u) ? "1" : "-1") + ", "
			+ ((offset & 2u) ? "1" : "-1") + ", "
			+ ((offset & 4u) ? "1" : "-1") + "\n";
		result += pad + "Depth: " + std::to_string(depth) + "\n";
		result += pad + "Position: " + to_string(positions.cpu_buffer_ptr[node_idx].absolute_central_position()) + "\n";
		result += pad + "Children: ";

		uint children_count = 0u;
		for (uint i = 0; i < 8; i++)
			if (children_idx_b.cpu_buffer_ptr[node_idx * 8u + i] != -1)
			{
				result += "\n\n" + print_node_and_children(children_idx_b.cpu_buffer_ptr[node_idx * 8u + i]);
				children_count++;
			}
		if (children_count == 0u) { result += "None"; }
		return result;
	}

public:
	const uint max_slots;
	uint timer_helper;

	void increment_timer() {
		(++timer_helper) &= ((1u << max_depth) - 1u);
	}
	AMR(const uint node_slots) : parent_idx_b(node_slots), children_idx_b(node_slots * 8u),
		max_depth(0), positions(node_slots), first_free_slot(1), max_slots(node_slots), 
		sim_data(node_slots, *this), boundaries(node_slots * 6u), final_used_slot(0u), dirty(true) {
		memset(parent_idx_b.cpu_buffer_ptr, -1, parent_idx_b.total_size());
		memset(children_idx_b.cpu_buffer_ptr, -1, children_idx_b.total_size());
		memset(positions.cpu_buffer_ptr, -1, positions.total_size());

		positions.cpu_buffer_ptr[0].data = 0u; // Root
		for (uint i = 0; i < 6; i++)
			boundaries.cpu_buffer_ptr[i] = domain_boundary();
	}

	uint read_first_free_slot() const {
		return first_free_slot;
	}
	uint final_slot_used() const {
		return final_used_slot;
	}
	void add_node(const int parent_idx, const int offset)
	{
		if ((first_free_slot >= max_slots) || (children_idx_b.cpu_buffer_ptr[parent_idx * 8u + offset] != -1)) { return; }
		if (positions.cpu_buffer_ptr[parent_idx].depth() >= depth_limit) { return; }
		
		parent_idx_b.cpu_buffer_ptr[first_free_slot] = parent_idx;
		children_idx_b.cpu_buffer_ptr[parent_idx * 8u + offset] = first_free_slot;
		positions.cpu_buffer_ptr[first_free_slot].set_child(positions.cpu_buffer_ptr[parent_idx], offset);
		int new_depth = positions.cpu_buffer_ptr[first_free_slot].depth();
		int change_max_depth = max(max_depth, new_depth) - max_depth;

		timer_helper <<= change_max_depth;
		max_depth += change_max_depth; sim_data.node_ctor(first_free_slot);

		final_used_slot = max_uint(final_used_slot, first_free_slot);
		for (; (positions.cpu_buffer_ptr[first_free_slot].depth() != -1) 
			&& (first_free_slot < max_slots); first_free_slot++) {}

		for (uint i = 0; i <= final_used_slot; i++)
			if (positions.cpu_buffer_ptr[i].depth() >= new_depth)
				for (uint j = 0; j < 6; j++)
					boundaries.cpu_buffer_ptr[i * 6u + j] = yield_boundary(i, j);
		dirty = true;
	}
	void remove_node(const int node_idx)
	{
		int curr_depth = positions.cpu_buffer_ptr[node_idx].depth();
		if (node_idx == -1 || curr_depth == -1) { return; }
		remove_node_recursive(node_idx);

		int old_max_depth = max_depth; max_depth = 0;
		for (uint i = 0; i < max_slots; i++)
			if (i != node_idx)
				max_depth = max(max_depth, positions.cpu_buffer_ptr[i].depth());
		
		int change_max_depth = old_max_depth - max_depth;
		timer_helper >>= change_max_depth;

		for (; (positions.cpu_buffer_ptr[final_used_slot].depth() == -1) && (final_used_slot > 0u); final_used_slot--) {}

		for (uint i = 0; i <= final_used_slot; i++)
			if (positions.cpu_buffer_ptr[i].depth() >= curr_depth)
				for (uint j = 0; j < 6; j++)
					boundaries.cpu_buffer_ptr[i * 6u + j] = yield_boundary(i, j);
		dirty = true;
	}
	uint find_node_idx(uint packed_pos, const uint max_depth = ~0u) const {
		uint curr_idx = 0u, depth = 0u;
		while (true)
		{
			int childidx = children_idx_b.cpu_buffer_ptr[curr_idx * 8u + (packed_pos & 7u)];
			if (childidx == -1 || depth >= max_depth) { return curr_idx; }
			packed_pos >>= 3u; curr_idx = childidx; depth++;
		}
	}
	inline constexpr int read_max_depth() const {
		return max_depth;
	}

	void copy_to_gpu() {
		if (!dirty) { return; }
		parent_idx_b.copy_to_gpu();
		children_idx_b.copy_to_gpu();
		positions.copy_to_gpu();
		boundaries.copy_to_gpu();
		dirty = false;
	}
	domain_boundary yield_boundary(int node_idx, uint face) const
	{
		int depth = positions.cpu_buffer_ptr[node_idx].depth();
		if (depth <= 0) { return domain_boundary(); }
		uint old_pos = reverse_bits(positions.cpu_buffer_ptr[node_idx].data) >> 5u;
		
		if (face < 3)
			old_pos = yield_sum_coordinates(old_pos, (directions[face] << ((9u - depth) * 3u)) & ((1u << 27u) - 1u));
		else
			old_pos = yield_diff_coordinates(old_pos, (directions[face - 3] << ((9u - depth) * 3u)) & ((1u << 27u) - 1u));
		
		int new_idx = find_node_idx(reverse_bits(old_pos) >> 5u, depth);
		if (old_pos == (~0u)) { new_idx = 0; } // use the outer boundary as the domain instead.
		return domain_boundary(positions.cpu_buffer_ptr[node_idx], positions.cpu_buffer_ptr[new_idx], new_idx);
	}
	std::string to_string_debug() const {
		return print_node_and_children(0);
	}
};

struct blank_AMR_data
{
	AMR<blank_AMR_data>& parent;

	blank_AMR_data(const uint node_slots,
		AMR<blank_AMR_data>& parent) : parent(parent) {}
	void node_ctor(const int node_idx) {}
	void node_dtor(const int node_idx) {}
};

struct vector_ptrs
{
	float *x, *y, *z;
	__host__ vector_ptrs (float* x, float* y, float* z) : x(x), y(y), z(z) {}

	__device__ float3 vec(uint idx) const {
		return make_float3(x[idx], y[idx], z[idx]);
	}
	__device__ void set(uint idx, const float3& dat) {
		x[idx] = dat.x;
		y[idx] = dat.y;
		z[idx] = dat.z;
	}
};
struct vector_field
{
	smart_gpu_buffer<float> x, y, z;
	vector_field(const uint node_slots) : x(node_slots* cells_domain), y(node_slots* cells_domain), z(node_slots* cells_domain) {}
	__host__ vector_ptrs ptrs() const {
		return vector_ptrs(x.gpu_buffer_ptr, y.gpu_buffer_ptr, z.gpu_buffer_ptr);
	}
};
struct tensor2_ptrs
{
	float* xx, * xy, * xz;
	float* yx, * yy, * yz;
	float* zx, * zy, * zz;
	__host__ tensor2_ptrs(float* xx, float* xy, float* xz,
		float* yx, float* yy, float* yz, 
		float* zx, float* zy, float* zz) 
		: xx(xx), xy(xy), xz(xz),
		yx(yx), yy(yy), yz(yz),
		zx(zx), zy(zy), zz(zz) {}

	__device__ float3x3 tens(uint idx) const {
		return float3x3(xx[idx], xy[idx], xz[idx], yx[idx], yy[idx], yz[idx], zx[idx], zy[idx], zz[idx]);
	}
	__device__ void set(uint idx, const float3x3& dat) {
		xx[idx] = dat.xx;
		xy[idx] = dat.xy;
		xz[idx] = dat.xz;
		yx[idx] = dat.yx;
		yy[idx] = dat.yy;
		yz[idx] = dat.yz;
		zx[idx] = dat.zx;
		zy[idx] = dat.zy;
		zz[idx] = dat.zz;
	}
};
struct tensor2_field
{
	smart_gpu_buffer<float> xx, xy, xz,yx,yy,yz,zx,zy,zz;
	tensor2_field(const uint node_slots) : xx(node_slots* cells_domain), 
		xy(node_slots* cells_domain),
		xz(node_slots* cells_domain), 
		yx(node_slots* cells_domain),
		yy(node_slots* cells_domain),
		yz(node_slots* cells_domain),
		zx(node_slots* cells_domain),
		zy(node_slots* cells_domain),
		zz(node_slots* cells_domain) {}
	__host__ tensor2_ptrs ptrs() const {
		return tensor2_ptrs(xx.gpu_buffer_ptr, xy.gpu_buffer_ptr, xz.gpu_buffer_ptr,
			yx.gpu_buffer_ptr, yy.gpu_buffer_ptr, yz.gpu_buffer_ptr,
			zx.gpu_buffer_ptr, zy.gpu_buffer_ptr, zz.gpu_buffer_ptr);
	}
};
struct tensor2_sym_ptrs
{
	float* xx, * xy, * xz;
	float	   * yy, * yz;
	float            * zz;
	__host__ tensor2_sym_ptrs(float* xx, float* xy, float* xz,
								float* yy, float* yz,
										   float* zz)
		: xx(xx), xy(xy), xz(xz),
		          yy(yy), yz(yz),
						  zz(zz) {}

	__device__ float3x3_sym tens(uint idx) const {
		return float3x3_sym(xx[idx], xy[idx], xz[idx], yy[idx], yz[idx], zz[idx]);
	}
	__device__ void set(uint idx, const float3x3_sym& dat) {
		xx[idx] = dat.xx;
		xy[idx] = dat.xy;
		xz[idx] = dat.xz;
		yy[idx] = dat.yy;
		yz[idx] = dat.yz;
		zz[idx] = dat.zz;
	}
};
struct tensor2_sym_field
{
	smart_gpu_buffer<float> xx, xy, xz, yy, yz, zz;
	tensor2_sym_field(const uint node_slots) : xx(node_slots* cells_domain),
		xy(node_slots* cells_domain),
		xz(node_slots* cells_domain),
		yy(node_slots* cells_domain),
		yz(node_slots* cells_domain),
		zz(node_slots* cells_domain) {}
	__host__ tensor2_sym_ptrs ptrs() const {
		return tensor2_sym_ptrs(xx.gpu_buffer_ptr, xy.gpu_buffer_ptr, xz.gpu_buffer_ptr,
											   yy.gpu_buffer_ptr, yz.gpu_buffer_ptr,
																  zz.gpu_buffer_ptr);
	}
};
struct comp_tensor2_ptrs
{
	compressed_float3* x, * y, * z;
	comp_tensor2_ptrs(compressed_float3* x, compressed_float3* y, compressed_float3* z) : x(x), y(y), z(z) {}

	__device__ float3x3 tens(uint idx) const {
		return decompress(x[idx], y[idx], z[idx]);
	}
	__device__ void set(uint idx, const float3x3& dat) {
		compress(dat, x + idx, y + idx, z + idx);
	}
	__device__ void set(uint idx, const float3x3& dat, fast_prng& rng) {
		x[idx] = compressed_float3(make_float3(dat.xx, dat.xy, dat.xz), rng);
		y[idx] = compressed_float3(make_float3(dat.yx, dat.yy, dat.yz), rng);
		z[idx] = compressed_float3(make_float3(dat.zx, dat.zy, dat.zz), rng);
	}
};
struct comp_tensor2_field
{
	smart_gpu_buffer<compressed_float3> x, y, z;
	comp_tensor2_field(const uint node_slots) : x(node_slots* cells_domain), y(node_slots* cells_domain), z(node_slots* cells_domain) {}
	comp_tensor2_ptrs ptrs() const {
		return comp_tensor2_ptrs(x.gpu_buffer_ptr, y.gpu_buffer_ptr, z.gpu_buffer_ptr);
	}
};
struct comp_tensor2_sym_ptrs
{
	compressed_float3* diag, * off_diag;
	comp_tensor2_sym_ptrs(compressed_float3* diag, compressed_float3* off_diag) : diag(diag), off_diag(off_diag) {}

	__device__ float3x3_sym tens(uint idx) const {
		return decompress(diag[idx], off_diag[idx]);
	}
	__device__ void set(uint idx, const float3x3_sym& dat) {
		compress(dat, diag + idx, off_diag + idx);
	}
	__device__ void set(uint idx, const float3x3_sym& dat, fast_prng& rng) {
		diag[idx] = compressed_float3(dat.diag(), rng);
		off_diag[idx] = compressed_float3(make_float3(dat.xy, dat.xz, dat.yz), rng);
	}
};
struct comp_tensor2_sym_field
{
	smart_gpu_buffer<compressed_float3> diag, off_diag;
	comp_tensor2_sym_field(const uint node_slots) : diag(node_slots* cells_domain), off_diag(node_slots* cells_domain) {}
	comp_tensor2_sym_ptrs ptrs() const {
		return comp_tensor2_sym_ptrs(diag.gpu_buffer_ptr, off_diag.gpu_buffer_ptr);
	}
};
struct comp_tensor3_ptrs
{
	compressed_float3 * xx, * xy, * xz;
	compressed_float3 * yx, * yy, * yz;
	compressed_float3 * zx, * zy, * zz;
	comp_tensor3_ptrs(compressed_float3* xx, compressed_float3* xy, compressed_float3* xz,
		compressed_float3* yx, compressed_float3* yy, compressed_float3* yz,
		compressed_float3* zx, compressed_float3* zy, compressed_float3* zz)
		: xx(xx), xy(xy), xz(xz),
		yx(yx), yy(yy), yz(yz),
		zx(zx), zy(zy), zz(zz) {}

	__device__ void tens(uint idx, float3x3& x, float3x3& y, float3x3&z) const {
		x = float3x3((float3)xx[idx], (float3)xy[idx], (float3)xz[idx]);
		y = float3x3((float3)yx[idx], (float3)yy[idx], (float3)yz[idx]);
		z = float3x3((float3)zx[idx], (float3)zy[idx], (float3)zz[idx]);
	}
	__device__ void set(uint idx, const float3x3& x, const float3x3& y, const float3x3& z)  {
		xx[idx] = compressed_float3(make_float3(x.xx, x.xy, x.xz));
		xy[idx] = compressed_float3(make_float3(x.yx, x.yy, x.yz));
		xz[idx] = compressed_float3(make_float3(x.zx, x.zy, x.zz));
		yx[idx] = compressed_float3(make_float3(y.xx, y.xy, y.xz));
		yy[idx] = compressed_float3(make_float3(y.yx, y.yy, y.yz));
		yz[idx] = compressed_float3(make_float3(y.zx, y.zy, y.zz));
		zx[idx] = compressed_float3(make_float3(z.xx, z.xy, z.xz));
		zy[idx] = compressed_float3(make_float3(z.yx, z.yy, z.yz));
		zz[idx] = compressed_float3(make_float3(z.zx, z.zy, z.zz));
	}
	__device__ void set(uint idx, const float3x3& x, const float3x3& y, const float3x3& z, fast_prng& rng)  {
		xx[idx] = compressed_float3(make_float3(x.xx, x.xy, x.xz), rng);
		xy[idx] = compressed_float3(make_float3(x.yx, x.yy, x.yz), rng);
		xz[idx] = compressed_float3(make_float3(x.zx, x.zy, x.zz), rng);
		yx[idx] = compressed_float3(make_float3(y.xx, y.xy, y.xz), rng);
		yy[idx] = compressed_float3(make_float3(y.yx, y.yy, y.yz), rng);
		yz[idx] = compressed_float3(make_float3(y.zx, y.zy, y.zz), rng);
		zx[idx] = compressed_float3(make_float3(z.xx, z.xy, z.xz), rng);
		zy[idx] = compressed_float3(make_float3(z.yx, z.yy, z.yz), rng);
		zz[idx] = compressed_float3(make_float3(z.zx, z.zy, z.zz), rng);
	}
};
struct comp_tensor3_field
{
	smart_gpu_buffer<compressed_float3> xx, xy, xz, yx, yy, yz, zx, zy, zz;
	comp_tensor3_field(const uint node_slots) : xx(node_slots* cells_domain),
		xy(node_slots* cells_domain),
		xz(node_slots* cells_domain),
		yx(node_slots* cells_domain),
		yy(node_slots* cells_domain),
		yz(node_slots* cells_domain),
		zx(node_slots* cells_domain),
		zy(node_slots* cells_domain),
		zz(node_slots* cells_domain) {}
	comp_tensor3_ptrs ptrs() const {
		return comp_tensor3_ptrs(xx.gpu_buffer_ptr, xy.gpu_buffer_ptr, xz.gpu_buffer_ptr,
			yx.gpu_buffer_ptr, yy.gpu_buffer_ptr, yz.gpu_buffer_ptr,
			zx.gpu_buffer_ptr, zy.gpu_buffer_ptr, zz.gpu_buffer_ptr);
	}
};
struct comp_tensor3_sym_ptrs
{
	compressed_float3* xx, * xy, * xz;
	compressed_float3* yy, * yz;
	compressed_float3* zz;
	comp_tensor3_sym_ptrs(compressed_float3* xx, compressed_float3* xy, compressed_float3* xz,
		compressed_float3* yy, compressed_float3* yz,
		compressed_float3* zz)
		: xx(xx), xy(xy), xz(xz),
		yy(yy), yz(yz),
		zz(zz) {}

	__device__ void tens(uint idx, float3x3_sym& x, float3x3_sym& y, float3x3_sym& z) const {
		float3 x_i = (float3)xx[idx];
		float3 y_i = (float3)xy[idx];
		float3 z_i = (float3)xz[idx];

		x.xx = x_i.x;
		y.xx = x_i.y;
		z.xx = x_i.z;

		x.xy = y_i.x;
		y.xy = y_i.y;
		z.xy = y_i.z;

		x.xz = z_i.x;
		y.xz = z_i.y;
		z.xz = z_i.z;

		y_i = (float3)yy[idx];
		z_i = (float3)yz[idx];

		x.yy = y_i.x;
		y.yy = y_i.y;
		z.yy = y_i.z;

		x.yz = z_i.x;
		y.yz = z_i.y;
		z.yz = z_i.z;

		z_i = (float3)zz[idx];

		x.zz = z_i.x;
		y.zz = z_i.y;
		z.zz = z_i.z;
	}
	__device__ void set(uint idx, const float3x3& x, const float3x3& y, const float3x3& z) {
		xx[idx] = compressed_float3(make_float3(x.xx, y.xx, z.xx));
		xy[idx] = compressed_float3(make_float3(x.xy, y.xy, z.xy));
		xz[idx] = compressed_float3(make_float3(x.xz, y.xz, z.xz));
		yy[idx] = compressed_float3(make_float3(x.yy, y.yy, z.yy));
		yz[idx] = compressed_float3(make_float3(x.yz, y.yz, z.yz));
		zz[idx] = compressed_float3(make_float3(x.zz, y.zz, z.zz));
	}
	__device__ void set(uint idx, const float3x3& x, const float3x3& y, const float3x3& z, fast_prng& rng) {
		xx[idx] = compressed_float3(make_float3(x.xx, y.xx, z.xx), rng);
		xy[idx] = compressed_float3(make_float3(x.xy, y.xy, z.xy), rng);
		xz[idx] = compressed_float3(make_float3(x.xz, y.xz, z.xz), rng);
		yy[idx] = compressed_float3(make_float3(x.yy, y.yy, z.yy), rng);
		yz[idx] = compressed_float3(make_float3(x.yz, y.yz, z.yz), rng);
		zz[idx] = compressed_float3(make_float3(x.zz, y.zz, z.zz), rng);
	}
};
struct comp_tensor3_sym_field // For christoffel symbols
{
	smart_gpu_buffer<compressed_float3> xx, xy, xz, yy, yz, zz;
	comp_tensor3_sym_field(const uint node_slots) : xx(node_slots* cells_domain),
		xy(node_slots* cells_domain),
		xz(node_slots* cells_domain),
		yy(node_slots* cells_domain),
		yz(node_slots* cells_domain),
		zz(node_slots* cells_domain) {}
	comp_tensor3_sym_ptrs ptrs() const {
		return comp_tensor3_sym_ptrs(xx.gpu_buffer_ptr, xy.gpu_buffer_ptr, xz.gpu_buffer_ptr,
			yy.gpu_buffer_ptr, yz.gpu_buffer_ptr,
			zz.gpu_buffer_ptr);
	}
};

__inline__ __device__ float2 relative_time(const uint substep_index,
	const domain_boundary data, const uint max_depth) {
	uint rel_depth = data.depth_diff();
	float temp = float((substep_index >> (max_depth - data.curr_depth())) & ((1u << rel_depth) - 1u));
	return make_float2(temp, temp + 1.f) / (1u << rel_depth);
}
__inline__ __device__ bool active_depth(const uint substep_index,
	const uint depth, const uint max_depth)
{
	return (substep_index & ((1u << (max_depth - depth)) - 1u)) == 0u;
}

__device__ constexpr uint threadsD = padding_domain;
__device__ constexpr uint threadsA()
{
	const float tgt_sq = 512.f / threadsD;
	float guess = size_domain * .5f;
	guess = .5f * (guess + tgt_sq / guess);
	guess = .5f * (guess + tgt_sq / guess);
	guess = .5f * (guess + tgt_sq / guess);
	uint result = uint(float(size_domain) / guess);
	return size_domain / (result + (result == 0));
}
__device__ constexpr uint threadsB()
{
	float result = 512.f / (threadsD * threadsA());
	result = float(size_domain) / result;
	uint temp = uint(result); temp += (result - temp) > 0.f;
	return size_domain / (temp + (temp == 0u));
}
__device__ constexpr uint threadsA_v = threadsA();
__device__ constexpr uint threadsB_v = threadsB();

template <class T>
__inline__ __device__ T __interpolate_dat(const T* dat_old, const float3& read_pos, const uint3& read_idx, const uint& read_idx_n)
{
	T read_00 = dat_old[read_idx_n];
	read_00 = read_00 * (1.f - read_pos.x + read_idx.x) + dat_old[read_idx_n + 1u] * (read_pos.x - read_idx.x);

	T read_01 = dat_old[read_idx_n + total_size_domain];
	read_01 = read_01 * (1.f - read_pos.x + read_idx.x) + dat_old[read_idx_n + total_size_domain + 1u] * (read_pos.x - read_idx.x);
	read_00 = read_00 * (1.f - read_pos.y + read_idx.y) + read_01 * (read_pos.y - read_idx.y);

	T read_10 = dat_old[read_idx_n + total_size_domain * total_size_domain];
	read_10 = read_10 * (1.f - read_pos.x + read_idx.x) + dat_old[read_idx_n
		+ total_size_domain * total_size_domain + 1u] * (read_pos.x - read_idx.x);

	T read_11 = dat_old[read_idx_n + total_size_domain * total_size_domain + total_size_domain];
	read_11 = read_11 * (1.f - read_pos.x + read_idx.x) + dat_old[read_idx_n
		+ total_size_domain * total_size_domain + total_size_domain + 1u] * (read_pos.x - read_idx.x);
	read_10 = read_10 * (1.f - read_pos.y + read_idx.y) + read_11 * (read_pos.y - read_idx.y);
	return read_00 * (1.f - read_pos.z + read_idx.z) + read_10 * (read_pos.z - read_idx.z);
}

// Note: all past data must be in dat_old.
template <class T>
__global__ void __copy_boundaries_new(T* __restrict__ dat_old, T* __restrict__ dat_new,
	const domain_boundary* data, const uint substep_index, const uint max_depth, const bool copy_only_new) {
	uint3 idx = threadIdx + blockDim * blockIdx;
	uint node_idx = idx.z / (6u * padding_domain);
	uint dir_idx = idx.z - node_idx * (6u * padding_domain); 
	dir_idx /= padding_domain; idx.z %= padding_domain;

	const domain_boundary bd = data[node_idx * 6u + dir_idx];
	if (bd.target_idx() == -1 || bd.curr_depth() <= 0) { return; }
	if (!active_depth(substep_index, bd.curr_depth(), max_depth)) { return; }

	uint3 write_idx;
	switch (dir_idx)
	{
	case 0:
		write_idx = zxy(idx) + make_uint3(padding_domain + size_domain, padding_domain, padding_domain);
		break;
	case 1:
		write_idx = xzy(idx) + make_uint3(padding_domain, padding_domain + size_domain, padding_domain);
		break;
	case 2:
		write_idx = idx + make_uint3(padding_domain, padding_domain, padding_domain + size_domain);
		break;
	case 3:
		write_idx = zxy(idx) + make_uint3(0u, padding_domain, padding_domain);
		break;
	case 4:
		write_idx = xzy(idx) + make_uint3(padding_domain, 0u, padding_domain);
		break;
	case 5:
		write_idx = idx + make_uint3(padding_domain, padding_domain, 0u);
		break;
	}
	
	float3 read_pos = (make_float3(write_idx) - padding_domain + .5f) * (1.f / size_domain) - .5f; // source coordinates
	read_pos = (read_pos * bd.rel_scale()) + bd.rel_pos_t(); // target domain coordinates
	read_pos = (read_pos + .5f) * size_domain + padding_domain - 0.4999f; // in the target domain index coordinate system

	uint3 read_idx = make_uint3(0u);

	if (bd.target_idx() == 0) // allow boundary
		read_idx = make_uint3(min_uint(fmaxf(read_pos.x, 0.5f), total_size_domain - 2u),
						min_uint(fmaxf(read_pos.y, 0.5f), total_size_domain - 2u),
						min_uint(fmaxf(read_pos.z, 0.5f), total_size_domain - 2u));
	else // disallow boundary
		read_idx = make_uint3(min_uint(fmaxf(read_pos.x, 0.5f + padding_domain), total_size_domain - padding_domain - 2u),
			min_uint(fmaxf(read_pos.y, 0.5f + padding_domain), total_size_domain - padding_domain - 2u),
			min_uint(fmaxf(read_pos.z, 0.5f + padding_domain), total_size_domain - padding_domain - 2u));

	uint read_idx_n = bd.target_idx() * cells_domain + read_idx.x + (read_idx.y + read_idx.z * total_size_domain) * total_size_domain;

	float2 time = relative_time(substep_index, bd, max_depth);
	T old_d = __interpolate_dat<T>(dat_old, read_pos, read_idx, read_idx_n);
	T new_d = __interpolate_dat<T>(dat_new, read_pos, read_idx, read_idx_n);
	uint write_idx_n = node_idx * cells_domain + write_idx.x + (write_idx.y + write_idx.z * total_size_domain) * total_size_domain;

	if (!copy_only_new) { dat_old[write_idx_n] = old_d * (1.f - time.x) + new_d * time.x; }
	dat_new[write_idx_n] = old_d * (1.f - time.y) + new_d * time.y;
}

struct round_robin_threads {
	cudaStream_t streams[8];
	uint current_stream;
	round_robin_threads() : current_stream(0u) {
		for (int i = 0; i < 8; i++)
			cudaStreamCreate(streams + i);
	}
	cudaStream_t& yield_stream() {
		current_stream &= 7u;
		return streams[current_stream++];
	}
	~round_robin_threads()
	{
		for (int i = 0; i < 8; i++)
			cudaStreamDestroy(streams[i]);
	}
};

template <class T, class AMR_data>
void copy_boundaries(T* old_ptr, T* new_ptr, AMR<AMR_data>& amr, cudaStream_t& stream, bool copy_only_new = true)
{
	amr.copy_to_gpu();
	
	dim3 threads(threadsA_v, threadsB_v, threadsD);
	dim3 blocks(size_domain / threads.x, size_domain / threads.y, amr.final_slot_used() * 12u / threads.z);
	__copy_boundaries_new<<<blocks, threads, 0, stream>>>(old_ptr, new_ptr,
		amr.boundaries.gpu_buffer_ptr, amr.timer_helper, amr.read_max_depth(), copy_only_new);
}

template <class T, class AMR_data>
void copy_boundaries(T* old_ptr, T* new_ptr, AMR<AMR_data>& amr, bool copy_only_new = true)
{
	amr.copy_to_gpu();
	
	dim3 threads(threadsA_v, threadsB_v, threadsD);
	dim3 blocks(size_domain / threads.x, size_domain / threads.y, amr.final_slot_used() * 12u / threads.z);
	__copy_boundaries_new<<<blocks, threads>>>(old_ptr, new_ptr,
		amr.boundaries.gpu_buffer_ptr, amr.timer_helper, amr.read_max_depth(), copy_only_new);
}

template <class AMR_data>
void copy_boundaries(vector_field& old_field, vector_field& new_field, AMR<AMR_data>& amr, round_robin_threads& threads)
{
	copy_boundaries(old_field.x.gpu_buffer_ptr, new_field.x.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.y.gpu_buffer_ptr, new_field.y.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.z.gpu_buffer_ptr, new_field.z.gpu_buffer_ptr, amr, threads.yield_stream());
}

template <class AMR_data>
void copy_boundaries(tensor2_field& old_field, tensor2_field& new_field, AMR<AMR_data>& amr, round_robin_threads& threads)
{
	copy_boundaries(old_field.xx.gpu_buffer_ptr, new_field.xx.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.xy.gpu_buffer_ptr, new_field.xy.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.xz.gpu_buffer_ptr, new_field.xz.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.yx.gpu_buffer_ptr, new_field.yx.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.yy.gpu_buffer_ptr, new_field.yy.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.yz.gpu_buffer_ptr, new_field.yz.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.zx.gpu_buffer_ptr, new_field.zx.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.zy.gpu_buffer_ptr, new_field.zy.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.zz.gpu_buffer_ptr, new_field.zz.gpu_buffer_ptr, amr, threads.yield_stream());
}

template <class AMR_data>
void copy_boundaries(tensor2_sym_field& old_field, tensor2_sym_field& new_field, AMR<AMR_data>& amr, round_robin_threads& threads)
{
	copy_boundaries(old_field.xx.gpu_buffer_ptr, new_field.xx.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.xy.gpu_buffer_ptr, new_field.xy.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.xz.gpu_buffer_ptr, new_field.xz.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.yy.gpu_buffer_ptr, new_field.yy.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.yz.gpu_buffer_ptr, new_field.yz.gpu_buffer_ptr, amr, threads.yield_stream());
	copy_boundaries(old_field.zz.gpu_buffer_ptr, new_field.zz.gpu_buffer_ptr, amr, threads.yield_stream());
}

#endif