#ifndef TENSOR_H
#define TENSOR_H

#include "helper_math.h"
#include "CUDA_memory.h"

struct ALIGN_BYTES(4) float3x3
{
    float xx, xy, xz, yx, yy, yz, zx, zy, zz;

    __host__ __device__ constexpr float3x3() : xx(0.f), xy(0.f), xz(0.f), yx(0.f), yy(0.f), yz(0.f), zx(0.f), zy(0.f), zz(0.f) {}
    __host__ __device__ constexpr float3x3(const float3 x, const float3 y, const float3 z) : xx(x.x), xy(x.y), xz(x.z), yx(y.x), yy(y.y), yz(y.z), zx(z.x), zy(z.y), zz(z.z) { }
    __host__ __device__ constexpr float3x3(const float xx, const float xy, const float xz, const float yx, const float yy, const float yz, const float zx, const float zy, const float zz) : xx(xx), xy(xy), xz(xz), yx(yx), yy(yy), yz(yz), zx(zx), zy(zy), zz(zz) { }
    __inline__ __host__ __device__ float3 diag() const
    {
        return make_float3(xx, yy, zz);
    }
    __inline__ __host__ __device__ constexpr float3x3 transpose() const
    {
        return float3x3(xx, yx, zx, xy, yy, zy, xz, yz, zz);
    }
    __inline__ __host__ __device__ constexpr float3x3 operator-() const
    {
        return float3x3(-xx, -xy, -xz, -yx, -yy, -yz, -zx, -zy, -zz);
    }
    __inline__ __host__ __device__ constexpr float3x3 operator-(const float3x3 & a) const
    {
        return float3x3(xx - a.xx, xy - a.xy, xz - a.xz, yx - a.yx, yy - a.yy, yz - a.yz, zx - a.zx, zy - a.zy, zz - a.zz);
    }
    __inline__ __host__ __device__ constexpr float3x3 operator+(const float3x3 & a) const
    {
        return float3x3(xx + a.xx, xy + a.xy, xz + a.xz, yx + a.yx, yy + a.yy, yz + a.yz, zx + a.zx, zy + a.zy, zz + a.zz);
    }
    __inline__ __host__ __device__ constexpr float3x3 operator/(const float a) const
    {
        return float3x3(xx / a, xy / a, xz / a, yx / a, yy / a, yz / a, zx / a, zy / a, zz / a);
    }
    __inline__ __host__ __device__ constexpr float3x3& operator*=(const float a) {
        xx *= a;
        xy *= a;
        xz *= a;
        yx *= a;
        yy *= a;
        yz *= a;
        zx *= a;
        zy *= a;
        zz *= a;
        return *this;
    }
    __inline__ __host__ __device__ constexpr float3x3& operator+=(const float3x3 & a) {
        xx += a.xx;
        xy += a.xy;
        xz += a.xz;
        yx += a.yx;
        yy += a.yy;
        yz += a.yz;
        zx += a.zx;
        zy += a.zy;
        zz += a.zz;
        return *this;
    }
    __inline__ __host__ __device__ constexpr float3x3& operator-=(const float3x3& a) {
        xx -= a.xx;
        xy -= a.xy;
        xz -= a.xz;
        yx -= a.yx;
        yy -= a.yy;
        yz -= a.yz;
        zx -= a.zx;
        zy -= a.zy;
        zz -= a.zz;
        return *this;
    }
    __inline__ __host__ __device__ constexpr float3x3& operator/=(const float a) {
        xx /= a;
        xy /= a;
        xz /= a;
        yx /= a;
        yy /= a;
        yz /= a;
        zx /= a;
        zy /= a;
        zz /= a;
        return *this;
    }
    __inline__ __host__ __device__ constexpr float3x3 operator*(const float a) const
    {
        return float3x3(xx * a, xy * a, xz * a, yx * a, yy * a, yz * a, zx * a, zy * a, zz * a);
    }
    __inline__ __host__ __device__ float3 operator*(const float3 a) const
    {
        return make_float3(a.x * xx + a.y * yx + a.z * zx,
            a.x * xy + a.y * yy + a.z * zy,
            a.x * xz + a.y * yz + a.z * zz);
    }
    __inline__ __host__ __device__ constexpr float3x3 operator*(const float3x3 & a) const
    {
        float3x3 result;
        result.xx = xx * a.xx + yx * a.xy + zx * a.xz;
        result.xy = xy * a.xx + yy * a.xy + zy * a.xz;
        result.xz = xz * a.xx + yz * a.xy + zz * a.xz;
        result.yx = xx * a.yx + yx * a.yy + zx * a.yz;
        result.yy = xy * a.yx + yy * a.yy + zy * a.yz;
        result.yz = xz * a.yx + yz * a.yy + zz * a.yz;
        result.zx = xx * a.zx + yx * a.zy + zx * a.zz;
        result.zy = xy * a.zx + yy * a.zy + zy * a.zz;
        result.zz = xz * a.zx + yz * a.zy + zz * a.zz;
        return result;
    }
    __inline__ __host__ __device__ constexpr float trace() const { return xx + yy + zz; }
    __inline__ __host__ __device__ constexpr float determinant() const {
        return -xz * yy * zx + xy * yz * zx + xz * yx * zy - xx * yz * zy - xy * yx * zz + xx * yy * zz;
    }
    __inline__ __host__ __device__ constexpr float3x3 inverse() const {
        float3x3 result;
        
        result.xx = -yz * zy + yy * zz;
        result.yx = yz * zx - yx * zz;
        result.zx = -yy * zx + yx * zy;

        result.xy = xz * zy - xy * zz;
        result.yy = -xz * zx + xx * zz;
        result.zy = xy * zx - xx * zy;

        result.xz = -xz * yy + xy * yz;
        result.yz = xz * yx - xx * yz;
        result.zz = -xy * yx + xx * yy;

        result /= determinant();
        return result;
    }
};
struct ALIGN_BYTES(8) float3x3_sym
{
    float xx, xy, xz, yy, yz, zz;

    __host__ __device__ constexpr float3x3_sym() : xx(0.f), xy(0.f), xz(0.f), yy(0.f), yz(0.f), zz(0.f) {}
    __host__ __device__ constexpr float3x3_sym(const float3 x, const float3 y, const float3 z) : xx(x.x), xy(x.y), xz(x.z), yy(y.y), yz(y.z), zz(z.z) { }
    __host__ __device__ constexpr float3x3_sym(const float xx, const float xy, const float xz, const float yy, const float yz, const float zz) : xx(xx), xy(xy), xz(xz), yy(yy), yz(yz), zz(zz) { }
    __host__ __device__ constexpr float3x3_sym(const float3x3& to_sym) : xx(to_sym.xx), xy((to_sym.xy + to_sym.yx) * .5f), xz((to_sym.xz + to_sym.zx) * .5f), yy(to_sym.yy), yz((to_sym.yz+ to_sym.zy) * .5f), zz(to_sym.zz) {}
    __inline__ __host__ __device__ float3 diag() const
    {
        return make_float3(xx, yy, zz);
    }
    __inline__ __host__ __device__ constexpr float3x3_sym operator-() const
    {
        return float3x3_sym(-xx, -xy, -xz, -yy, -yz, -zz);
    }
    __inline__ __host__ __device__ constexpr float3x3_sym operator-(const float3x3_sym & a) const
    {
        return float3x3_sym(xx - a.xx, xy - a.xy, xz - a.xz, yy - a.yy, yz - a.yz, zz - a.zz);
    }
    __inline__ __host__ __device__ constexpr float3x3_sym operator+(const float3x3 & a) const
    {
        return float3x3_sym(xx + a.xx, xy + a.xy, xz + a.xz, yy + a.yy, yz + a.yz, zz + a.zz);
    }
    __inline__ __host__ __device__ constexpr float3x3_sym operator/(const float a) const
    {
        return float3x3_sym(xx / a, xy / a, xz / a, yy / a, yz / a, zz / a);
    }
    __inline__ __host__ __device__ constexpr float3x3_sym& operator*=(const float a) {
        xx *= a;
        xy *= a;
        xz *= a;
        yy *= a;
        yz *= a;
        zz *= a;
        return *this;
    }
    __inline__ __host__ __device__ constexpr float3x3_sym& operator+=(const float3x3_sym & a) {
        xx += a.xx;
        xy += a.xy;
        xz += a.xz;
        yy += a.yy;
        yz += a.yz;
        zz += a.zz;
        return *this;
    }
    __inline__ __host__ __device__ constexpr float3x3_sym& operator-=(const float3x3_sym & a) {
        xx -= a.xx;
        xy -= a.xy;
        xz -= a.xz;
        yy -= a.yy;
        yz -= a.yz;
        zz -= a.zz;
        return *this;
    }
    __inline__ __host__ __device__ constexpr float3x3_sym& operator/=(const float a) {
        xx /= a;
        xy /= a;
        xz /= a;
        yy /= a;
        yz /= a;
        zz /= a;
        return *this;
    }
    __inline__ __host__ __device__ constexpr float3x3_sym operator*(const float a) const
    {
        return float3x3_sym(xx * a, xy * a, xz * a, yy * a, yz * a, zz * a);
    }
    __inline__ __host__ __device__ float3 operator*(const float3 a) const
    {
        return make_float3(a.x * xx + a.y * xy + a.z * xz,
            a.x * xy + a.y * yy + a.z * yz,
            a.x * xz + a.y * yz + a.z * zz);
    }
    __inline__ __host__ __device__ constexpr float trace() const { return xx + yy + zz; }
    __inline__ __host__ __device__ constexpr float determinant() const {
        return -xz * yy * xz + xy * yz * xz + xz * xy * yz - xx * yz * yz - xy * xy * zz + xx * yy * zz;
    }
    __inline__ __host__ __device__ constexpr float3x3_sym inverse() const {
        float3x3_sym result;

        result.xx = -yz * yz + yy * zz;
        result.xy = yz * xz - xy * zz;
        result.xz = -yy * xz + xy * yz;

        result.yy = -xz * xz + xx * zz;
        result.yz = xy * xz - xx * yz;

        result.zz = -xy * xy + xx * yy;
        result /= determinant();

        return result;
    }
    __host__ __device__ constexpr operator float3x3() const {
        return float3x3(xx, xy, xz, xy, yy, yz, xz, yz, zz);
    }
};

__inline__ __host__ __device__ float trace_contra(const float3x3& contra, const float3x3_sym& metric) {
    return contra.xx * metric.xx + contra.xy * metric.xy +
        contra.yx * metric.xy + contra.xz * metric.xz +
        contra.zx * metric.xz + contra.yy * metric.yy +
        contra.yz * metric.yz + contra.zy * metric.yz +
        contra.zz * metric.zz;
}
__inline__ __host__ __device__ float trace_contra(const float3x3_sym& contra, const float3x3_sym& metric) {
    return contra.xx * metric.xx + contra.yy * metric.yy + contra.zz * metric.zz 
        + (contra.xy * metric.xy + contra.xz * metric.xz + contra.yz * metric.yz) * 2.f;
}
__inline__ __host__ __device__ void tracefree_contra(float3x3& contra, const float3x3_sym& metric) {
    float trace_factor = trace_contra(contra, metric) / 3.f;
    contra -= metric.inverse() * trace_factor;
}
__inline__ __host__ __device__ void tracefree_covar(float3x3& covar, const float3x3_sym& metric) {
    float trace_factor = trace_contra(covar, metric.inverse()) / 3.f;
    covar -= metric * trace_factor;
}

#define IEEE754_FLOAT_MANTISSA_BITS     23u
#define IEEE754_FLOAT_EXPONENT_BITS     8u
#define IEEE754_FLOAT_EXPONENT_OFFSET   127u

#define MANTISSA_BITS 8u
#define SHARED_EXPONENT (32u - 3u * (MANTISSA_BITS + 1u))
#define SHARED_EXPONENT_OFFSET ((1u << (SHARED_EXPONENT - 1u)) - 1u)

#define MANTISSA_MASK   ((1u << MANTISSA_BITS) - 1u)
#define EXPONENT_MASK   ((1u << SHARED_EXPONENT) - 1u)

#define SIGN1_POS       MANTISSA_BITS
#define MANTISSA2_POS   (MANTISSA_BITS + 1u)
#define SIGN2_POS       (MANTISSA_BITS * 2u + 1u)
#define MANTISSA3_POS   (MANTISSA_BITS * 2u + 2u)
#define SIGN3_POS       (MANTISSA_BITS * 3u + 2u)
#define EXP_POS         (MANTISSA_BITS * 3u + 3u)

__device__ constexpr float compr_relative_error = 1.f / (1u << (MANTISSA_BITS - 1u));
__device__ constexpr float compr_machine_epsilon = compr_relative_error / (1u << SHARED_EXPONENT_OFFSET);

struct ALIGN_BYTES(4) fast_prng
{
    int seed;

    __inline__ __host__ __device__ fast_prng(int seed = 12894712) : seed(seed) {}
    __inline__ __host__ __device__ int generate_int() {
        seed += 1124781675;
        seed ^= seed >> 7;
        return seed *= 1246477263;
    }
    __inline__ __host__ __device__ int generate_int(int min_incl, int max_excl) {
        seed += 1124781675;
        seed ^= seed >> 7;
        seed *= 1246477263;
        return ((int)((((ulong)(uint)seed) * (max_excl - min_incl)) >> 32ull)) + min_incl;
    }
    __inline__ __host__ __device__ float generate_float() {
        seed += 1124781675;
        seed ^= seed >> 7;
        seed *= 1246477263;
        return seed * 2.3283064E-10f;
    }
    __inline__ __host__ __device__ float generate_float01() {
        seed += 1124781675;
        seed ^= seed >> 7;
        seed *= 1246477263;
        return ((uint)seed) * 2.3283064E-10f;
    }
};
struct ALIGN_BYTES(4) compressed_float3
{
    uint data;

    __inline__ constexpr __host__ __device__ compressed_float3() : data(0) {}

private:
    __inline__ constexpr __host__ __device__ uint exponent() const {
        return data >> EXP_POS;
    }
    __inline__ constexpr __host__ __device__ uint sign3() const {
        return (data >> SIGN3_POS) & 1u;
    }
    __inline__ constexpr __host__ __device__ uint sign2() const {
        return (data >> SIGN2_POS) & 1u;
    }
    __inline__ constexpr __host__ __device__ uint sign1() const {
        return (data >> SIGN1_POS) & 1u;
    }
    __inline__ constexpr __host__ __device__ uint mts3() const {
        return (data >> MANTISSA3_POS) & MANTISSA_MASK;
    }
    __inline__ constexpr __host__ __device__ uint mts2() const {
        return (data >> MANTISSA2_POS) & MANTISSA_MASK;
    }
    __inline__ constexpr __host__ __device__ uint mts1() const {
        return data & MANTISSA_MASK;
    }

public:
    __inline__ constexpr __host__ __device__ explicit operator float3() const
    {
        float3 result = float3();
        uint* f1 = (uint*)(&result.x);

        *f1 = (IEEE754_FLOAT_EXPONENT_OFFSET - SHARED_EXPONENT_OFFSET + exponent()) << IEEE754_FLOAT_MANTISSA_BITS;
        result.z = result.y = result.x;

        result.x *= mts1() * (1.f - 2.f * sign1()) * compr_relative_error;
        result.y *= mts2() * (1.f - 2.f * sign2()) * compr_relative_error;
        result.z *= mts3() * (1.f - 2.f * sign3()) * compr_relative_error;

        return result;
    }
    /// Constructs and eliminates correlation errors.
    __inline__ __host__ __device__ compressed_float3(float3 result, fast_prng& rng) : data(0) {
        uint* f1 = (uint*)(&result.x);
        uint* f2 = (uint*)(&result.y);
        uint* f3 = (uint*)(&result.z);

        data = min_uint(max_uint(max_uint(((*f1) >> IEEE754_FLOAT_MANTISSA_BITS) & ((1u << IEEE754_FLOAT_EXPONENT_BITS) - 1u),
            max_uint(((*f2) >> IEEE754_FLOAT_MANTISSA_BITS) & ((1u << IEEE754_FLOAT_EXPONENT_BITS) - 1u),
            ((*f3) >> IEEE754_FLOAT_MANTISSA_BITS) & ((1u << IEEE754_FLOAT_EXPONENT_BITS) - 1u))), 
            IEEE754_FLOAT_EXPONENT_OFFSET - SHARED_EXPONENT_OFFSET) - (IEEE754_FLOAT_EXPONENT_OFFSET - SHARED_EXPONENT_OFFSET),
            (1u << SHARED_EXPONENT) - 1u);

        result.x *= ((1u << SHARED_EXPONENT_OFFSET) / compr_relative_error) / (float(1u << data));
        result.y *= ((1u << SHARED_EXPONENT_OFFSET) / compr_relative_error) / (float(1u << data));
        result.z *= ((1u << SHARED_EXPONENT_OFFSET) / compr_relative_error) / (float(1u << data)); 

        result.x += rng.generate_float();
        result.y += rng.generate_float();
        result.z += rng.generate_float();

        result = clamp(result, 1.f - 2.f / compr_relative_error, 2.f / compr_relative_error - 1.f);
        data <<= EXP_POS; data |= uint(fabsf(result.x) + .5f) | ((result.x < 0.f) << SIGN1_POS);
        data |= (uint(fabsf(result.y) + .5f) << MANTISSA2_POS) | ((result.y < 0.f) << SIGN2_POS);
        data |= (uint(fabsf(result.z) + .5f) << MANTISSA3_POS) | ((result.z < 0.f) << SIGN3_POS);
    }
    __inline__ __host__ __device__ compressed_float3(float3 result) : data(0) {
        uint* f1 = (uint*)(&result.x);
        uint* f2 = (uint*)(&result.y);
        uint* f3 = (uint*)(&result.z);

        data = min_uint(max_uint(max_uint(((*f1) >> IEEE754_FLOAT_MANTISSA_BITS) & ((1u << IEEE754_FLOAT_EXPONENT_BITS) - 1u),
            max_uint(((*f2) >> IEEE754_FLOAT_MANTISSA_BITS) & ((1u << IEEE754_FLOAT_EXPONENT_BITS) - 1u),
                ((*f3) >> IEEE754_FLOAT_MANTISSA_BITS) & ((1u << IEEE754_FLOAT_EXPONENT_BITS) - 1u))),
            IEEE754_FLOAT_EXPONENT_OFFSET - SHARED_EXPONENT_OFFSET) - (IEEE754_FLOAT_EXPONENT_OFFSET - SHARED_EXPONENT_OFFSET),
            (1u << SHARED_EXPONENT) - 1u);

        result.x *= ((1u << SHARED_EXPONENT_OFFSET) / compr_relative_error) / (float(1u << data));
        result.y *= ((1u << SHARED_EXPONENT_OFFSET) / compr_relative_error) / (float(1u << data));
        result.z *= ((1u << SHARED_EXPONENT_OFFSET) / compr_relative_error) / (float(1u << data));

        result = clamp(result, 1.f - 2.f / compr_relative_error, 2.f / compr_relative_error - 1.f);
        data <<= EXP_POS; data |= uint(fabsf(result.x) + .5f) + ((result.x < 0.f) << SIGN1_POS);
        data |= (uint(fabsf(result.y) + .5f) << MANTISSA2_POS) + ((result.y < 0.f) << SIGN2_POS);
        data |= (uint(fabsf(result.z) + .5f) << MANTISSA3_POS) + ((result.z < 0.f) << SIGN3_POS);
    }
};

#undef SIGN1_POS       
#undef MANTISSA2_POS   
#undef SIGN2_POS       
#undef MANTISSA3_POS   
#undef SIGN3_POS       
#undef EXP_POS         

#undef MANTISSA_MASK
#undef EXPONENT_MASK

#undef SHARED_EXPONENT_OFFSET
#undef SHARED_EXPONENT
#undef MANTISSA_BITS

#undef IEEE754_FLOAT_MANTISSA_BITS  
#undef IEEE754_FLOAT_EXPONENT_BITS  
#undef IEEE754_FLOAT_EXPONENT_OFFSET

__inline__ __host__ __device__ void compress(const float3x3& m, compressed_float3* x, compressed_float3* y, compressed_float3* z)
{
    *x = compressed_float3(make_float3(m.xx, m.xy, m.xz));
    *y = compressed_float3(make_float3(m.yx, m.yy, m.yz));
    *z = compressed_float3(make_float3(m.zx, m.zy, m.zz));
}
__inline__ __host__ __device__ void compress(const float3x3_sym& m, compressed_float3* diag, compressed_float3* off_diag)
{
    *diag = compressed_float3(make_float3(m.xx, m.yy, m.zz));
    *off_diag = compressed_float3(make_float3(m.xy, m.xz, m.yz));
}
__inline__ constexpr __host__ __device__ float3x3 decompress(const compressed_float3& x, const compressed_float3& y, const compressed_float3& z)
{
    return float3x3((float3)x, (float3)y, (float3)z);
}
__inline__ constexpr __host__ __device__ float3x3_sym decompress(const compressed_float3& diag, const compressed_float3& off_diag)
{
    float3 diag_v = (float3)diag;
    float3 odiag_v = (float3)off_diag;
    return float3x3_sym(diag_v.x, odiag_v.x, odiag_v.y, diag_v.y, odiag_v.z, diag_v.z);
}

#endif // !TENSOR_H
