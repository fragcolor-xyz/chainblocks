/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

#ifndef SH_LINALG_SHIM_HPP
#define SH_LINALG_SHIM_HPP

#include "shards.hpp"
#include "math_ops.hpp"
#include <linalg.h>
#include <vector>

namespace shards {
struct alignas(16) Mat4 : public linalg::aliases::float4x4 {
  using linalg::aliases::float4x4::mat;
  Mat4(const SHVar &var) { *this = var; }
  Mat4(const linalg::aliases::float4x4 &other) : linalg::aliases::float4x4(other) {}

  template <typename NUMBER> static Mat4 FromVector(const std::vector<NUMBER> &mat) {
    // used by gltf
    assert(mat.size() == 16);
    int idx = 0;
    Mat4 res;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        res[i][j] = float(mat[idx]);
        idx++;
      }
    }
    return res;
  }

  template <typename NUMBER> static Mat4 FromArray(const std::array<NUMBER, 16> &mat) {
    // used by gltf
    assert(mat.size() == 16);
    int idx = 0;
    Mat4 res;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        res[i][j] = float(mat[idx]);
        idx++;
      }
    }
    return res;
  }

  static Mat4 FromArrayUnsafe(const float *mat) {
    // used by gltf
    int idx = 0;
    Mat4 res;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        res[i][j] = mat[idx];
        idx++;
      }
    }
    return res;
  }

  static void ToArrayUnsafe(const Mat4 &mat, float *array) {
    int idx = 0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        array[idx] = mat[i][j];
        idx++;
      }
    }
  }

  static Mat4 Identity() {
    Mat4 res;
    res[0] = {1, 0, 0, 0};
    res[1] = {0, 1, 0, 0};
    res[2] = {0, 0, 1, 0};
    res[3] = {0, 0, 0, 1};
    return res;
  }

  Mat4 &operator=(linalg::aliases::float4x4 &&mat) {
    (*this)[0] = std::move(mat[0]);
    (*this)[1] = std::move(mat[1]);
    (*this)[2] = std::move(mat[2]);
    (*this)[3] = std::move(mat[3]);
    return *this;
  }

  Mat4 &operator=(const SHVar &var) {
    if (var.valueType != SHType::Seq || var.payload.seqValue.len != 4)
      throw SHException("Invalid Mat4 variable given as input");
    auto vm = reinterpret_cast<const Mat4 *>(var.payload.seqValue.elements);
    (*this)[0] = (*vm)[0];
    (*this)[1] = (*vm)[1];
    (*this)[2] = (*vm)[2];
    (*this)[3] = (*vm)[3];
    return *this;
  }

  operator SHVar() const {
    SHVar res{};
    res.valueType = SHType::Seq;
    res.payload.seqValue.elements = reinterpret_cast<SHVar *>(const_cast<shards::Mat4 *>(this));
    res.payload.seqValue.len = 4;
    res.payload.seqValue.cap = 0;
    for (auto i = 0; i < 4; i++) {
      res.payload.seqValue.elements[i].valueType = SHType::Float4;
    }
    return res;
  }
};

struct alignas(16) Vec2 : public linalg::aliases::float2 {
  using linalg::aliases::float2::vec;

  Vec2 &operator=(const SHVar &var);

  template <typename XY_TYPE> Vec2(const XY_TYPE &vec) {
    x = float(vec.x);
    y = float(vec.y);
  }

  template <typename NUMBER> Vec2(NUMBER x_, NUMBER y_) {
    x = float(x_);
    y = float(y_);
  }

  Vec2 &operator=(const linalg::aliases::float2 &vec) {
    x = vec.x;
    y = vec.y;
    return *this;
  }
};

struct alignas(16) Vec3 : public linalg::aliases::float3 {
  using linalg::aliases::float3::vec;

  Vec3 &operator=(const SHVar &var);

  template <typename XYZ_TYPE> Vec3(const XYZ_TYPE &vec) {
    x = float(vec.x);
    y = float(vec.y);
    z = float(vec.z);
  }

  template <typename NUMBER> Vec3(NUMBER x_, NUMBER y_, NUMBER z_) {
    x = float(x_);
    y = float(y_);
    z = float(z_);
  }

  template <typename NUMBER> static Vec3 FromVector(const std::vector<NUMBER> &vec) {
    // used by gltf
    assert(vec.size() == 3);
    Vec3 res;
    for (int j = 0; j < 3; j++) {
      res[j] = float(vec[j]);
    }
    return res;
  }

  Vec3 &applyMatrix(const linalg::aliases::float4x4 &mat) {
    const auto w = 1.0f / (mat.x.w * x + mat.y.w * y + mat.z.w * z + mat.w.w);
    x = (mat.x.x * x + mat.y.x * y + mat.z.x * z + mat.w.x) * w;
    y = (mat.x.y * x + mat.y.y * y + mat.z.y * z + mat.w.y) * w;
    z = (mat.x.z * x + mat.y.z * y + mat.z.z * z + mat.w.z) * w;
    return *this;
  }

  Vec3 &operator=(const linalg::aliases::float3 &vec) {
    x = vec.x;
    y = vec.y;
    z = vec.z;
    return *this;
  }

  operator SHVar() const {
    auto v = reinterpret_cast<SHVar *>(const_cast<shards::Vec3 *>(this));
    v->valueType = SHType::Float3;
    return *v;
  }
};

struct alignas(16) Vec4 : public linalg::aliases::float4 {
  using linalg::aliases::float4::vec;

  template <typename XYZW_TYPE> Vec4(const XYZW_TYPE &vec) {
    x = float(vec.x);
    y = float(vec.y);
    z = float(vec.z);
    w = float(vec.w);
  }

  template <typename NUMBER> Vec4(NUMBER x_, NUMBER y_, NUMBER z_, NUMBER w_) {
    x = float(x_);
    y = float(y_);
    z = float(z_);
    w = float(w_);
  }

  Vec4 &operator=(const SHVar &var);

  constexpr static Vec4 Quaternion() {
    Vec4 q;
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0;
    return q;
  }

  template <typename NUMBER> static Vec4 FromVector(const std::vector<NUMBER> &vec) {
    // used by gltf
    assert(vec.size() == 4);
    Vec4 res;
    for (int j = 0; j < 4; j++) {
      res[j] = float(vec[j]);
    }
    return res;
  }

  Vec4 &operator=(const linalg::aliases::float4 &vec) {
    x = vec.x;
    y = vec.y;
    z = vec.z;
    w = vec.w;
    return *this;
  }

  operator SHVar() const {
    auto v = reinterpret_cast<SHVar *>(const_cast<shards::Vec4 *>(this));
    v->valueType = SHType::Float4;
    return *v;
  }
};

template <typename T> struct VectorConversion {};
template <> struct VectorConversion<float> {
  static inline constexpr SHType ShardsType = SHType::Float;
  static auto convert(const SHVar &value) { return value.payload.floatValue; }
  static SHVar rconvert(const float &value) { return SHVar{.payload = {.floatValue = value}, .valueType = ShardsType}; }
};
template <int N> struct VectorConversion<linalg::vec<float, N>> {
  static inline constexpr SHType ShardsType = SHType(int(SHType::Float) + N - 1);
  static auto convert(const SHVar &value) {
    static_assert(N <= 4, "Not implemented for N > 4");
    linalg::vec<float, N> r;
    for (int i = 0; i < N; i++)
      r[i] = Math::PayloadTraits<ShardsType>{}.getContents(const_cast<SHVarPayload &>(value.payload))[i];
    return r;
  }
  static SHVar rconvert(const linalg::vec<float, N> &v) {
    static_assert(N <= 4, "Not implemented for N > 4");
    SHVar r{};
    r.valueType = ShardsType;
    for (int i = 0; i < N; i++)
      Math::PayloadTraits<ShardsType>{}.getContents(const_cast<SHVarPayload &>(r.payload))[i] = v[i];
    return r;
  }
};

template <int N> struct VectorConversion<linalg::vec<int, N>> {
  static inline constexpr SHType ShardsType = SHType(int(SHType::Int) + N - 1);
  static auto convert(const SHVar &value) {
    static_assert(N <= 4, "Not implemented for N > 4");
    linalg::vec<int, N> r;
    for (int i = 0; i < N; i++)
      r[i] = int(Math::PayloadTraits<ShardsType>{}.getContents(const_cast<SHVarPayload &>(value.payload))[i]);
    return r;
  }
  static SHVar rconvert(const linalg::vec<int, N> &v) {
    static_assert(N <= 4, "Not implemented for N > 4");
    SHVar r{};
    r.valueType = ShardsType;
    for (int i = 0; i < N; i++)
      Math::PayloadTraits<ShardsType>{}.getContents(const_cast<SHVarPayload &>(r.payload))[i] = v[i];
    return r;
  }
};

template <typename TVec> inline auto toVec(const SHVar &value) {
  using Conv = VectorConversion<TVec>;
  if (value.valueType != Conv::ShardsType)
    throw std::runtime_error(fmt::format("Invalid vector type {}, expected {}", value.valueType, Conv::ShardsType));
  return Conv::convert(value);
}

inline auto toFloat2(const SHVar &value) { return toVec<linalg::aliases::float2>(value); }
inline auto toFloat3(const SHVar &value) { return toVec<linalg::aliases::float3>(value); }
inline auto toFloat4(const SHVar &value) { return toVec<linalg::aliases::float4>(value); }
inline auto toInt2(const SHVar &value) { return toVec<linalg::aliases::int2>(value); }
inline auto toInt3(const SHVar &value) { return toVec<linalg::aliases::int3>(value); }
inline auto toInt4(const SHVar &value) { return toVec<linalg::aliases::int4>(value); }

template <typename TVec> inline SHVar genericToVar(const TVec &value) {
  using Conv = VectorConversion<TVec>;
  return Conv::rconvert(value);
}

inline auto toVar(const linalg::aliases::float2 &value) { return genericToVar(value); }
inline auto toVar(const linalg::aliases::float3 &value) { return genericToVar(value); }
inline auto toVar(const linalg::aliases::float4 &value) { return genericToVar(value); }
inline auto toVar(const linalg::aliases::int2 &value) { return genericToVar(value); }
inline auto toVar(const linalg::aliases::int3 &value) { return genericToVar(value); }
inline auto toVar(const linalg::aliases::int4 &value) { return genericToVar(value); }

inline linalg::aliases::float4x4 toFloat4x4(const SHVar &vec) { return Mat4(vec); }

constexpr linalg::aliases::float3 AxisX{1.0, 0.0, 0.0};
constexpr linalg::aliases::float3 AxisY{0.0, 1.0, 0.0};
constexpr linalg::aliases::float3 AxisZ{0.0, 0.0, 1.0};

}; // namespace shards

#endif
