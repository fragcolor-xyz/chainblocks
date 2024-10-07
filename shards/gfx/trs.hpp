#ifndef FE5CC8D3_6FD1_4BE7_821E_C1C35CD76DBF
#define FE5CC8D3_6FD1_4BE7_821E_C1C35CD76DBF

#include "linalg.hpp"

namespace gfx {

struct TRS;
struct TRSRef {
  float3 &translation;
  float4 &rotation;
  float3 &scale;
  TRSRef(float3 &translation, float4 &rotation, float3 &scale) : translation(translation), rotation(rotation), scale(scale) {}
  TRSRef(TRS &trs);
  explicit TRSRef(const TRS &trs);
  operator TRS() const;

  TRSRef &operator=(const float4x4 &other) {
    float3x3 rotationMatrix;
    decomposeTRS(other, translation, scale, rotationMatrix);
    this->rotation = linalg::rotation_quat(rotationMatrix);
    return *this;
  }

  float4x4 getMatrix() const {
    float4x4 rot = linalg::rotation_matrix(this->rotation);
    float4x4 scale = linalg::scaling_matrix(this->scale);
    float4x4 tsl = linalg::translation_matrix(this->translation);
    return linalg::mul(tsl, linalg::mul(rot, scale));
  }
};

struct TRS {
  float3 translation;
  float3 scale = float3(1.0f);
  float4 rotation = float4(0.0f, 0.0f, 0.0f, 1.0f);

  TRS() = default;
  TRS(const float3 &translation) : translation(translation) {}
  TRS(const float3 &translation, const float4 &rotation, const float3 &scale = float3(1.0f))
      : translation(translation), scale(scale), rotation(rotation) {}
  TRS(const float4x4 &other) { *this = other; }
  TRS(const TRS& other) = default;
  TRS &operator=(const TRS &other) = default;
  TRS &operator=(const float4x4 &other) {
    TRSRef(*this) = other;
    return *this;
  }

  bool operator==(const TRS &other) const {
    return translation == other.translation && scale == other.scale && rotation == other.rotation;
  }
  bool operator!=(const TRS &other) const { return !(*this == other); }

  float4x4 getMatrix() const { return TRSRef(*this).getMatrix(); }

  TRS translated(const float3 &delta) const { return TRS(translation + delta, rotation, scale); }
  TRS rotated(const float4 &q) const { return TRS(translation, linalg::qmul(q, rotation), scale); }
  TRS rotatedAround(const float3 &pivot, const float4 &q) const {
    float4x4 mat = linalg::translation_matrix(-pivot);
    mat = linalg::mul(linalg::rotation_matrix(q), mat);
    mat = linalg::mul(linalg::translation_matrix(pivot), mat);
    return TRS(linalg::mul(mat, getMatrix()));
  }
  TRS scaleAround(const TRS &pivot, const float3 &s) const {
    float4 toRotSpaceQ = linalg::qmul(rotation, linalg::qinv(pivot.rotation));
    float3 relScale =  linalg::qrot(toRotSpaceQ, scale) * s;
    float3 newLocalScale = linalg::qrot(linalg::qinv(toRotSpaceQ), relScale);

    float3 relTsl = (translation - pivot.translation) * s;
    return TRS(pivot.translation + relTsl, rotation, newLocalScale);
  }
  TRS scaled(const float3 &s) const { return TRS(translation, rotation, scale * s); }
};

inline TRSRef::TRSRef(TRS &trs) : translation(trs.translation), rotation(trs.rotation), scale(trs.scale) {}
inline TRSRef::TRSRef(const TRS &trs) : TRSRef(const_cast<TRS &>(trs)) {}
inline TRSRef::operator TRS() const { return TRS(translation, rotation, scale); }

} // namespace gfx

#endif /* FE5CC8D3_6FD1_4BE7_821E_C1C35CD76DBF */
