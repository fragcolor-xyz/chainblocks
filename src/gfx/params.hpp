#pragma once
#include "fwd.hpp"
#include "linalg.hpp"
#include <memory>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <variant>

namespace gfx {


enum class ShaderParamType : uint8_t {
#define PARAM_TYPE(_cppType, _displayName, ...) _displayName,
#include "param_types.def"
#undef PARAM_TYPE
};

typedef std::variant<std::monostate, float, float2, float3, float4, float4x4> ParamVariant;

size_t packParamVariant(uint8_t *outData, size_t outLength, const ParamVariant &variant);
ShaderParamType getParamVariantType(const ParamVariant &variant);
size_t getParamTypeSize(ShaderParamType type);
size_t getParamTypeWGSLAlignment(ShaderParamType type);

struct IDrawDataCollector {
	virtual void setParam(const std::string &name, ParamVariant &&value) = 0;
};

struct DrawData : IDrawDataCollector {
	std::unordered_map<std::string, ParamVariant> data;

	void setParam(const std::string &name, ParamVariant &&value) { data.insert_or_assign(name, std::move(value)); }
	void setParam(const std::string &name, const ParamVariant &value) { data.insert_or_assign(name, value); }
	void append(const DrawData &other) {
		for (auto &it : other.data) {
			data.insert_or_assign(it.first, it.second);
		}
	}
};

} // namespace gfx
