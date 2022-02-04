#pragma once
#include "enums.hpp"
#include "linalg.hpp"
#include <functional>
#include <memory>
#include <optional>
#include <stdint.h>
#include <string>

namespace gfx {
namespace shader {
struct EntryPoint;
}

struct BlendComponent {
	WGPUBlendOperation operation{};
	WGPUBlendFactor srcFactor{};
	WGPUBlendFactor dstFactor{};

	template <typename T> void hashStatic(T &hasher) const {
		hasher(operation);
		hasher(srcFactor);
		hasher(dstFactor);
	}
	bool operator==(const BlendComponent &other) const = default;
	bool operator!=(const BlendComponent &other) const = default;
};

struct BlendState {
	BlendComponent color;
	BlendComponent alpha;

	operator const WGPUBlendState &() const { return *reinterpret_cast<const WGPUBlendState *>(this); }
	template <typename T> void hashStatic(T &hasher) const {
		hasher(color);
		hasher(alpha);
	}
	bool operator==(const BlendState &other) const = default;
	bool operator!=(const BlendState &other) const = default;
};
} // namespace gfx

namespace std {
template <> struct hash<gfx::BlendComponent> {
	size_t operator()(const gfx::BlendComponent &v) {
		size_t result{};
		result = std::hash<WGPUBlendOperation>{}(v.operation);
		result = (result * 3) ^ std::hash<WGPUBlendFactor>{}(v.srcFactor);
		result = (result * 3) ^ std::hash<WGPUBlendFactor>{}(v.dstFactor);
		return result;
	}
};

template <> struct hash<gfx::BlendState> {
	size_t operator()(const gfx::BlendState &v) {
		size_t result{};
		result = std::hash<gfx::BlendComponent>{}(v.color);
		result = (result * 3) ^ std::hash<gfx::BlendComponent>{}(v.alpha);
		return result;
	}
};
} // namespace std

namespace gfx {

struct FeaturePipelineState {
#define PIPELINE_STATE(_name, _type)                              \
public:                                                           \
	std::optional<_type> _name;                                   \
                                                                  \
public:                                                           \
	void set_##_name(const _type &_name) { this->_name = _name; } \
	void clear_##_name(const _type &_name) { this->_name.reset(); }
#include "pipeline_states.def"

public:
	FeaturePipelineState combine(const FeaturePipelineState &other) const {
		FeaturePipelineState result = *this;
#define PIPELINE_STATE(_name, _type) \
	if (other._name)                 \
		result._name = other._name;
#include "pipeline_states.def"
		return result;
	}

	bool operator==(const FeaturePipelineState &other) const {
#define PIPELINE_STATE(_name, _type) \
	if (_name != other._name)        \
		return false;
#include "pipeline_states.def"
		return true;
	}

	bool operator!=(const FeaturePipelineState &other) const { return !(*this == other); }

	size_t getHash() const {
		size_t hash = 0;
#define PIPELINE_STATE(_name, _type) hash = (hash * 3) ^ std::hash<decltype(_name)>{}(_name);
#include "pipeline_states.def"
		return hash;
	}

	template <typename T> void hashStatic(T &hasher) const {
#define PIPELINE_STATE(_name, _type) hasher(_name);
#include "pipeline_states.def"
	}
};

struct Feature {
	FeaturePipelineState state;
	// std::vector<FeatureDrawDataFunction> sharedDrawData;
	// std::vector<FeatureDrawDataFunction> drawData;
	// std::vector<FeaturePrecomputeFunction> precompute;
	// std::vector<FeatureShaderField> shaderParams;
	// std::vector<FeatureShaderField> shaderGlobals;
	// std::vector<FeatureShaderField> shaderVaryings;
	std::vector<std::unique_ptr<shader::EntryPoint>> shaderEntryPoints;

	virtual ~Feature() = default;

	template <typename T> void hashStatic(T &hasher) const {
		hasher(state);
		// hasher(shaderParams);
		// hasher(shaderGlobals);
		// hasher(shaderVaryings);
		// hasher(shaderCode);
	}
};
typedef std::shared_ptr<Feature> FeaturePtr;

} // namespace gfx
