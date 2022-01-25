#pragma once
#include "gfx/context.hpp"
#include <string>

namespace gfx {
struct Context;
struct TestPlatformId {
	std::string platformName;
	std::string renderTypeName;

	static TestPlatformId get(const Context& context);
	operator std::string() const;
};
} // namespace gfx
