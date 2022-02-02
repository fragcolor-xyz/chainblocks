#include <catch2/catch_all.hpp>
#include <cctype>
#include <gfx/context.hpp>
#include <gfx/shader/blocks.hpp>
#include <gfx/shader/generator.hpp>
#include <spdlog/spdlog.h>

using namespace gfx;
using namespace gfx::shader;
using String = std::string;

bool compareNoWhitespace(const String &a, const String &b) {
	using Iterator = String::const_iterator;
	using StringView = std::string_view;

	auto isWhitespace = [](const Iterator &it) { return *it == '\n' || *it == '\t' || *it == '\r' || *it == ' '; };
	auto isSymbol = [](const Iterator &it) { return !std::isalnum(*it); };

	// Skip whitespace, returns true on end of string
	auto skipWhitespace = [=](Iterator &it, const Iterator &end) {
		do {
			if (isWhitespace(it)) {
				it++;
			} else {
				return false;
			}
		} while (it != end);
		return true;
	};

	auto split = [=](StringView &sv, Iterator &it, const Iterator &end) {
		if (skipWhitespace(it, end))
			return false;

		auto itStart = it;
		size_t len = 0;
		bool hasSymbols = false;
		do {
			if (isSymbol(it)) {
				hasSymbols = true;
				if (len > 0)
					break;
			}
			it++;
			len++;
		} while (it != end && !hasSymbols && !isWhitespace(it));
		sv = StringView(&*itStart, len);
		return true;
	};

	auto itA = a.cbegin();
	auto itB = b.cbegin();
	auto shouldAbort = [&]() { return itA == a.end() || itB == b.end(); };

	while (!shouldAbort()) {
		std::string_view svA;
		std::string_view svB;
		if (!split(svA, itA, a.end()))
			break;
		if (!split(svB, itB, b.end()))
			break;
		if (svA != svB)
			return false;
	}

	bool endA = itA == a.end() ? true : skipWhitespace(itA, a.end());
	bool endB = itB == b.end() ? true : skipWhitespace(itB, b.end());
	return endA == endB;
}

TEST_CASE("Compare & ignore whitespace", "[Shader]") {
	String a = R"(string with whitespace)";
	String b = R"(
string 	with
	whitespace

)";
	CHECK(compareNoWhitespace(a, b));

	String c = "stringwithwhitespace";
	CHECK(!compareNoWhitespace(a, c));

	String d = "string\rwith\nwhitespace";
	CHECK(compareNoWhitespace(a, d));

	String e = "string\twith whitespace";
	CHECK(compareNoWhitespace(a, e));

	String f = "string\t";
	CHECK(!compareNoWhitespace(a, f));
}

static bool validateShaderModule(Context &context, const String &code) {
	WGPUShaderModuleDescriptor desc{};
	WGPUShaderModuleWGSLDescriptor wgslDesc{};
	desc.nextInChain = &wgslDesc.chain;
	wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
	wgslDesc.code = code.c_str();
	WGPUShaderModule module = wgpuDeviceCreateShaderModule(context.wgpuDevice, &desc);
	if (module)
		wgpuShaderModuleRelease(module);
	return module != nullptr;
}

TEST_CASE("Shader basic", "[Shader]") {
	Generator generator;
	generator.meshFormat.vertexAttributes = {
		MeshVertexAttribute("position", 3),
		MeshVertexAttribute("normal", 3),
		MeshVertexAttribute("texcoord0", 2),
		MeshVertexAttribute("texcoord1", 3),
	};

	Context context;
	context.init();

	UniformBufferLayoutBuilder viewLayoutBuilder;
	viewLayoutBuilder.push("viewProj", ShaderParamType::Float4x4);
	generator.viewBufferLayout = viewLayoutBuilder.finalize();

	UniformBufferLayoutBuilder objectLayoutBuilder;
	objectLayoutBuilder.push("world", ShaderParamType::Float4x4);
	generator.objectBufferLayout = objectLayoutBuilder.finalize();

	generator.interpolatedFields.emplace_back("position", FieldType(ShaderFieldBaseType::Float32, 4));
	generator.interpolatedFields.emplace_back("color", FieldType(ShaderFieldBaseType::Float32, 4));

	generator.outputFields.emplace_back("color", FieldType(ShaderFieldBaseType::Float32, 4));

	std::vector<EntryPoint> entryPoints;
	auto vec4Pos = blocks::makeCompoundBlock("vec4<f32>(", blocks::ReadInput("position"), ".xyz, 1.0)");
	entryPoints.emplace_back("position", ProgrammableGraphicsStage::Vertex,
							 blocks::makeCompoundBlock(blocks::WriteOutput("position", std::move(vec4Pos), "*", blocks::ReadBuffer("object"), ".world", "*",
																		   blocks::ReadBuffer("view"), ".viewProj")));

	entryPoints.emplace_back("color", ProgrammableGraphicsStage::Fragment,
							 blocks::makeCompoundBlock(blocks::WriteOutput("color", "vec4<f32>(0.0, 1.0, 0.0, 1.0);")));

	GeneratorOutput output = generator.build(entryPoints);
	spdlog::info(output.wgslSource);

	CHECK(validateShaderModule(context, output.wgslSource));
}

TEST_CASE("Shader globals & dependencies", "[Shader]") {
	Generator generator;
	generator.meshFormat.vertexAttributes = {
		MeshVertexAttribute("position", 3),
	};

	Context context;
	context.init();

	UniformBufferLayoutBuilder viewLayoutBuilder;
	viewLayoutBuilder.push("viewProj", ShaderParamType::Float4x4);
	generator.viewBufferLayout = viewLayoutBuilder.finalize();

	UniformBufferLayoutBuilder objectLayoutBuilder;
	objectLayoutBuilder.push("world", ShaderParamType::Float4x4);
	generator.objectBufferLayout = objectLayoutBuilder.finalize();

	generator.interpolatedFields.emplace_back("position", FieldType(ShaderFieldBaseType::Float32, 4));
	generator.outputFields.emplace_back("color", FieldType(ShaderFieldBaseType::Float32, 4));

	std::vector<EntryPoint> entryPoints;
	auto vec4Pos = blocks::makeCompoundBlock("vec4<f32>(", blocks::ReadInput("position"), ".xyz, 1.0)");
	entryPoints.emplace_back("position", ProgrammableGraphicsStage::Vertex,
							 blocks::makeCompoundBlock(blocks::WriteOutput("position", std::move(vec4Pos), "*", blocks::ReadBuffer("object"), ".world", "*",
																		   blocks::ReadBuffer("view"), ".viewProj")));

	entryPoints.emplace_back(
		"colorDefault", ProgrammableGraphicsStage::Fragment,
		blocks::makeCompoundBlock(blocks::WriteGlobal("color", FieldType(ShaderFieldBaseType::Float32, 4), "vec4<f32>(0.0, 1.0, 0.0, 1.0);")));

	entryPoints.emplace_back("color", ProgrammableGraphicsStage::Fragment,
							 blocks::makeCompoundBlock(blocks::WriteOutput("color", blocks::ReadGlobal("color"))));
	entryPoints.back().dependencies.emplace_back("colorDefault", DependencyType::After);

	GeneratorOutput output = generator.build(entryPoints);
	spdlog::info(output.wgslSource);

	CHECK(validateShaderModule(context, output.wgslSource));
}