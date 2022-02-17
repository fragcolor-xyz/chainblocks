#pragma once
#include "fwd.hpp"
#include "imgui/imgui.h"
#include "linalg.hpp"
#include "window.hpp"


namespace gfx {

inline ImVec2 toImVec2(const float2 &v) { return ImVec2(v.x, v.y); }

inline ImVec2 toImVec2(const int2 &v) { return ImVec2(v.x, v.y); }

struct ImGuiRenderer {
	Context &context;
	ImGuiContext *imguiContext{};
	SDL_Window* sdlWindow{};

	ImGuiRenderer(Context &context);
	ImGuiRenderer(const ImGuiContext &other) = delete;
	~ImGuiRenderer();

	void beginFrame(const std::vector<SDL_Event> &inputEvents);
	void endFrame();
	void render();

private:
	void init();
	void cleanup();
};

} // namespace gfx
