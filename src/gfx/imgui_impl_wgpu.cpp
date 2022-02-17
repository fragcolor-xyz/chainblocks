// dear imgui: Renderer for WebGPU
// This needs to be used along with a Platform Binding (e.g. GLFW)
// (Please note that WebGPU is currently experimental, will not run on non-beta browsers, and may break.)

// Implemented features:
//  [X] Renderer: User texture binding. Use 'WGPUTextureView' as ImTextureID. Read the FAQ about ImTextureID!
//  [X] Renderer: Support for large meshes (64k+ vertices) with 16-bit indices.

// You can use unmodified imgui_impl_* files in your project. See examples/ folder for examples of using this.
// Prefer including the entire imgui/ repository into your project (either as a copy or as a submodule), and only build the backends you need.
// If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs

// CHANGELOG
// (minor and older changes stripped away, please see git history for details)
//  2021-11-29: Passing explicit buffer sizes to wgpuRenderPassEncoderSetVertexBuffer()/wgpuRenderPassEncoderSetIndexBuffer().
//  2021-08-24: Fix for latest specs.
//  2021-05-24: Add support for draw_data->FramebufferScale.
//  2021-05-19: Replaced direct access to ImDrawCmd::TextureId with a call to ImDrawCmd::GetTexID(). (will become a requirement)
//  2021-05-16: Update to latest WebGPU specs (compatible with Emscripten 2.0.20 and Chrome Canary 92).
//  2021-02-18: Change blending equation to preserve alpha in output buffer.
//  2021-01-28: Initial version.

#include "imgui.h"
#include "imgui_impl_wgpu.h"
#include <limits.h>
#include "gfx_wgpu.hpp"

#define HAS_EMSCRIPTEN_VERSION(major, minor, tiny) (__EMSCRIPTEN_major__ > (major) || (__EMSCRIPTEN_major__ == (major) && __EMSCRIPTEN_minor__ > (minor)) || (__EMSCRIPTEN_major__ == (major) && __EMSCRIPTEN_minor__ == (minor) && __EMSCRIPTEN_tiny__ >= (tiny)))

#if defined(__EMSCRIPTEN__) && !HAS_EMSCRIPTEN_VERSION(2, 0, 20)
#error "Requires at least emscripten 2.0.20"
#endif

// Dear ImGui prototypes from imgui_internal.h
extern ImGuiID ImHashData(const void* data_p, size_t data_size, ImU32 seed = 0);

// WebGPU data
static WGPUDevice               g_wgpuDevice = NULL;
static WGPUQueue                g_defaultQueue = NULL;
static WGPUTextureFormat        g_renderTargetFormat = WGPUTextureFormat_Undefined;
static WGPURenderPipeline       g_pipelineState = NULL;

struct RenderResources
{
    WGPUTexture         FontTexture;            // Font texture
    WGPUTextureView     FontTextureView;        // Texture view for font texture
    WGPUSampler         Sampler;                // Sampler for the font texture
    WGPUBuffer          Uniforms;               // Shader uniforms
    WGPUBindGroup       CommonBindGroup;        // Resources bind-group to bind the common resources to pipeline
    ImGuiStorage        ImageBindGroups;        // Resources bind-group to bind the font/image resources to pipeline (this is a key->value map)
    WGPUBindGroup       ImageBindGroup;         // Default font-resource of Dear ImGui
    WGPUBindGroupLayout ImageBindGroupLayout;   // Cache layout used for the image bind group. Avoids allocating unnecessary JS objects when working with WebASM
};
static RenderResources  g_resources;

struct FrameResources
{
    WGPUBuffer  IndexBuffer;
    WGPUBuffer  VertexBuffer;
    ImDrawIdx*  IndexBufferHost;
    ImDrawVert* VertexBufferHost;
    int         IndexBufferSize;
    int         VertexBufferSize;
};
static FrameResources*  g_pFrameResources = NULL;
static unsigned int     g_numFramesInFlight = 0;
static unsigned int     g_frameIndex = UINT_MAX;

struct Uniforms
{
    float MVP[4][4];
};

//-----------------------------------------------------------------------------
// SHADERS
//-----------------------------------------------------------------------------
static const char *__wgsl_shader = R"(
struct Block0 {
    mvp: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> block0: Block0;

[[group(0), binding(1)]]
var s_color: sampler;

[[group(1), binding(0)]]
var t_color: texture_2d<f32>;

struct VSOut {
    [[builtin(position)]] Position: vec4<f32>;
    [[location(0)]] Color: vec4<f32>;
    [[location(1)]] UV: vec2<f32>;
};

struct FSIn {
    [[location(0)]] Color: vec4<f32>;
    [[location(1)]] UV: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] aPos: vec2<f32>,
    [[location(1)]] aUV: vec2<f32>,
    [[location(2)]] aColor: vec4<f32>) -> VSOut {
    var Out: VSOut;
    Out.Color = aColor;
    Out.UV = aUV;
    Out.Position = block0.mvp * vec4<f32>(aPos, 0.0, 1.0);
    return Out;
}

[[stage(fragment)]]
fn fs_main(In: FSIn) -> [[location(0)]] vec4<f32> {
    return In.Color * textureSample(t_color, s_color, In.UV);
}
)";

static size_t alignTo(size_t size, size_t alignTo) {
		size_t alignment = alignTo;
		if (alignment == 0)
			return size;

		size_t remainder = size % alignment;
		if (remainder > 0) {
			return size + (alignment - remainder);
		}
		return size;
}

static void SafeRelease(ImDrawIdx*& res)
{
    if (res)
        delete[] res;
    res = NULL;
}
static void SafeRelease(ImDrawVert*& res)
{
    if (res)
        delete[] res;
    res = NULL;
}
static void SafeRelease(WGPUBindGroupLayout& res)
{
    if (res)
        wgpuBindGroupLayoutRelease(res);
    res = NULL;
}
static void SafeRelease(WGPUBindGroup& res)
{
    if (res)
        wgpuBindGroupRelease(res);
    res = NULL;
}
static void SafeRelease(WGPUBuffer& res)
{
    if (res)
        wgpuBufferRelease(res);
    res = NULL;
}
static void SafeRelease(WGPURenderPipeline& res)
{
    if (res)
        wgpuRenderPipelineRelease(res);
    res = NULL;
}
static void SafeRelease(WGPUSampler& res)
{
    if (res)
        wgpuSamplerRelease(res);
    res = NULL;
}
static void SafeRelease(WGPUShaderModule& res)
{
    if (res)
        wgpuShaderModuleRelease(res);
    res = NULL;
}
static void SafeRelease(WGPUPipelineLayout& res)
{
    if (res)
        wgpuPipelineLayoutRelease(res);
    res = NULL;
}
static void SafeRelease(WGPUTextureView& res)
{
    if (res)
        wgpuTextureViewRelease(res);
    res = NULL;
}
static void SafeRelease(WGPUTexture& res)
{
    if (res)
        wgpuTextureRelease(res);
    res = NULL;
}

static void SafeRelease(RenderResources& res)
{
    SafeRelease(res.FontTexture);
    SafeRelease(res.FontTextureView);
    SafeRelease(res.Sampler);
    SafeRelease(res.Uniforms);
    SafeRelease(res.CommonBindGroup);
    SafeRelease(res.ImageBindGroup);
    SafeRelease(res.ImageBindGroupLayout);
};

static void SafeRelease(FrameResources& res)
{
    SafeRelease(res.IndexBuffer);
    SafeRelease(res.VertexBuffer);
    SafeRelease(res.IndexBufferHost);
    SafeRelease(res.VertexBufferHost);
}

static WGPUBindGroup ImGui_ImplWGPU_CreateImageBindGroup(WGPUBindGroupLayout layout, WGPUTextureView texture)
{
    WGPUBindGroupEntry image_bg_entries[] = { { nullptr, 0, 0, 0, 0, 0, texture } };

    WGPUBindGroupDescriptor image_bg_descriptor = {};
    image_bg_descriptor.layout = layout;
    image_bg_descriptor.entryCount = sizeof(image_bg_entries) / sizeof(WGPUBindGroupEntry);
    image_bg_descriptor.entries = image_bg_entries;
    return wgpuDeviceCreateBindGroup(g_wgpuDevice, &image_bg_descriptor);
}

static void ImGui_ImplWGPU_SetupRenderState(ImDrawData* draw_data, WGPURenderPassEncoder ctx, FrameResources* fr)
{
    // Setup orthographic projection matrix into our constant buffer
    // Our visible imgui space lies from draw_data->DisplayPos (top left) to draw_data->DisplayPos+data_data->DisplaySize (bottom right).
    {
        float L = draw_data->DisplayPos.x;
        float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
        float T = draw_data->DisplayPos.y;
        float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
        float mvp[4][4] =
        {
            { 2.0f/(R-L),   0.0f,           0.0f,       0.0f },
            { 0.0f,         2.0f/(T-B),     0.0f,       0.0f },
            { 0.0f,         0.0f,           0.5f,       0.0f },
            { (R+L)/(L-R),  (T+B)/(B-T),    0.5f,       1.0f },
        };
        wgpuQueueWriteBuffer(g_defaultQueue, g_resources.Uniforms, 0, mvp, sizeof(mvp));
    }

    // Setup viewport
    wgpuRenderPassEncoderSetViewport(ctx, 0, 0, draw_data->FramebufferScale.x * draw_data->DisplaySize.x, draw_data->FramebufferScale.y * draw_data->DisplaySize.y, 0, 1);

    // Bind shader and vertex buffers
    wgpuRenderPassEncoderSetVertexBuffer(ctx, 0, fr->VertexBuffer, 0, fr->VertexBufferSize * sizeof(ImDrawVert));
    wgpuRenderPassEncoderSetIndexBuffer(ctx, fr->IndexBuffer, sizeof(ImDrawIdx) == 2 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32, 0, fr->IndexBufferSize * sizeof(ImDrawIdx));
    wgpuRenderPassEncoderSetPipeline(ctx, g_pipelineState);
    wgpuRenderPassEncoderSetBindGroup(ctx, 0, g_resources.CommonBindGroup, 0, NULL);

    // Setup blend factor
    WGPUColor blend_color = { 0.f, 0.f, 0.f, 0.f };
    wgpuRenderPassEncoderSetBlendConstant(ctx, &blend_color);
}

// Render function
// (this used to be set in io.RenderDrawListsFn and called by ImGui::Render(), but you can now call this directly from your main loop)
void ImGui_ImplWGPU_RenderDrawData(ImDrawData* draw_data, WGPURenderPassEncoder pass_encoder)
{
    // Avoid rendering when minimized
    if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f)
        return;

    // FIXME: Assuming that this only gets called once per frame!
    // If not, we can't just re-allocate the IB or VB, we'll have to do a proper allocator.
    g_frameIndex = g_frameIndex + 1;
    FrameResources* fr = &g_pFrameResources[g_frameIndex % g_numFramesInFlight];

    // Create and grow vertex/index buffers if needed
    if (fr->VertexBuffer == NULL || fr->VertexBufferSize < draw_data->TotalVtxCount)
    {
        if (fr->VertexBuffer)
        {
            wgpuBufferDestroy(fr->VertexBuffer);
            wgpuBufferRelease(fr->VertexBuffer);
        }
        SafeRelease(fr->VertexBufferHost);
		fr->VertexBufferSize = draw_data->TotalVtxCount + 5000;

        WGPUBufferDescriptor vb_desc =
        {
            NULL,
            "Dear ImGui Vertex buffer",
            WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
            fr->VertexBufferSize * sizeof(ImDrawVert),
            false
        };
        fr->VertexBuffer = wgpuDeviceCreateBuffer(g_wgpuDevice, &vb_desc);
        if (!fr->VertexBuffer)
            return;

        fr->VertexBufferHost = new ImDrawVert[fr->VertexBufferSize];
    }
    if (fr->IndexBuffer == NULL || fr->IndexBufferSize < draw_data->TotalIdxCount)
    {
        if (fr->IndexBuffer)
        {
            wgpuBufferDestroy(fr->IndexBuffer);
            wgpuBufferRelease(fr->IndexBuffer);
        }
		SafeRelease(fr->IndexBufferHost);

		// Align to 4 bytes for buffer initialization code
        fr->IndexBufferSize = alignTo(draw_data->TotalIdxCount + 10000, 2);

        WGPUBufferDescriptor ib_desc =
        {
            NULL,
            "Dear ImGui Index buffer",
            WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
            fr->IndexBufferSize * sizeof(ImDrawIdx),
            false
        };
        fr->IndexBuffer = wgpuDeviceCreateBuffer(g_wgpuDevice, &ib_desc);
        if (!fr->IndexBuffer)
            return;

        fr->IndexBufferHost = new ImDrawIdx[fr->IndexBufferSize];
    }

    // Upload vertex/index data into a single contiguous GPU buffer
    ImDrawVert* vtx_dst = (ImDrawVert*)fr->VertexBufferHost;
    ImDrawIdx* idx_dst = (ImDrawIdx*)fr->IndexBufferHost;
    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        memcpy(vtx_dst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idx_dst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtx_dst += cmd_list->VtxBuffer.Size;
        idx_dst += cmd_list->IdxBuffer.Size;
    }
    int64_t vb_write_size = ((char*)vtx_dst - (char*)fr->VertexBufferHost + 3) & ~3;
    int64_t ib_write_size = ((char*)idx_dst - (char*)fr->IndexBufferHost  + 3) & ~3;
    wgpuQueueWriteBuffer(g_defaultQueue, fr->VertexBuffer, 0, fr->VertexBufferHost, vb_write_size);
    wgpuQueueWriteBuffer(g_defaultQueue, fr->IndexBuffer,  0, fr->IndexBufferHost,  ib_write_size);

    // Setup desired render state
    ImGui_ImplWGPU_SetupRenderState(draw_data, pass_encoder, fr);

    // Render command lists
    // (Because we merged all buffers into a single one, we maintain our own offset into them)
    int global_vtx_offset = 0;
    int global_idx_offset = 0;
    ImVec2 clip_scale = draw_data->FramebufferScale;
    ImVec2 clip_off = draw_data->DisplayPos;
    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback != NULL)
            {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                    ImGui_ImplWGPU_SetupRenderState(draw_data, pass_encoder, fr);
                else
                    pcmd->UserCallback(cmd_list, pcmd);
            }
            else
            {
                // Bind custom texture
                ImTextureID tex_id = pcmd->GetTexID();
                ImGuiID tex_id_hash = ImHashData(&tex_id, sizeof(tex_id));
                auto bind_group = g_resources.ImageBindGroups.GetVoidPtr(tex_id_hash);
                if (bind_group)
                {
                    wgpuRenderPassEncoderSetBindGroup(pass_encoder, 1, (WGPUBindGroup)bind_group, 0, NULL);
                }
                else
                {
                    WGPUBindGroup image_bind_group = ImGui_ImplWGPU_CreateImageBindGroup(g_resources.ImageBindGroupLayout, (WGPUTextureView)tex_id);
                    g_resources.ImageBindGroups.SetVoidPtr(tex_id_hash, image_bind_group);
                    wgpuRenderPassEncoderSetBindGroup(pass_encoder, 1, image_bind_group, 0, NULL);
                }

                // Project scissor/clipping rectangles into framebuffer space
                ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x, (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
                ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x, (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
                    continue;

                // Apply scissor/clipping rectangle, Draw
                wgpuRenderPassEncoderSetScissorRect(pass_encoder, (uint32_t)clip_min.x, (uint32_t)clip_min.y, (uint32_t)(clip_max.x - clip_min.x), (uint32_t)(clip_max.y - clip_min.y));
                wgpuRenderPassEncoderDrawIndexed(pass_encoder, pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset, pcmd->VtxOffset + global_vtx_offset, 0);
            }
        }
        global_idx_offset += cmd_list->IdxBuffer.Size;
        global_vtx_offset += cmd_list->VtxBuffer.Size;
    }
}

static void ImGui_ImplWGPU_CreateFontsTexture()
{
    // Build texture atlas
    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels;
    int width, height, size_pp;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height, &size_pp);

    // Upload texture to graphics system
    {
        WGPUTextureDescriptor tex_desc = {};
        tex_desc.label = "Dear ImGui Font Texture";
        tex_desc.dimension = WGPUTextureDimension_2D;
        tex_desc.size.width = width;
        tex_desc.size.height = height;
        tex_desc.size.depthOrArrayLayers = 1;
        tex_desc.sampleCount = 1;
        tex_desc.format = WGPUTextureFormat_RGBA8Unorm;
        tex_desc.mipLevelCount = 1;
        tex_desc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
        g_resources.FontTexture = wgpuDeviceCreateTexture(g_wgpuDevice, &tex_desc);

        WGPUTextureViewDescriptor tex_view_desc = {};
        tex_view_desc.format = WGPUTextureFormat_RGBA8Unorm;
        tex_view_desc.dimension = WGPUTextureViewDimension_2D;
        tex_view_desc.baseMipLevel = 0;
        tex_view_desc.mipLevelCount = 1;
        tex_view_desc.baseArrayLayer = 0;
        tex_view_desc.arrayLayerCount = 1;
        tex_view_desc.aspect = WGPUTextureAspect_All;
        g_resources.FontTextureView = wgpuTextureCreateView(g_resources.FontTexture, &tex_view_desc);
    }

    // Upload texture data
    {
        WGPUImageCopyTexture dst_view = {};
        dst_view.texture = g_resources.FontTexture;
        dst_view.mipLevel = 0;
        dst_view.origin = { 0, 0, 0 };
        dst_view.aspect = WGPUTextureAspect_All;
        WGPUTextureDataLayout layout = {};
        layout.offset = 0;
        layout.bytesPerRow = width * size_pp;
        layout.rowsPerImage = height;
        WGPUExtent3D size = { (uint32_t)width, (uint32_t)height, 1 };
        wgpuQueueWriteTexture(g_defaultQueue, &dst_view, pixels, (uint32_t)(width * size_pp * height), &layout, &size);
    }

    // Create the associated sampler
    {
        WGPUSamplerDescriptor sampler_desc = {};
        sampler_desc.minFilter = WGPUFilterMode_Linear;
        sampler_desc.magFilter = WGPUFilterMode_Linear;
        sampler_desc.mipmapFilter = WGPUFilterMode_Linear;
        sampler_desc.addressModeU = WGPUAddressMode_Repeat;
        sampler_desc.addressModeV = WGPUAddressMode_Repeat;
        sampler_desc.addressModeW = WGPUAddressMode_Repeat;
        sampler_desc.maxAnisotropy = 1;
        g_resources.Sampler = wgpuDeviceCreateSampler(g_wgpuDevice, &sampler_desc);
    }

    // Store our identifier
    static_assert(sizeof(ImTextureID) >= sizeof(g_resources.FontTexture), "Can't pack descriptor handle into TexID, 32-bit not supported yet.");
    io.Fonts->SetTexID((ImTextureID)g_resources.FontTextureView);
}

static void ImGui_ImplWGPU_CreateUniformBuffer()
{
    WGPUBufferDescriptor ub_desc =
    {
        NULL,
        "Dear ImGui Uniform buffer",
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
        sizeof(Uniforms),
        false
    };
    g_resources.Uniforms = wgpuDeviceCreateBuffer(g_wgpuDevice, &ub_desc);
}

bool ImGui_ImplWGPU_CreateDeviceObjects()
{
    if (!g_wgpuDevice)
        return false;
    if (g_pipelineState)
        ImGui_ImplWGPU_InvalidateDeviceObjects();

    WGPUBindGroupLayoutEntry layout_entries[3] = {};
    // block 0
	layout_entries[0].binding = 0; // transform
    layout_entries[0].visibility = WGPUShaderStage_Vertex;
    layout_entries[0].buffer.type = WGPUBufferBindingType_Uniform;
	layout_entries[0].buffer.minBindingSize = sizeof(float) * 16;
	layout_entries[1].binding = 1; // sampler
    layout_entries[1].visibility = WGPUShaderStage_Fragment;
	layout_entries[1].sampler.type = WGPUSamplerBindingType_Filtering;
    // block 1
	layout_entries[2].binding = 0; // texture
    layout_entries[2].visibility = WGPUShaderStage_Fragment;
    layout_entries[2].texture.sampleType = WGPUTextureSampleType_Float;
	layout_entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor block0_layout_desc{};
    block0_layout_desc.entries = layout_entries;
    block0_layout_desc.entryCount = 2;

    WGPUBindGroupLayoutDescriptor block1_layout_desc{};
    block1_layout_desc.entries = &layout_entries[2];
    block1_layout_desc.entryCount = 1;

    WGPUBindGroupLayout bind_group_layouts[2] = {
        wgpuDeviceCreateBindGroupLayout(g_wgpuDevice, &block0_layout_desc),
        wgpuDeviceCreateBindGroupLayout(g_wgpuDevice, &block1_layout_desc),
    };

    WGPUPipelineLayoutDescriptor pipeline_layout_desc{};
    pipeline_layout_desc.bindGroupLayouts = bind_group_layouts;
    pipeline_layout_desc.bindGroupLayoutCount = 2;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(g_wgpuDevice, &pipeline_layout_desc);

    // Create render pipeline
    WGPURenderPipelineDescriptor graphics_pipeline_desc = {};
    graphics_pipeline_desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    graphics_pipeline_desc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    graphics_pipeline_desc.primitive.frontFace = WGPUFrontFace_CW;
    graphics_pipeline_desc.primitive.cullMode = WGPUCullMode_None;
    graphics_pipeline_desc.multisample.count = 1;
    graphics_pipeline_desc.multisample.mask = UINT_MAX;
    graphics_pipeline_desc.multisample.alphaToCoverageEnabled = false;
    graphics_pipeline_desc.layout = pipeline_layout;

    // Compile the shader module
    WGPUShaderModuleDescriptor shaderModuleDesc{};
    WGPUShaderModuleWGSLDescriptor shaderModuleWgslDesc{};
    shaderModuleWgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    shaderModuleDesc.nextInChain = &shaderModuleWgslDesc.chain;
    wgpuShaderModuleWGSLDescriptorSetCode(shaderModuleWgslDesc, __wgsl_shader);
    WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(g_wgpuDevice, &shaderModuleDesc);

    graphics_pipeline_desc.vertex.module = shaderModule;
    graphics_pipeline_desc.vertex.entryPoint = "vs_main";

    // Vertex input configuration
    WGPUVertexAttribute attribute_desc[] =
    {
        { WGPUVertexFormat_Float32x2, (uint64_t)IM_OFFSETOF(ImDrawVert, pos), 0 },
        { WGPUVertexFormat_Float32x2, (uint64_t)IM_OFFSETOF(ImDrawVert, uv),  1 },
        { WGPUVertexFormat_Unorm8x4,  (uint64_t)IM_OFFSETOF(ImDrawVert, col), 2 },
    };

    WGPUVertexBufferLayout buffer_layouts[1];
    buffer_layouts[0].arrayStride = sizeof(ImDrawVert);
    buffer_layouts[0].stepMode = WGPUVertexStepMode_Vertex;
    buffer_layouts[0].attributeCount = 3;
    buffer_layouts[0].attributes = attribute_desc;

    graphics_pipeline_desc.vertex.bufferCount = 1;
    graphics_pipeline_desc.vertex.buffers = buffer_layouts;

    // Create the blending setup
    WGPUBlendState blend_state = {};
    blend_state.alpha.operation = WGPUBlendOperation_Add;
    blend_state.alpha.srcFactor = WGPUBlendFactor_One;
    blend_state.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend_state.color.operation = WGPUBlendOperation_Add;
    blend_state.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_state.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;

    WGPUColorTargetState color_state = {};
    color_state.format = g_renderTargetFormat;
    color_state.blend = &blend_state;
    color_state.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragment_state = {};
    fragment_state.module = shaderModule;
    fragment_state.entryPoint = "fs_main";
    fragment_state.targetCount = 1;
    fragment_state.targets = &color_state;

    graphics_pipeline_desc.fragment = &fragment_state;

    // Create depth-stencil State
    WGPUDepthStencilState depth_stencil_state = {};
    depth_stencil_state.depthBias = 0;
    depth_stencil_state.depthBiasClamp = 0;
    depth_stencil_state.depthBiasSlopeScale = 0;

    // Configure disabled depth-stencil state
    graphics_pipeline_desc.depthStencil = nullptr;

    g_pipelineState = wgpuDeviceCreateRenderPipeline(g_wgpuDevice, &graphics_pipeline_desc);

    ImGui_ImplWGPU_CreateFontsTexture();
    ImGui_ImplWGPU_CreateUniformBuffer();

    WGPUBindGroupEntry common_bg_entries[] =
    {
        { nullptr, 0, g_resources.Uniforms, 0, sizeof(Uniforms), 0, 0 },
        { nullptr, 1, 0, 0, 0, g_resources.Sampler, 0 },
    };

    WGPUBindGroupDescriptor common_bg_descriptor = {};
    common_bg_descriptor.layout = bind_group_layouts[0];
    common_bg_descriptor.entryCount = sizeof(common_bg_entries) / sizeof(WGPUBindGroupEntry);
    common_bg_descriptor.entries = common_bg_entries;
    g_resources.CommonBindGroup = wgpuDeviceCreateBindGroup(g_wgpuDevice, &common_bg_descriptor);

    WGPUBindGroup image_bind_group = ImGui_ImplWGPU_CreateImageBindGroup(bind_group_layouts[1], g_resources.FontTextureView);
    g_resources.ImageBindGroup = image_bind_group;
    g_resources.ImageBindGroupLayout = bind_group_layouts[1];
    g_resources.ImageBindGroups.SetVoidPtr(ImHashData(&g_resources.FontTextureView, sizeof(ImTextureID)), image_bind_group);

    SafeRelease(shaderModule);
    SafeRelease(pipeline_layout);
    SafeRelease(bind_group_layouts[0]);
    SafeRelease(bind_group_layouts[1]);

    return true;
}

void ImGui_ImplWGPU_InvalidateDeviceObjects()
{
    if (!g_wgpuDevice)
        return;

    SafeRelease(g_pipelineState);
    SafeRelease(g_resources);

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->SetTexID(NULL); // We copied g_pFontTextureView to io.Fonts->TexID so let's clear that as well.

    for (unsigned int i = 0; i < g_numFramesInFlight; i++)
        SafeRelease(g_pFrameResources[i]);
}

bool ImGui_ImplWGPU_Init(WGPUDevice device, int num_frames_in_flight, WGPUTextureFormat rt_format)
{
    // Setup backend capabilities flags
    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "imgui_impl_webgpu";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;  // We can honor the ImDrawCmd::VtxOffset field, allowing for large meshes.

    g_wgpuDevice = device;
    g_defaultQueue = wgpuDeviceGetQueue(g_wgpuDevice);
    g_renderTargetFormat = rt_format;
    g_pFrameResources = new FrameResources[num_frames_in_flight];
    g_numFramesInFlight = num_frames_in_flight;
    g_frameIndex = UINT_MAX;

    g_resources.FontTexture = NULL;
    g_resources.FontTextureView = NULL;
    g_resources.Sampler = NULL;
    g_resources.Uniforms = NULL;
    g_resources.CommonBindGroup = NULL;
    g_resources.ImageBindGroups.Data.reserve(100);
    g_resources.ImageBindGroup = NULL;
    g_resources.ImageBindGroupLayout = NULL;

    // Create buffers with a default size (they will later be grown as needed)
    for (int i = 0; i < num_frames_in_flight; i++)
    {
        FrameResources* fr = &g_pFrameResources[i];
        fr->IndexBuffer = NULL;
        fr->VertexBuffer = NULL;
        fr->IndexBufferHost = NULL;
        fr->VertexBufferHost = NULL;
        fr->IndexBufferSize = 10000;
        fr->VertexBufferSize = 5000;
    }

    return true;
}

void ImGui_ImplWGPU_Shutdown()
{
    ImGui_ImplWGPU_InvalidateDeviceObjects();
    delete[] g_pFrameResources;
    g_pFrameResources = NULL;
    wgpuQueueRelease(g_defaultQueue);
    g_wgpuDevice = NULL;
    g_numFramesInFlight = 0;
    g_frameIndex = UINT_MAX;
}

void ImGui_ImplWGPU_NewFrame()
{
    if (!g_pipelineState)
        ImGui_ImplWGPU_CreateDeviceObjects();
}
