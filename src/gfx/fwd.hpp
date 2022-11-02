#ifndef GFX_FWD
#define GFX_FWD

#include <memory>

namespace gfx {

struct Context;
struct Pipeline;

struct DrawQueue;
typedef std::shared_ptr<DrawQueue> DrawQueuePtr;

struct Drawable;
typedef std::shared_ptr<Drawable> DrawablePtr;

struct DrawableHierarchy;
typedef std::shared_ptr<DrawableHierarchy> DrawableHierarchyPtr;

struct Mesh;
typedef std::shared_ptr<Mesh> MeshPtr;

struct Feature;
typedef std::shared_ptr<Feature> FeaturePtr;

struct IPipelineModifier;
typedef std::shared_ptr<IPipelineModifier> PipelineModifierPtr;

struct View;
typedef std::shared_ptr<View> ViewPtr;

struct Material;
typedef std::shared_ptr<Material> MaterialPtr;

struct Texture;
typedef std::shared_ptr<Texture> TexturePtr;

struct RenderTargetAttachment;
typedef std::shared_ptr<RenderTargetAttachment> RenderTargetAttachmentPtr;

struct RenderTarget;
typedef std::shared_ptr<RenderTarget> RenderTargetPtr;

} // namespace gfx

#endif // GFX_FWD
