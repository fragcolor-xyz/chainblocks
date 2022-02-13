#pragma once

namespace gfx {

struct Context;
struct Pipeline;
struct DrawQueue;

struct Drawable;
typedef std::shared_ptr<Drawable> DrawablePtr;

struct Mesh;
typedef std::shared_ptr<Mesh> MeshPtr;

struct Feature;
typedef std::shared_ptr<Feature> FeaturePtr;

struct View;
typedef std::shared_ptr<View> ViewPtr;

struct Material;
typedef std::shared_ptr<Material> MaterialPtr;

struct Texture;
typedef std::shared_ptr<Texture> TexturePtr;

} // namespace gfx
