#include "drawables/mesh_drawable.hpp"
#include "drawable_processors/mesh_drawable_processor.hpp"
#include "../renderer_cache.hpp"
#include "../mesh.hpp"
#include "../feature.hpp"
#include "../material.hpp"
#include "bounds.hpp"
#include <cassert>
#include <set>

namespace gfx {

DrawablePtr MeshDrawable::clone() const { return cloneSelfWithId(this, getNextDrawableId()); }
DrawableProcessorConstructor MeshDrawable::getProcessor() const {
  static DrawableProcessorConstructor fn = [](Context &context) -> detail::DrawableProcessorPtr {
    return std::make_shared<detail::MeshDrawableProcessor>(context);
  };
  return fn;
}
OrientedBounds MeshDrawable::getBounds() const {
  if (!mesh)
    return OrientedBounds::empty();

  const MeshDescCPUCopy *descPtr = std::get_if<MeshDescCPUCopy>(&mesh->getDesc());
  if (!descPtr)
    throw std::runtime_error("MeshDesc is not a MeshDescCPUCopy");
  const MeshDescCPUCopy &meshDesc = *descPtr;

  if (skin) {
    // AABounds bounds = AABounds::empty();
    // for (auto &joint : skin->jointMatrices) {
    //   bounds.expand(extractTranslation(joint));
    // }
    AABounds meshBounds = OrientedBounds(getMeshBounds(meshDesc), transform).toAligned();
    AABounds bounds = AABounds::both(meshBounds, skin->bounds);
    return OrientedBounds(bounds, linalg::identity).expand(float3(0.5f));
  }

  return OrientedBounds(getMeshBounds(meshDesc), transform);
}

} // namespace gfx
