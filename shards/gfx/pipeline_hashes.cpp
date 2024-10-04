#include "material.hpp"
#include "drawables/mesh_drawable.hpp"
#include "feature.hpp"
#include "mesh.hpp"

// This file groups all the pipeline hash functions so that they can be compiled -O3 even in debug

using gfx::detail::PipelineHashCollector;
namespace gfx {
void MaterialParameters::pipelineHashCollect(PipelineHashCollector &pipelineHashCollector) const {
  for (auto &pair : textures) {
    pipelineHashCollector(pair.first);
    if (pair.second.texture) {
      pipelineHashCollector(1u);
      pipelineHashCollector(pair.second.defaultTexcoordBinding);
    } else {
      pipelineHashCollector(0u);
    }
  }
}

void Material::pipelineHashCollect(PipelineHashCollector &pipelineHashCollector) const {
  for (auto &feature : features) {
    pipelineHashCollector(feature);
  }
  parameters.pipelineHashCollect(pipelineHashCollector);
}

void Feature::pipelineHashCollect(PipelineHashCollector &pipelineHashCollector) const { pipelineHashCollector(id); }

void Mesh::pipelineHashCollect(PipelineHashCollector &pipelineHashCollector) const { pipelineHashCollector(getFormat()); }

} // namespace gfx