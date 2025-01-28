#ifndef CDDE58F0_98A7_4185_87DD_DF680AB4C011
#define CDDE58F0_98A7_4185_87DD_DF680AB4C011

#include "../linalg.hpp"
#include "../drawables/mesh_tree_drawable.hpp"
#include "animation.hpp"
#include <functional>
#include <optional>
#include <unordered_map>

namespace gfx {
struct glTF {
  MeshTreeDrawable::Ptr root;
  std::unordered_map<std::string, Animation> animations;
  std::unordered_map<std::string, MaterialPtr> materials;
  linalg::vec<float, 3> boundsMin = {0, 0, 0};
  linalg::vec<float, 3> boundsMax = {0, 0, 0};

  glTF() = default;

  glTF(glTF &&) = default;
  glTF &operator=(glTF &&) = default;

  glTF(const glTF &) = delete;
  glTF &operator=(const glTF &) = delete;
};

glTF loadGltfFromFile(std::string_view file);
glTF loadGltfFromMemory(const uint8_t *data, size_t dataLength);
bool isBinary(uint8_t (&peekBuffer)[4]);

std::vector<uint8_t> convertToGlb(const std::string &inputPath);
} // namespace gfx

#endif /* CDDE58F0_98A7_4185_87DD_DF680AB4C011 */
