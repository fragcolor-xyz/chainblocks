#ifndef AEEB0564_D448_4541_8B2E_2AC321866795
#define AEEB0564_D448_4541_8B2E_2AC321866795
#ifndef CDDE58F0_98A7_4185_87DD_DF680AB4C011
#define CDDE58F0_98A7_4185_87DD_DF680AB4C011

#include "../linalg.hpp"
#include "../drawables/mesh_tree_drawable.hpp"
#include "animation.hpp"
#include <functional>
#include <optional>
#include <unordered_map>

namespace gfx {
struct glTF2 {
  MeshTreeDrawable::Ptr root;
  std::unordered_map<std::string, Animation> animations;
  std::unordered_map<std::string, MaterialPtr> materials;

  glTF2() = default;

  glTF2(glTF2 &&) = default;
  glTF2 &operator=(glTF2 &&) = default;

  glTF2(const glTF2 &) = delete;
  glTF2 &operator=(const glTF2 &) = delete;
};

// glTF2 loadGltfFromFile2(std::string_view file);
// glTF2 loadGltfFromMemory2(const uint8_t *data, size_t dataLength);
// bool isBinary2(uint8_t (&peekBuffer)[4]);
// std::vector<uint8_t> convertToGlb2(const std::string &inputPath);
} // namespace gfx

#endif /* CDDE58F0_98A7_4185_87DD_DF680AB4C011 */


#endif /* AEEB0564_D448_4541_8B2E_2AC321866795 */
