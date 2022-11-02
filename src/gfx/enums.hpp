#ifndef GFX_ENUMS
#define GFX_ENUMS

#include "gfx_wgpu.hpp"
#include <string_view>

namespace gfx {

enum class PrimitiveType { TriangleList, TriangleStrip };
enum class WindingOrder { CW, CCW };
enum class ProgrammableGraphicsStage { Vertex, Fragment };
enum class StorageType { UInt8, Int8, UNorm8, SNorm8, UInt16, Int16, UNorm16, SNorm16, UInt32, Int32, Float16, Float32 };
enum class IndexFormat { UInt16, UInt32 };

struct TextureFormatDesc {
  // What format components are stored in
  StorageType storageType;

  // Number of components
  size_t numComponents;

  // Number of bytes per pixel
  uint8_t pixelSize;

  TextureFormatDesc(StorageType storageType, size_t numComponents);
};

// TextureFormat
const TextureFormatDesc &getTextureFormatDescription(WGPUTextureFormat pixelFormat);
size_t getStorageTypeSize(const StorageType &type);
size_t getIndexFormatSize(const IndexFormat &type);
WGPUVertexFormat getWGPUVertexFormat(const StorageType &type, size_t dim);
WGPUIndexFormat getWGPUIndexFormat(const IndexFormat &type);

} // namespace gfx

#endif // GFX_ENUMS
