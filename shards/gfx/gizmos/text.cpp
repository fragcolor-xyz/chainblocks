
#define STB_TRUETYPE_IMPLEMENTATION
#define STB_RECT_PACK_IMPLEMENTATION
#include <stb_rect_pack.h>
#include <stb_truetype.h>

#include "text.hpp"
#include "../isb.hpp"
#include "../texture.hpp"
#include "../../shards/core/assert.hpp"
#include <bundled/BigBlue_TerminalPlus.bin.h>

#include <brotli/decode.h>

namespace gfx::gizmos {

struct FontMapEntry {
  char c;
  stbtt_packedchar charData;
};

struct FontMapImpl {
  std::vector<stbtt_packedchar> charDataForRange;
  stbtt_pack_range mainRange;
};

FontMap::FontMap() { impl = new FontMapImpl(); }
FontMap::~FontMap() { delete impl; }

static FontMap::Ptr loadDefaultFontmap() {
  auto result = std::make_shared<FontMap>();
  auto &impl = *result->impl;

  stbtt_pack_range mainRange{};
  mainRange.first_unicode_codepoint_in_range = 0x0;
  mainRange.array_of_unicode_codepoints = NULL;
  mainRange.num_chars = 0x7E - mainRange.first_unicode_codepoint_in_range;

  impl.charDataForRange.resize(mainRange.num_chars);
  mainRange.chardata_for_range = impl.charDataForRange.data();
  mainRange.font_size = 12.0f;

  std::vector<uint8_t> decodedBuffer;
  size_t decodedSize = *(uint32_t *)bundled_BigBlue_TerminalPlusΘbin_getData();
  decodedBuffer.resize(decodedSize);
  auto bres = BrotliDecoderDecompress(bundled_BigBlue_TerminalPlusΘbin_getLength() - 4,
                                      bundled_BigBlue_TerminalPlusΘbin_getData() + 4, &decodedSize, decodedBuffer.data());
  shassert(bres == BROTLI_DECODER_RESULT_SUCCESS);

  // SDL_Surface *fontMapSurf = SDL_CreateSurface(256, 256, SDL_PIXELFORMAT_ARGB32);
  TexturePtr fontMapTexture = std::make_shared<Texture>();
  TextureDescCPUCopy textureDesc{
      .format =
          TextureFormat{
              .resolution = int2(256, 256),
              .pixelFormat = WGPUTextureFormat_RGBA8Unorm,
          },
  };
  int2 &res = textureDesc.format.resolution;

  stbtt_pack_context pctx{};

  std::vector<uint8_t> singleChanMap;
  singleChanMap.resize(res.x * res.y);

  stbtt_PackBegin(&pctx, singleChanMap.data(), res.x, res.y, res.x, 0, nullptr);
  stbtt_PackFontRanges(&pctx, decodedBuffer.data(), 0, &mainRange, 1);
  stbtt_PackEnd(&pctx);

  std::vector<uint8_t> imageData;
  imageData.resize(res.x * res.y * sizeof(uint32_t));
  uint32_t imageDataRowStride = res.x * sizeof(uint32_t);

  for (size_t y = 0; y < res.y; ++y) {
    for (size_t x = 0; x < res.x; ++x) {
      uint8_t *src = &singleChanMap[x + y * res.x];
      uint32_t *dst = (uint32_t *)((uint8_t *)imageData.data() + (y * imageDataRowStride) + x * sizeof(uint32_t));
      *dst = (uint32_t(*src) << 24) | 0x00FFFFFF;
    }
  }

  // Determine spacing
  for (size_t i = 0; i < mainRange.num_chars; ++i) {
    if (mainRange.chardata_for_range[i].xadvance != 0) {
      result->spaceSize.x = mainRange.chardata_for_range[i].xadvance;
      break;
    }
  }
  for (size_t i = 0; i < mainRange.num_chars; ++i) {
    if (mainRange.chardata_for_range[i].yoff2 != 0) {
      result->spaceSize.y = mainRange.chardata_for_range[i].yoff2 - mainRange.chardata_for_range[i].yoff;
      break;
    }
  }

  textureDesc.sourceData = ImmutableSharedBuffer(std::move(imageData));
  textureDesc.sourceChannels = 4;
  textureDesc.sourceRowStride = imageDataRowStride;
  
  fontMapTexture->init(textureDesc)
      .initWithSamplerState(SamplerState{
          .addressModeU = WGPUAddressMode_ClampToEdge,
          .addressModeV = WGPUAddressMode_ClampToEdge,
          .addressModeW = WGPUAddressMode_ClampToEdge,
          .filterMode = WGPUFilterMode_Nearest,
      });

  result->image = fontMapTexture;
  result->impl->mainRange = mainRange;
  return result;
}

FontMap::Ptr FontMap::getDefault() {
  static FontMap::Ptr instance = loadDefaultFontmap();
  return instance;
}

void stbtt_GetPackedQuadScaled(const stbtt_packedchar *chardata, int pw, int ph, int char_index, float scale, float *xpos,
                               float *ypos, stbtt_aligned_quad *q, int align_to_integer) {
  float ipw = 1.0f / pw, iph = 1.0f / ph;
  const stbtt_packedchar *b = chardata + char_index;

  if (align_to_integer) {
    float x = (float)STBTT_ifloor((*xpos + b->xoff * scale) + 0.5f);
    float y = (float)STBTT_ifloor((*ypos + b->yoff * scale) + 0.5f);
    q->x0 = x;
    q->y0 = y;
    q->x1 = x + (b->xoff2 - b->xoff) * scale;
    q->y1 = y + (b->yoff2 - b->yoff) * scale;
  } else {
    q->x0 = *xpos + b->xoff * scale;
    q->y0 = *ypos + b->yoff * scale;
    q->x1 = *xpos + b->xoff2 * scale;
    q->y1 = *ypos + b->yoff2 * scale;
  }

  q->s0 = b->x0 * ipw;
  q->t0 = b->y0 * iph;
  q->s1 = b->x1 * ipw;
  q->t1 = b->y1 * iph;

  *xpos += b->xadvance * scale;
}

void TextPlacer::appendChar(FontMap::Ptr fontMap, uint32_t c, float scale) {
  auto &impl = *fontMap->impl;
  if (c == U' ') {
    pos.x += fontMap->spaceSize.x * scale;
  } else if (c == U'\n') {
    ++numLines;
    pos.x = origin.x;
    pos.y += fontMap->spaceSize.y * scale;
  } else {
    int charIdx = c - impl.mainRange.first_unicode_codepoint_in_range;
    if (charIdx > impl.mainRange.num_chars)
      return;

    int2 res = fontMap->image->getResolution();

    float2 tmpPos = pos;

    stbtt_aligned_quad quadRef;
    stbtt_GetPackedQuad(impl.charDataForRange.data(), res.x, res.y, charIdx, &tmpPos.x, &tmpPos.y, &quadRef, 0);

    tmpPos = pos;
    stbtt_aligned_quad quad;
    stbtt_GetPackedQuadScaled(impl.charDataForRange.data(), res.x, res.y, charIdx, scale, &tmpPos.x, &tmpPos.y, &quad, 0);
    float2 posDelta = tmpPos - pos;
    pos += posDelta;

    TextQuad tq{
        .quad = float4{quad.x0, quad.y0, quad.x1, quad.y1},
        .uv = float4{quad.s0, quad.t0, quad.s1, quad.t1},
    };
    textQuads.push_back(tq);
  }

  max.x = std::max(max.x, pos.x);
  max.y = std::max(max.y, pos.y);
}

void TextPlacer::appendString(FontMap::Ptr fontMap, std::string_view text, float scale) {
  for (auto c : text) {
    appendChar(fontMap, c, scale);
  }
}

} // namespace gfx::gizmos