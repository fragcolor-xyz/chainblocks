/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

#include "imaging.hpp"
#include <shards/modules/core/file_base.hpp>
#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>
#include <fstream>

#include "linalg.h"
using namespace linalg::aliases;

// Implemented externally
// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_RESIZE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_resize.h>
#include <stb_image.h>
#include <stb_image_write.h>

// Thanks windows for polluting the global namespace
#ifdef LoadImage
#undef LoadImage
#endif

namespace shards {
namespace Imaging {
struct Convolve {
  static SHOptionalString help() {
    return SHCCSTR("This shard extracts a patch from an image of dimensions specified in the Radius parameter and outputs the "
                   "extracted patch as an image. Everytime the shard is subsequently called, the centre of the patch to extract "
                   "is advanced by the number of pixels specified in the Step parameter.");
  }
  static SHOptionalString inputHelp() { return SHCCSTR("The image to extract patches from."); }
  static SHOptionalString outputHelp() { return SHCCSTR("The extracted patch as an image."); }
  static SHTypesInfo inputTypes() { return CoreInfo::ImageType; }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }

  static inline Parameters _params{
      {"Radius",
       SHCCSTR("The radius of the kernel, e.g. 1 = 1x1; 2 = 3x3; 3 = 5x5 and "
               "so on."),
       {CoreInfo::IntType}},
      {"Step", SHCCSTR("How many pixels to advance the centre of the patch extracted each activation."), {CoreInfo::IntType}}};

  static SHParametersInfo parameters() { return _params; }

  OwnedVar _output{};

  SHVar getParam(int index) {
    if (index == 0)
      return Var(int64_t(_radius));
    else
      return Var(int64_t(_step));
  }

  void setParam(int index, const SHVar &value) {
    if (index == 0) {
      _radius = int32_t(value.payload.intValue);
      if (_radius <= 0)
        _radius = 1;
      _kernel = (_radius - 1) * 2 + 1;
    } else {
      _step = int32_t(value.payload.intValue);
      if (_step <= 0)
        _step = 1;
    }
  }

  void warmup(SHContext *context) {}

  void cleanup(SHContext *context) {
    _xindex = 0;
    _yindex = 0;
  }

  template <typename T>
  void process(uint8_t *outData, const SHVar &pixels, int32_t w, int32_t h, int32_t inRowStride, int32_t c) {
    const int high = _radius - 1;
    const int low = high * -1;
    int index = 0;
    auto to = reinterpret_cast<T *>(outData);
    for (int y = low; y <= high; y++) {
      const int cidxy = _yindex + y;
      const auto idxy = std::clamp<int>(cidxy, 0, h - 1);
      const auto from = reinterpret_cast<T *>(pixels.payload.imageValue->data + inRowStride * idxy);
      for (int x = low; x <= high; x++) {
        const int cidxx = _xindex + x;
        const auto idxx = std::clamp<int>(cidxx, 0, w - 1);
        const int addr = idxx * c;
        for (int i = 0; i < c; i++) {
          to[index++] = from[addr + i];
        }
      }
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    int32_t w = int32_t(input.payload.imageValue->width);
    int32_t h = int32_t(input.payload.imageValue->height);
    int32_t c = int32_t(input.payload.imageValue->channels);

    auto pixsize = imageGetPixelSize(input.payload.imageValue);
    _output = makeImage(_kernel * _kernel * c * pixsize);
    SHImage &outImage = *_output->payload.imageValue;
    outImage.width = _kernel;
    outImage.height = _kernel;
    outImage.channels = input.payload.imageValue->channels;
    outImage.flags = input.payload.imageValue->flags;

    uint8_t *outData = outImage.data;

    if (_xindex >= w) {
      _xindex = 0;
      _yindex++;
      if (_yindex >= h) {
        _yindex = 0;
      }
    }

    if (pixsize == 1) {
      process<uint8_t>(outData, input, w, h, imageGetRowStride(input.payload.imageValue), c);
    } else if (pixsize == 2) {
      process<uint16_t>(outData, input, w, h, imageGetRowStride(input.payload.imageValue), c);
    } else if (pixsize == 4) {
      process<float>(outData, input, w, h, imageGetRowStride(input.payload.imageValue), c);
    }

    // advance the scan
    _xindex += _step;

    return _output;
  }

private:
  int32_t _radius{1};
  int32_t _step{1};
  uint32_t _kernel{1};
  int32_t _xindex{0};
  int32_t _yindex{0};
};

struct StripAlpha {
  static SHOptionalString help() {
    return SHCCSTR("This shard takes an image with 4 channels(RGBA) and converts it to a 3-channel(RGB) image with the alpha "
                   "channel removed.");
  }
  static SHOptionalString inputHelp() { return SHCCSTR("The image to remove the alpha channel from."); }
  static SHOptionalString outputHelp() { return SHCCSTR("The image with the alpha channel removed."); }

  static SHTypesInfo inputTypes() { return CoreInfo::ImageType; }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }

  OwnedVar _output{};

  template <typename T> void process(uint8_t *outData, const SHVar &input, int32_t w, int32_t h, int32_t inRowStride) {
    auto to = reinterpret_cast<T *>(outData);
    for (auto y = 0; y < h; y++) {
      const auto from = reinterpret_cast<T *>(input.payload.imageValue->data + inRowStride * y);
      for (auto x = 0; x < w; x++) {
        const auto faddr = x * 4;
        const auto taddr = ((w * y) + x) * 3;
        for (auto z = 0; z < 3; z++) {
          to[taddr + z] = from[faddr + z];
        }
      }
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (input.payload.imageValue->channels < 4)
      return input; // nothing to do

    int32_t w = int32_t(input.payload.imageValue->width);
    int32_t h = int32_t(input.payload.imageValue->height);

    auto pixsize = 1;
    if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_16BITS_INT) == SHIMAGE_FLAGS_16BITS_INT)
      pixsize = 2;
    else if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_32BITS_FLOAT) == SHIMAGE_FLAGS_32BITS_FLOAT)
      pixsize = 4;

    _output.reset();
    _output = makeImage(w * h * 3 * pixsize);
    auto &outImage = *_output.payload.imageValue;
    outImage.channels = 3;
    outImage.width = uint16_t(w);
    outImage.height = uint16_t(h);
    uint8_t *outData = outImage.data;

    if (pixsize == 1) {
      process<uint8_t>(outData, input, w, h, imageGetRowStride(input.payload.imageValue));
    } else if (pixsize == 2) {
      process<uint16_t>(outData, input, w, h, imageGetRowStride(input.payload.imageValue));
    } else if (pixsize == 4) {
      process<float>(outData, input, w, h, imageGetRowStride(input.payload.imageValue));
    }

    _output.version = input.version + 1;
    return _output;
  }
};

struct PremultiplyAlpha {
  static SHOptionalString help() {
    return SHCCSTR(
        R"(Applies the premultiplication of alpha channels of an image to its RGB channels. Does nothing if the image has already been premultiplied in Shards. This mainly applies to PNG images.)");
  }
  static SHTypesInfo inputTypes() { return CoreInfo::ImageType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The image to apply the premultiplication of alpha channels to."); }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The image as a result of the application of the premultiplication of alpha channels.");
  }

  OwnedVar _output{};

  template <typename T> void process(const SHVar &input, SHVar &output, int32_t w, int32_t h, int32_t rowStride) {
    premultiplyAlpha<T>(input, output, w, h, rowStride);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (input.payload.imageValue->channels < 4)
      return input; // nothing to do

    if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA) == SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA)
      return input; // already premultiplied

    // find number of bytes per pixel
    auto pixsize = 1;
    if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_16BITS_INT) == SHIMAGE_FLAGS_16BITS_INT)
      pixsize = 2;
    else if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_32BITS_FLOAT) == SHIMAGE_FLAGS_32BITS_FLOAT)
      pixsize = 4;

    int32_t w = int32_t(input.payload.imageValue->width);
    int32_t h = int32_t(input.payload.imageValue->height);

    _bytes.resize(w * h * 4 * pixsize);
    _output = makeImage(w * h * 4 * pixsize);
    auto &outImage = *_output->payload.imageValue;
    outImage.channels = 4;
    outImage.width = uint16_t(w);
    outImage.height = uint16_t(h);
    outImage.flags = input.payload.imageValue->flags;

    if (pixsize == 1) {
      process<uint8_t>(input, _output, w, h, imageGetRowStride(input.payload.imageValue));
    } else if (pixsize == 2) {
      process<uint16_t>(input, _output, w, h, imageGetRowStride(input.payload.imageValue));
    } else if (pixsize == 4) {
      process<float>(input, _output, w, h, imageGetRowStride(input.payload.imageValue));
    }

    return _output;
  }

private:
  std::vector<uint8_t> _bytes;
};

struct DemultiplyAlpha {
  static SHOptionalString help() {
    return SHCCSTR(
        R"(Applies the demultiplication of alpha channels of an image to its RGB channels. Does nothing if the image has already been demultiplied or never been premultiplied in Shards. This mainly applies to PNG images.)");
  }
  static SHTypesInfo inputTypes() { return CoreInfo::ImageType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The image to apply the demultiplication of alpha channels to."); }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The image as a result of the application of the demultiplication of alpha channels.");
  }

  OwnedVar _output{};

  template <typename T> void process(const SHVar &input, SHVar &output, int32_t w, int32_t h, int32_t rowStride) {
    demultiplyAlpha<T>(input, output, w, h, rowStride);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (input.payload.imageValue->channels < 4)
      return input; // nothing to do

    if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA) != SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA)
      return input; // already straight alpha

    // find number of bytes per pixel
    auto pixsize = 1;
    if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_16BITS_INT) == SHIMAGE_FLAGS_16BITS_INT)
      pixsize = 2;
    else if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_32BITS_FLOAT) == SHIMAGE_FLAGS_32BITS_FLOAT)
      pixsize = 4;

    int32_t w = int32_t(input.payload.imageValue->width);
    int32_t h = int32_t(input.payload.imageValue->height);

    _output = makeImage(w * h * 4 * pixsize);
    auto &outImage = *_output->payload.imageValue;
    outImage.channels = 4;
    outImage.width = uint16_t(w);
    outImage.height = uint16_t(h);
    outImage.flags = input.payload.imageValue->flags;

    if (pixsize == 1) {
      process<uint8_t>(input, _output, w, h, imageGetRowStride(input.payload.imageValue));
    } else if (pixsize == 2) {
      process<uint16_t>(input, _output, w, h, imageGetRowStride(input.payload.imageValue));
    } else if (pixsize == 4) {
      process<float>(input, _output, w, h, imageGetRowStride(input.payload.imageValue));
    }

    _output.version = input.version + 1;
    return _output;
  }

private:
  std::vector<uint8_t> _bytes;
};

struct FillAlpha {
  static SHOptionalString help() {
    return SHCCSTR("This shard takes an image with 3-channels(RGB) and converts it to a 4-channel(RGBA) image with the alpha "
                   "channel set to 255.");
  }
  static SHOptionalString inputHelp() { return SHCCSTR("The 3 channel image to convert."); }
  static SHOptionalString outputHelp() { return SHCCSTR("The converted 4 channel image with the alpha channel filled."); }

  static SHTypesInfo inputTypes() { return CoreInfo::ImageType; }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }

  OwnedVar _output{};

  template <typename T, typename TA>
  void process(uint8_t *outData, const SHVar &input, int32_t w, int32_t h, int32_t inRowStride, TA alpha_value) {
    auto to = reinterpret_cast<T *>(outData);
    for (auto y = 0; y < h; y++) {
      const auto from = reinterpret_cast<T *>(input.payload.imageValue->data + inRowStride * y);
      for (auto x = 0; x < w; x++) {
        const auto faddr = x * 3;
        const auto taddr = ((w * y) + x) * 4;
        for (auto z = 0; z < 3; z++) {
          to[taddr + z] = from[faddr + z];
        }
        to[taddr + 3] = alpha_value;
      }
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (input.payload.imageValue->channels == 4)
      return input; // nothing to do

    // TODO remove this limit maybe!
    if (input.payload.imageValue->channels != 3)
      throw ActivationError("A 3 or 4 channels image was expected.");

    int32_t w = int32_t(input.payload.imageValue->width);
    int32_t h = int32_t(input.payload.imageValue->height);

    auto pixsize = 1;
    if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_16BITS_INT) == SHIMAGE_FLAGS_16BITS_INT)
      pixsize = 2;
    else if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_32BITS_FLOAT) == SHIMAGE_FLAGS_32BITS_FLOAT)
      pixsize = 4;

    _output = makeImage(w * h * 4 * pixsize);
    auto &outImage = *_output->payload.imageValue;
    outImage.channels = 4;
    outImage.width = uint16_t(w);
    outImage.height = uint16_t(h);
    outImage.flags = input.payload.imageValue->flags;
    uint8_t *outData = outImage.data;

    if (pixsize == 1) {
      process<uint8_t>(outData, input, w, h, imageGetRowStride(input.payload.imageValue), 255);
    } else if (pixsize == 2) {
      process<uint16_t>(outData, input, w, h, imageGetRowStride(input.payload.imageValue), 65535);
    } else if (pixsize == 4) {
      process<float>(outData, input, w, h, imageGetRowStride(input.payload.imageValue), 1.0);
    }

    _output.version = input.version + 1;
    return _output;
  }
};

struct Resize {
  static SHOptionalString help() {
    return SHCCSTR("This shard resizes an image to the specified dimensions specified in the Width and Height parameters.");
  }
  static SHOptionalString inputHelp() { return SHCCSTR("The image to resize."); }
  static SHOptionalString outputHelp() { return SHCCSTR("The resized image."); }

  static SHTypesInfo inputTypes() { return CoreInfo::ImageType; }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }

  static inline Parameters _params{{"Width", SHCCSTR("The target width."), CoreInfo::IntOrIntVar},
                                   {"Height", SHCCSTR("The target height."), CoreInfo::IntOrIntVar}};

  static SHParametersInfo parameters() { return _params; }

  OwnedVar _output{};

  SHVar getParam(int index) {
    if (index == 0)
      return _width;
    else
      return _height;
  }

  void setParam(int index, const SHVar &value) {
    if (index == 0) {
      _width = value;
    } else {
      _height = value;
    }
  }

  void warmup(SHContext *context) {
    _width.warmup(context);
    _height.warmup(context);
  }

  void cleanup(SHContext *context) {
    _width.cleanup();
    _height.cleanup();
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    int w = uint32_t(input.payload.imageValue->width);
    int h = uint32_t(input.payload.imageValue->height);
    int c = uint32_t(input.payload.imageValue->channels);

    int width = int(_width.get().payload.intValue);
    int height = int(_height.get().payload.intValue);
    if (width == 0) {
      width = int(float(w) * float(height) / float(h));
    } else if (height == 0) {
      height = int(float(h) * float(width) / float(w));
    }

    auto pixsize = imageGetPixelSize(input.payload.imageValue);

    _output = makeImage(width * height * c * pixsize);
    auto &outImage = *_output->payload.imageValue;
    outImage.channels = c;
    outImage.width = uint16_t(width);
    outImage.height = uint16_t(height);
    outImage.flags = input.payload.imageValue->flags;
    uint8_t *outData = outImage.data;

    int flags = 0;
    if ((input.payload.imageValue->flags & SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA) == SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA)
      flags = STBIR_FLAG_ALPHA_PREMULTIPLIED;

    if (pixsize == 1) {
      auto res = stbir_resize_uint8_generic(input.payload.imageValue->data, w, h, imageGetRowStride(input.payload.imageValue),
                                            outData, width, height, width * c, c, c == 4 ? 3 : STBIR_ALPHA_CHANNEL_NONE, flags,
                                            STBIR_EDGE_ZERO, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_SRGB, nullptr);
      if (res == 0) {
        throw ActivationError("Failed to resize image!");
      }
    } else if (pixsize == 2) {
      auto res = stbir_resize_uint16_generic((uint16_t *)input.payload.imageValue->data, w, h,
                                             imageGetRowStride(input.payload.imageValue), (uint16_t *)outData, width, height,
                                             width * c * 2, c, c == 4 ? 3 : STBIR_ALPHA_CHANNEL_NONE, flags, STBIR_EDGE_ZERO,
                                             STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_SRGB, nullptr);
      if (res == 0) {
        throw ActivationError("Failed to resize image!");
      }
    } else if (pixsize == 4) {
      auto res =
          stbir_resize_float_generic((float *)input.payload.imageValue->data, w, h, imageGetRowStride(input.payload.imageValue),
                                     (float *)outData, width, height, width * c * 4, c, c == 4 ? 3 : STBIR_ALPHA_CHANNEL_NONE,
                                     flags, STBIR_EDGE_ZERO, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR, nullptr);
      if (res == 0) {
        throw ActivationError("Failed to resize image!");
      }
    }

    _output.version = input.version + 1;
    return _output;
  }

private:
  std::vector<uint8_t> _bytes;
  ParamVar _width{Var(32)};
  ParamVar _height{Var(32)};
};

struct ImageGetPixel {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard analyzes the image(specified in the Image parameter) and outputs the colour value(RGBA) of the pixel at "
        "the specified position.");
  }
  static SHOptionalString inputHelp() {
    return SHCCSTR("The position of the pixel to obtain the colour value of, represented as an int2 vector. The first element of "
                   "the vector represents the x coordinate, and the second element represents the y coordinate.");
  }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The colour(RGBA) values of the pixel at the specified position as a int4 or float4 vector. The first element "
                   "of the vector represents the red value, the second element represents the green value, the third element "
                   "represents the blue value, and the fourth element represents the alpha value.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }

  static inline Types OutputTypes{CoreInfo::Int4Type, CoreInfo::Float4Type};
  static SHTypesInfo outputTypes() { return OutputTypes; }

  PARAM_PARAMVAR(_image, "Position", "The position of the pixel to retrieve", {CoreInfo::ImageType, CoreInfo::ImageVarType});
  PARAM_VAR(_asInteger, "AsInteger", "Read the pixel as an integer", {CoreInfo::BoolType});
  PARAM_VAR(_default, "Default",
            "When specified, out of bounds or otherwise failed reads will outputs this value instead of failing",
            {CoreInfo::NoneType, CoreInfo::Float4Type, CoreInfo::Int4Type});
  PARAM_IMPL(PARAM_IMPL_FOR(_image), PARAM_IMPL_FOR(_asInteger), PARAM_IMPL_FOR(_default));

  ImageGetPixel() { _asInteger = Var(false); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    if (*_asInteger) {
      if (!_default->isNone() && _default.valueType != SHType::Int4)
        throw ComposeError("Default value should be an Int4");
      return CoreInfo::Int4Type;
    } else {
      if (!_default->isNone() && _default.valueType != SHType::Float4)
        throw ComposeError("Default value should be a Float4");
      return CoreInfo::Float4Type;
    }
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  auto static constexpr Conv_UIntToFloat4 = [](const SHImage &image, auto *pixel) {
    constexpr float max = float(std::numeric_limits<std::remove_pointer_t<decltype(pixel)>>::max());
    Float4VarPayload result{};
    for (size_t i = 0; i < image.channels; i++)
      result.float4Value[i] = (float)pixel[i] / max;
    return result;
  };

  auto static constexpr Conv_FloatToInt4 = [](const SHImage &image, auto *pixel) {
    constexpr float max = float(std::numeric_limits<std::remove_pointer_t<decltype(pixel)>>::max());
    Int4VarPayload result{};
    for (size_t i = 0; i < image.channels; i++)
      result.int4Value[i] = (int32_t)pixel[i] * max;
    return result;
  };

  auto static constexpr Conv_CastFloat4 = [](const SHImage &image, auto *pixel) {
    Float4VarPayload result{};
    for (size_t i = 0; i < image.channels; i++)
      result.float4Value[i] = (float)pixel[i];
    return result;
  };

  auto static constexpr Conv_CastInt4 = [](const SHImage &image, auto *pixel) {
    Int4VarPayload result{};
    for (size_t i = 0; i < image.channels; i++)
      result.int4Value[i] = (int32_t)pixel[i];
    return result;
  };

  template <typename TPixel, typename TConv> SHVar convert(SHImage &image, SHInt2 coord, TConv conv) {
    TPixel *row = reinterpret_cast<TPixel *>(image.data + coord[1] * imageGetRowStride(&image));
    TPixel *pix = &row[image.channels * coord[0]];
    return Var(conv(image, pix));
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &image = *_image.get().payload.imageValue;
    int w = uint32_t(image.width);
    int h = uint32_t(image.height);

    SHInt2 coord = input.payload.int2Value;
    if (coord[0] < 0 || coord[0] >= w) {
      if (!_default->isNone())
        return _default;
      throw std::out_of_range("Image fetch x coordinate out of range");
    }
    if (coord[1] < 0 || coord[1] >= h) {
      if (!_default->isNone())
        return _default;
      throw std::out_of_range("Image fetch y coordinate out of range");
    }

    auto pixsize = imageGetPixelSize(_image.get().payload.imageValue);

    if (pixsize == 1) {
      return (bool)*_asInteger ? convert<uint8_t>(image, coord, Conv_CastInt4) //
                               : convert<uint8_t>(image, coord, Conv_UIntToFloat4);
    } else if (pixsize == 2) {
      return (bool)*_asInteger ? convert<uint16_t>(image, coord, Conv_CastInt4) //
                               : convert<uint16_t>(image, coord, Conv_UIntToFloat4);
    } else if (pixsize == 4) {
      return (bool)*_asInteger ? convert<float>(image, coord, Conv_FloatToInt4) //
                               : convert<float>(image, coord, Conv_CastFloat4);
    } else {
      throw std::logic_error("Invalid image format");
    }
  }
};

struct LoadImage {
  enum class BPP { u8, u16, f32 };
  DECL_ENUM_INFO(
      BPP, BPP, "Specifies the bits per pixel for image data. Determines the color depth and precision of image representations.",
      'ibpp');

  static SHOptionalString help() {
    return SHCCSTR("This shard loads an image from a file (specified in the File parameter) or byte array (passed as input) and "
                   "outputs it as an image type, that can "
                   "subsequently be used by other shards such as UI.Image and UI.ImageButton.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("An image represented as a byte array"); }
  static SHOptionalString outputHelp() { return SHCCSTR("The image as an image type."); }

  static SHTypesInfo inputTypes() { return CoreInfo::BytesOrAny; }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }

  PARAM_PARAMVAR(_filename, "File", "The file to load the image from", {CoreInfo::StringStringVarOrNone});
  PARAM_VAR(_bpp, "BPP", "bits per pixel (HDR images loading and such!)", {BPPEnumInfo::Type});
  PARAM_VAR(_premultiplyAlpha, "PremultiplyAlpha", "Toggle premultiplication of alpha channel", {CoreInfo::BoolType});
  PARAM_IMPL(PARAM_IMPL_FOR(_filename), PARAM_IMPL_FOR(_bpp), PARAM_IMPL_FOR(_premultiplyAlpha));

  OwnedVar _output{};

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  void setup() {
    _bpp = Var::Enum(BPP::u8, BPPEnumInfo::Type);
    _premultiplyAlpha = Var(false);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    bool bytesInput = input.valueType == SHType::Bytes;

    _output.reset();
    _output = makeImage();

    stbi_set_flip_vertically_on_load_thread(0);

    uint8_t *bytesValue;
    uint32_t bytesSize;

    // if we have a file input, load them into bytes form
    std::string filename;
    std::vector<uint8_t> buffer;

    if (!bytesInput) {
      // need a proper filename in this case
      if (!getPathChecked(filename, _filename, true)) {
        throw ActivationError(fmt::format("File not found: {}!", filename));
      }

      // load the file
      std::ifstream file(filename, std::ios::binary);
      if (!file) {
        throw ActivationError(fmt::format("Error opening file: {}!", filename));
      }

      // get length of file
      file.seekg(0, std::ios::end);
      std::streampos length = file.tellg();
      file.seekg(0, std::ios::beg);

      // read file into buffer
      buffer.resize(length);
      file.read(reinterpret_cast<char *>(buffer.data()), length);

      bytesValue = buffer.data();
      bytesSize = static_cast<uint32_t>(length);
    } else {
      // image already given in bytes form
      bytesValue = input.payload.bytesValue;
      bytesSize = input.payload.bytesSize;
    }

    _output.valueType = SHType::Image;
    int x, y, n;
    switch ((BPP)_bpp.payload.enumValue) {
    case BPP::u8:
      _output.payload.imageValue->data =
          reinterpret_cast<uint8_t *>(stbi_load_from_memory(bytesValue, static_cast<int>(bytesSize), &x, &y, &n, 0));

      _output.payload.imageValue->flags = 0;
      break;
    case BPP::u16:
      _output.payload.imageValue->data =
          reinterpret_cast<uint8_t *>(stbi_load_16_from_memory(bytesValue, static_cast<int>(bytesSize), &x, &y, &n, 0));

      _output.payload.imageValue->flags = SHIMAGE_FLAGS_16BITS_INT;
      break;
    default:
      _output.payload.imageValue->data =
          reinterpret_cast<uint8_t *>(stbi_loadf_from_memory(bytesValue, static_cast<int>(bytesSize), &x, &y, &n, 0));

      _output.payload.imageValue->flags = SHIMAGE_FLAGS_32BITS_FLOAT;
      break;
    }

    // Set the free function
    if (_output->payload.imageValue->data) {
      _output->payload.imageValue->free = [](SHImage *img) { stbi_image_free(img->data); };
    }

    if (!_output.payload.imageValue->data) {
      throw ActivationError("Failed to load image file");
    }

    _output.payload.imageValue->width = uint16_t(x);
    _output.payload.imageValue->height = uint16_t(y);
    _output.payload.imageValue->channels = uint16_t(n);

    // Premultiply the alpha channel if premultiply option is chosen
    auto pixsize = imageGetPixelSize(_output.payload.imageValue);
    if (_premultiplyAlpha.payload.boolValue) {
      OwnedVar premultipliedOutput = makeImage(imageDeriveDataLength(_output->payload.imageValue));
      auto &dstImage = premultipliedOutput->payload.imageValue;
      dstImage->channels = _output->payload.imageValue->channels;
      dstImage->width = _output->payload.imageValue->width;
      dstImage->height = _output->payload.imageValue->height;
      // premultiply the alpha channel
      switch (pixsize) {
      case 1:
        Imaging::premultiplyAlpha<uint8_t>(_output, premultipliedOutput, _output.payload.imageValue->width,
                                           _output.payload.imageValue->height, imageGetRowStride(_output.payload.imageValue));
        break;
      case 2:
        Imaging::premultiplyAlpha<uint16_t>(_output, premultipliedOutput, _output.payload.imageValue->width,
                                            _output.payload.imageValue->height, imageGetRowStride(_output.payload.imageValue));
        break;
      case 4:
        Imaging::premultiplyAlpha<float>(_output, premultipliedOutput, _output.payload.imageValue->width,
                                         _output.payload.imageValue->height, imageGetRowStride(_output.payload.imageValue));
        break;
      }
      std::swap(_output, premultipliedOutput);
    }

    _output.version = 0;
    return _output;
  }
};

struct WritePNG {
  static SHTypesInfo inputTypes() { return CoreInfo::ImageType; }
  SHTypesInfo outputTypes() { return OutputTypes; }
  static SHOptionalString help() { return SHCCSTR(""); }

  PARAM_PARAMVAR(_filename, "File", "The file to write the image to", {CoreInfo::StringStringVarOrNone});
  PARAM_IMPL(PARAM_IMPL_FOR(_filename));

  static inline Types OutputTypes = {{CoreInfo::BytesType, CoreInfo::ImageType}};
  std::vector<uint8_t> _scratch;
  std::vector<uint8_t> _output;

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  static void write_func(void *context, void *data, int size) {
    auto self = reinterpret_cast<WritePNG *>(context);
    self->_output.resize(size);
    memcpy(self->_output.data(), data, size);
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    // If param is none we output the bytes directly
    if (_filename->valueType == SHType::None) {
      return CoreInfo::BytesType;
    } else {
      return CoreInfo::ImageType;
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    std::string filename;
    if (_filename->valueType != SHType::None) {
      if (!getPathChecked(filename, _filename, false)) {
        throw ActivationError("Path does not exist!");
      }
    }

    auto &image = *input.payload.imageValue;

    auto pixsize = imageGetPixelSize(&image);
    int w = int(image.width);
    int h = int(image.height);
    int c = int(image.channels);

    // demultiply alpha if needed, limited to 4 channels
    if (c == 4 && (image.flags & SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA) == SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA) {
      _scratch.resize(w * h * 4 * pixsize);
      switch (pixsize) {
      case 1:
        Imaging::demultiplyAlpha<uint8_t>(input, _scratch, w, h, imageGetRowStride(&image));
        break;
      case 2:
        Imaging::demultiplyAlpha<uint16_t>(input, _scratch, w, h, imageGetRowStride(&image));
        break;
      case 4:
        Imaging::demultiplyAlpha<float>(input, _scratch, w, h, imageGetRowStride(&image));
        break;
      }

      // all done, write the file or buffer
      if (!filename.empty()) {
        if (0 == stbi_write_png(filename.c_str(), w, h, c, _scratch.data(), w * c))
          throw ActivationError("Failed to write PNG file.");
      } else {
        if (0 == stbi_write_png_to_func(write_func, this, w, h, c, _scratch.data(), w * c))
          throw ActivationError("Failed to write PNG file.");
        return Var(_output.data(), _output.size());
      }
    } else {
      // just write the file or buffer in this case straight
      if (!filename.empty()) {
        if (0 == stbi_write_png(filename.c_str(), w, h, c, image.data, imageGetRowStride(&image)))
          throw ActivationError("Failed to write PNG file.");
      } else {
        if (0 == stbi_write_png_to_func(write_func, this, w, h, c, image.data, imageGetRowStride(&image)))
          throw ActivationError("Failed to write PNG file.");
        return Var(_output.data(), _output.size());
      }
    }

    return input;
  }
};

} // namespace Imaging
} // namespace shards

SHARDS_REGISTER_FN(imaging) {
  using namespace shards::Imaging;

  REGISTER_ENUM(LoadImage::BPPEnumInfo);

  REGISTER_SHARD("GetImagePixel", ImageGetPixel);
  REGISTER_SHARD("Convolve", Convolve);
  REGISTER_SHARD("StripAlpha", StripAlpha);
  REGISTER_SHARD("PremultiplyAlpha", PremultiplyAlpha);
  REGISTER_SHARD("DemultiplyAlpha", DemultiplyAlpha);
  REGISTER_SHARD("FillAlpha", FillAlpha);
  REGISTER_SHARD("ResizeImage", Resize);
  REGISTER_SHARD("LoadImage", LoadImage);
  REGISTER_SHARD("WritePNG", WritePNG);
}
