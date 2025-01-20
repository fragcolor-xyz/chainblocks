/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include "casting.hpp"
#include "shards/inlined.hpp"
#include <shards/core/foundation.hpp>
#include <shards/core/params.hpp>
#include <shards/core/runtime.hpp>
#include <shards/core/module.hpp>
#include <shards/core/hash.hpp>
#include <shards/shards.h>
#include <boost/beast/core/detail/base64.hpp>
#include <type_traits>
#include <boost/algorithm/hex.hpp>
#include <unordered_map>
#include <shards/core/stream_buf.hpp>

namespace shards {
struct FromImage {
  // TODO SIMD this
  template <SHType OF> void toSeq(std::vector<Var> &output, const SHVar &input) {
    if constexpr (OF == SHType::Float) {
      // assume we want 0-1 normalized values
      if (input.valueType != SHType::Image)
        throw ActivationError("Expected Image type.");

      SHImage &image = *input.payload.imageValue;
      const int pixSize = imageGetPixelSize(&image);
      const int componentLen = image.height * image.width * image.channels;

      output.resize(componentLen, Var(0.0));

      if (pixSize == 1) {
        for (int i = 0; i < componentLen; i++) {
          const auto fval = double(image.data[i]) / 255.0;
          output[i].payload.floatValue = fval;
        }
      } else if (pixSize == 2) {
        const auto u16 = reinterpret_cast<uint16_t *>(image.data);
        for (int i = 0; i < componentLen; i++) {
          const auto fval = double(u16[i]) / 65535.0;
          output[i].payload.floatValue = fval;
        }
      } else if (pixSize == 4) {
        const auto f32 = reinterpret_cast<float *>(image.data);
        for (int i = 0; i < componentLen; i++) {
          output[i].payload.floatValue = f32[i];
        }
      }
    } else {
      throw ActivationError("Conversion pair not implemented yet.");
    }
  }
};

struct FromSeq {
  template <SHType OF> OwnedVar toImage(int w, int h, int c, const SHVar &input) {
    OwnedVar output;

    // TODO SIMD this
    if (input.payload.seqValue.len == 0)
      throw ActivationError("Input sequence was empty.");

    const int componentLen = w * h * c;
    const int imageDataLen = std::min(componentLen, int(input.payload.seqValue.len));
    output = makeImage(imageDataLen);
    SHImage &image = *output.payload.imageValue;
    image.channels = c;
    image.width = w;
    image.height = h;

    if constexpr (OF == SHType::Float) {
      // assuming it's scaled 0-1
      for (int i = 0; i < componentLen; i++) {
        image.data[i] = uint8_t(input.payload.seqValue.elements[i].payload.floatValue * 255.0);
      }
    } else {
      throw ActivationError("Conversion pair not implemented yet.");
    }

    return output;
  }

  template <SHType OF> void toBytes(std::vector<uint8_t> &buffer, const SHVar &input) {
    buffer.resize(size_t(input.payload.seqValue.len));

    if constexpr (OF == SHType::Int) {
      for (uint32_t i = 0; i < input.payload.seqValue.len; i++) {
        const auto val = input.payload.seqValue.elements[i].payload.intValue;
        if (val > UINT8_MAX || val < 0)
          throw ActivationError("Value out of byte range (0~255)");

        buffer[i] = uint8_t(val);
      }
    } else {
      throw ActivationError("Conversion pair not implemented yet.");
    }
  }
};

struct FromBytes {
  template <SHType OF> void toSeq(std::vector<Var> &output, const SHVar &input) {
    if constexpr (OF == SHType::Int) {
      // assume we want 0-1 normalized values
      if (input.valueType != SHType::Bytes)
        throw ActivationError("Expected Bytes type.");

      output.resize(input.payload.bytesSize);

      for (uint32_t i = 0; i < input.payload.bytesSize; i++) {
        const auto b = int64_t(input.payload.bytesValue[i]);
        output[i] = Var(b);
      }
    } else {
      throw ActivationError("Conversion pair not implemented yet.");
    }
  }
};

template <SHType SHTYPE, SHType SHOTHER> struct ToSeq {
  static inline Type _inputType{{SHTYPE}};
  static inline Type _outputElemType{{SHOTHER}};
  static inline Type _outputType{{SHType::Seq, {.seqTypes = _outputElemType}}};

  static SHTypesInfo inputTypes() { return _inputType; }
  static SHOptionalString inputHelp() {
    if constexpr (SHTYPE == SHType::Image) {
      return SHCCSTR("Takes an image as input.");
    } else if constexpr (SHTYPE == SHType::Bytes) {
      return SHCCSTR("Takes a byte array as input.");
    } else {
      return SHCCSTR("Takes an input value.");
    }
  }

  static SHTypesInfo outputTypes() { return _outputType; }
  static SHOptionalString outputHelp() {
    if constexpr (SHTYPE == SHType::Image) {
      return SHCCSTR("Outputs the input image represented as a seqeunce of floats.");
    } else if constexpr (SHTYPE == SHType::Bytes) {
      return SHCCSTR("Outputs the input bytes represented as a sequence of integers.");
    } else {
      return SHCCSTR(
          "The output is a sequence of the specified type created from the input bytes or image. Each byte or pixel is "
          "converted to an element of the output sequence.");
    }
  }

  static SHOptionalString help() {
    if constexpr (SHTYPE == SHType::Image) {
      return SHCCSTR("Convert an image into a sequence of floats. Each pixel in the image is converted to a float value between "
                     "0 and 1 and stored in the sequence.");
    } else if constexpr (SHTYPE == SHType::Bytes) {
      return SHCCSTR(
          "Convert bytes into a sequence of integers. Each byte is interpreted as an integer and stored in the sequence.");
    } else {
      return SHCCSTR("Converts a sequence of bytes or an image into a sequence of another specified type. Each byte or pixel in "
                     "the input is interpreted and converted to an element of the output sequence of the specified type.");
    }
  }

  std::vector<Var> _output;

  SHVar activate(SHContext *context, const SHVar &input) {
    // do not .clear here, for speed, let From manage that
    if constexpr (SHTYPE == SHType::Image) {
      FromImage c;
      c.toSeq<SHOTHER>(_output, input);
    } else if constexpr (SHTYPE == SHType::Bytes) {
      FromBytes c;
      c.toSeq<SHOTHER>(_output, input);
    } else {
      throw ActivationError("Conversion pair not implemented yet.");
    }
    return Var(_output);
  }
};

template <SHType SHTYPE> struct ToString1 {
  static inline Type _inputType{{SHTYPE}};

  static SHTypesInfo inputTypes() { return _inputType; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("Accepts a byte array as input. Each byte in the sequence is interpreted as a character.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output is a string created from the input sequence of bytes."); }

  static SHOptionalString help() {
    return SHCCSTR("Converts a sequence of bytes into a string. Each byte in the sequence is interpreted as a character in the "
                   "resulting string.");
  }

  std::string _output;

  SHVar activate(SHContext *context, const SHVar &input) {
    // do not .clear here, for speed, let From manage that
    if constexpr (SHTYPE == SHType::Bytes) {
      _output.clear();
      _output.assign((const char *)input.payload.bytesValue, size_t(input.payload.bytesSize));
    } else {
      throw ActivationError("Conversion pair not implemented yet.");
    }
    return Var(_output);
  }
};

template <SHType FROMTYPE> struct ToImage {
  static inline Type _inputElemType{{FROMTYPE}};
  static inline Type _inputType{{SHType::Seq, {.seqTypes = _inputElemType}}};

  static SHTypesInfo inputTypes() { return _inputType; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("Takes a sequence of floats as input and converts it into an image. The sequence length must be equal to "
                   "Width x Height x Channels.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }
  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs and image."); }

  static SHOptionalString help() {
    return SHCCSTR("Converts a sequence of floats into an image. The image dimensions (width and height) and the "
                   "number of channels are specified by the appropriate parameters.");
  }

  static inline Parameters _params{{"Width", SHCCSTR("The width of the output image."), {CoreInfo::IntType}},
                                   {"Height", SHCCSTR("The height of the output image."), {CoreInfo::IntType}},
                                   {"Channels", SHCCSTR("The channels of the output image."), {CoreInfo::IntType}}};

  static SHParametersInfo parameters() { return _params; }

  OwnedVar _output;

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(int(_width));
    case 1:
      return Var(int(_height));
    case 2:
      return Var(int(_channels));
    default:
      return {};
    }
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _width = uint16_t(value.payload.intValue);
      break;
    case 1:
      _height = uint16_t(value.payload.intValue);
      break;
    case 2:
      _channels = uint8_t(std::min(SHInt(4), std::max(SHInt(1), value.payload.intValue)));
      break;
    }
  }

  void cleanup(SHContext *) { _output.reset(); }

  SHVar activate(SHContext *context, const SHVar &input) {
    FromSeq c;
    _output = c.toImage<FROMTYPE>(int(_width), int(_height), int(_channels), input);
    return _output;
  }

private:
  uint8_t _channels = 1;
  uint16_t _width = 16;
  uint16_t _height = 16;
};

template <SHType FROMTYPE> struct ToBytes {
  static inline Type _inputElemType{{FROMTYPE}};
  static inline Type _inputType{{SHType::Seq, {.seqTypes = _inputElemType}}};

  static SHTypesInfo inputTypes() { return _inputType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Accepts sequence of integers as input."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BytesType; }
  static SHOptionalString outputHelp() { return SHCCSTR("A byte array representing the sequence of integers."); }

  static SHOptionalString help() {
    return SHCCSTR("Converts a sequence of integers into a byte array. Each integer in the sequence is serialized into its "
                   "binary representation and concatenated into the resulting byte array.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    FromSeq c;
    c.toBytes<FROMTYPE>(_buffer, input);
    return Var(_buffer);
  }

private:
  std::vector<uint8_t> _buffer;
};

struct ToString {
  VarStringStream stream;

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The string representation of the input value."); }

  static SHOptionalString help() { return SHCCSTR("Converts any input value to its string representation."); }

  SHVar activate(SHContext *context, const SHVar &input) {
    stream.write(input);
    return Var(stream.str());
  }
};

struct ToHex {
  VarStringStream stream;
  static inline Types toHexTypes{CoreInfo::IntType, CoreInfo::Int16Type, CoreInfo::BytesType, CoreInfo::StringType};

  static SHTypesInfo inputTypes() { return toHexTypes; }
  static SHOptionalString inputHelp() { return SHCCSTR("Takes an integer, byte array, or string value."); }

  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The hexadecimal string representation of the input value."); }

  static SHOptionalString help() {
    return SHCCSTR("Converts an integer, bytes, or string value into its hexadecimal string representation.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    stream.tryWriteHex(input);
    return Var(stream.str());
  }
};

struct ToAny {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Converts the input to any type"); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The same value as the input but typed as Any."); }

  static SHOptionalString help() {
    return SHCCSTR("Converts an integer, bytes, or string value into its hexadecimal string representation.");
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    data.shard->inlineShardId = InlineShard::NoopShard;
    return CoreInfo::AnyType;
  }
  SHVar activate(SHContext *context, const SHVar &input) { return input; }
};

struct VarAddr {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The name of the variable whose address is to be retrieved."); }

  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The memory address of the specified variable, represented as an integer.");
  }

  static SHOptionalString help() {
    return SHCCSTR("Retrieves the memory address of the specified variable and outputs it as an integer.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto v = referenceVariable(context, SHSTRVIEW(input));
    auto res = Var(reinterpret_cast<int64_t>(v));
    releaseVariable(v);
    return res;
  }
};

struct VarPtr {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("The variable whose pointer is to be retrieved. It must be of type sequence, bytes, or string.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The memory address of the specified variable's data, represented as an integer.");
  }

  static SHOptionalString help() {
    return SHCCSTR("Retrieves the memory address of the data contained in the specified variable and outputs it as an integer. "
                   "The variable must be of type sequence, bytes, or string.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (input.valueType == SHType::Seq)
      return Var(reinterpret_cast<int64_t>(input.payload.seqValue.elements));
    else if (input.valueType == SHType::Bytes)
      return Var(reinterpret_cast<int64_t>(input.payload.bytesValue));
    else if (input.valueType == SHType::String)
      return Var(reinterpret_cast<int64_t>(input.payload.stringValue));
    else
      throw ActivationError("Expected sequence, bytes or string type.");
  }
};

struct BitSwap32 {
  static SHTypesInfo inputTypes() { return CoreInfo::IntType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Takes a 32-bit integer value."); }

  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the reversed bytes as an integer."); }

  static SHOptionalString help() {
    return SHCCSTR("This shard takes a 32-bit integer, reverses their order of its bytes, and outputs the result as an integer. "
                   "This is useful for converting between different endianness formats.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto i32 = static_cast<uint32_t>(input.payload.intValue);
    i32 = __builtin_bswap32(i32);
    return Var(static_cast<int64_t>(i32));
  }
};

struct BitSwap64 {
  static SHTypesInfo inputTypes() { return CoreInfo::IntType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Takes a 64-bit integer value."); }

  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the reversed bytes as an integer."); }

  static SHOptionalString help() {
    return SHCCSTR("This shard takes a 64-bit integer, reverses their order of its bytes, and outputs the result as an integer. "
                   "This is useful for converting between different endianness formats.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto i64 = static_cast<uint64_t>(input.payload.intValue);
    i64 = __builtin_bswap64(i64);
    return Var(static_cast<int64_t>(i64));
  }
};

// Matches input against expected type and throws if it doesn't match
// unsafe only matches basic type
static inline void expectTypeCheck(const SHVar &input, uint64_t expectedTypeHash, const SHTypeInfo &expectedType, bool unsafe) {
  if (expectedType.basicType == SHType::Any) {
    return;
  } else if (input.valueType != expectedType.basicType) {
    throw ActivationError(fmt::format("Unexpected input type, value: {} expected type: {}", input, expectedType));
  } else if (input.valueType == SHType::Seq && input.payload.seqValue.len == 0) {
    // early out if seq is empty
    return;
  } else if (!unsafe) {
    static thread_local std::unordered_map<uint64_t, TypeInfo> typeCache;
    auto inputTypeHash = deriveTypeHash64(input);
    if (inputTypeHash != expectedTypeHash) {
      // Do an even deeper type check
      auto it = typeCache.find(inputTypeHash);
      if (it == typeCache.end()) {
        it = typeCache.emplace(inputTypeHash, TypeInfo{input, SHInstanceData{}}).first;
      }
      if (!matchTypes(it->second, expectedType, false, true, true)) {
        throw ActivationError(fmt::format("Unexpected value: {} expected type: {}", input, expectedType));
      }
    }
  }
}

template <SHType ET> struct ExpectX {
  static inline Type outputType{{ET}};
  SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyType; }

  SHTypesInfo outputTypes() { return outputType; }
  static SHOptionalString outputHelp() {
    if constexpr (ET == SHType::Int) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Int.");
    } else if constexpr (ET == SHType::Int2) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Int2.");
    } else if constexpr (ET == SHType::Int3) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Int3.");
    } else if constexpr (ET == SHType::Int4) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Int4.");
    } else if constexpr (ET == SHType::Int8) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Int8.");
    } else if constexpr (ET == SHType::Int16) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Int16");
    } else if constexpr (ET == SHType::Color) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Color.");
    } else if constexpr (ET == SHType::Float) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Float.");
    } else if constexpr (ET == SHType::Float2) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Float2.");
    } else if constexpr (ET == SHType::Float3) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Float3.");
    } else if constexpr (ET == SHType::Float4) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Float4.");
    } else if constexpr (ET == SHType::Bytes) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Bytes.");
    } else if constexpr (ET == SHType::Bool) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Bool.");
    } else if constexpr (ET == SHType::Image) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Image.");
    } else if constexpr (ET == SHType::String) {
      return SHCCSTR("Outputs the input value unchanged if it is of type String.");
    } else if constexpr (ET == SHType::Wire) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Wire.");
    } else if constexpr (ET == SHType::Table) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Table.");
    } else if constexpr (ET == SHType::Audio) {
      return SHCCSTR("Outputs the input value unchanged if it is of type Audio.");
    } else if constexpr (ET == SHType::None) {
      return SHCCSTR("Outputs the input value unchanged if it is of type None.");
    } else {
      return SHCCSTR("Outputs the input value unchanged if it is of the appropriate type.");
    }
  }

  static SHOptionalString help() {
    if constexpr (ET == SHType::Int) {
      return SHCCSTR("Checks the input value if it is of type Int. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Int2) {
      return SHCCSTR(
          "Checks the input value if it is a vector with two Int elements. The shard outputs the input value unchanged if it is "
          "of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Int3) {
      return SHCCSTR(
          "Checks the input value if it is a vector with three Int elements. The shard outputs the input value unchanged if it "
          "is of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Int4) {
      return SHCCSTR(
          "Checks the input value if it is a vector with four Int elements. The shard outputs the input value unchanged if it is "
          "of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Int8) {
      return SHCCSTR(
          "Checks the input value if it is a vector with eight Int elements. The shard outputs the input value unchanged if it "
          "is of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Int16) {
      return SHCCSTR(
          "Checks the input value if it is a vector with sixteen Int elements. The shard outputs the input value unchanged if it "
          "is of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Color) {
      return SHCCSTR(
          "Checks the input value if it is vector of four color channels (RGBA). The shard outputs the input value unchanged if "
          "it is of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Float) {
      return SHCCSTR("Checks the input value if it is of type Float. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, it will fail.");
    } else if constexpr (ET == SHType::Float2) {
      return SHCCSTR(
          "Checks the input value if it is a vector with two Float elements. The shard outputs the input value unchanged if it "
          "is of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Float3) {
      return SHCCSTR(
          "Checks the input value if it is a vector with three Float elements. The shard outputs the input value unchanged if it "
          "is of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Float4) {
      return SHCCSTR(
          "Checks the input value if it is a vector with float Float elements. The shard outputs the input value unchanged if it "
          "is of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Bytes) {
      return SHCCSTR("Checks the input value if it is of type Bytes. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Bool) {
      return SHCCSTR("Checks the input value if it is a Boolean. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Image) {
      return SHCCSTR("Checks the input value if it is an Image file. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::String) {
      return SHCCSTR("Checks the input value if it is a String. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Wire) {
      return SHCCSTR("Checks the input value if it is a Wire. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Table) {
      return SHCCSTR("Checks the input value if it is a Table. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::Audio) {
      return SHCCSTR("Checks the input value if it is an Audio file. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if constexpr (ET == SHType::None) {
      return SHCCSTR("Checks the input value if it is none. The shard outputs the input value unchanged if it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else {
      return SHCCSTR("Checks the input value if it is of the type specified. The shard outputs the input value unchanged if it "
                     "is of the appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    }
  }
  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(input.valueType != ET)) {
      SHLOG_ERROR("Unexpected value: {}", input);
      throw ActivationError("Expected type " + type2Name(ET) + " got instead " + type2Name(input.valueType));
    }
    return input;
  }
};

struct ExpectSeq {
  SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyType; }
  SHTypesInfo outputTypes() { return CoreInfo::AnySeqType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the input value unchanged if it is a sequence."); }
  static SHOptionalString help() {
    return SHCCSTR(
        "Checks if the input value is a sequence; otherwise, the shard will trigger an error, preventing further execution.");
  }
  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(input.valueType != SHType::Seq)) {
      SHLOG_ERROR("Unexpected value: {}", input);
      throw ActivationError("Expected a sequence type got instead " + type2Name(input.valueType));
    }
    return input;
  }
};

template <Type &ET, bool UNSAFE = false> struct ExpectXComplex {
  static inline Parameters params{
      {"Unsafe",
       SHCCSTR("When set to true, it will skip type comparison. Generally unsafe but it can improve performance. Only set to "
               "true if you are certain that the input type matches the expected type."),
       {CoreInfo::BoolType}}};
  static inline uint64_t ExpectedHash = deriveTypeHash64(ET);

  bool _unsafe{UNSAFE};

  SHParametersInfo parameters() { return params; }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("This shard accepts a sequence of values."); }
  static SHTypesInfo outputTypes() { return ET; }
  static SHOptionalString outputHelp() {
    if (ET == CoreInfo::FloatSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Floats.");
    } else if (ET == CoreInfo::Float2SeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Float2 vectors.");
    } else if (ET == CoreInfo::Float3SeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Float3 vectors.");
    } else if (ET == CoreInfo::Float4SeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Floats4 vectors.");
    } else if (ET == CoreInfo::IntSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Ints.");
    } else if (ET == CoreInfo::Int2SeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Int2 vectors.");
    } else if (ET == CoreInfo::Int3SeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Int3 vectors.");
    } else if (ET == CoreInfo::Int4SeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Int4 vectors.");
    } else if (ET == CoreInfo::Int8SeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Int8 vectors.");
    } else if (ET == CoreInfo::Int16SeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Int16 vectors.");
    } else if (ET == CoreInfo::ColorSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Color vectors.");
    } else if (ET == CoreInfo::BytesSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Bytes.");
    } else if (ET == CoreInfo::BoolSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Booleans.");
    } else if (ET == CoreInfo::ImageSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Images.");
    } else if (ET == CoreInfo::StringSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Strings.");
    } else if (ET == CoreInfo::WireSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Wires.");
    } else if (ET == CoreInfo::AudioSeqType) {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of Audio data.");
    } else {
      return SHCCSTR("Outputs the input value unchanged if it is a sequence of values of the appropriate type.");
    }
  }

  static SHOptionalString help() {
    if (ET == CoreInfo::FloatSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Floats. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::Float2SeqType) {
      return SHCCSTR("Checks if the input value is a sequence of Float2 vectors. The shard outputs the input value unchanged if "
                     "it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::Float3SeqType) {
      return SHCCSTR("Checks if the input value is a sequence of Float3 vectors. The shard outputs the input value unchanged if "
                     "it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::Float4SeqType) {
      return SHCCSTR("Checks if the input value is a sequence of Float4 vectors. The shard outputs the input value unchanged if "
                     "it is of the "
                     "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::IntSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Ints. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::Int2SeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Int2 vectors. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::Int3SeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Int3 vectors. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::Int4SeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Int4 vectors. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::Int8SeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Int8 vectors. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::Int16SeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Int16 vectors. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::ColorSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Color vectors. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::BytesSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Bytes. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::BoolSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Bools. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::ImageSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Images files. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::StringSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Strings. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::WireSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Wires. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else if (ET == CoreInfo::AudioSeqType) {
      return SHCCSTR(
          "Checks if the input value is a sequence of Audio files. The shard outputs the input value unchanged if it is of the "
          "appropriate type; otherwise, the shard will trigger an error, preventing further execution.");
    } else {
      return SHCCSTR(
          "Checks if the input value matches the expected complex type. If the input value does not match the expected type, an "
          "error is thrown.");
    }
  }

  void setParam(int index, const SHVar &value) { _unsafe = value.payload.boolValue; }

  SHVar getParam(int index) { return Var(_unsafe); }

  SHVar activate(SHContext *context, const SHVar &input) {
    const static SHTypeInfo info = ET;
    expectTypeCheck(input, ExpectedHash, info, (bool)_unsafe);
    return input;
  }
};

struct Expect {
  SHTypeInfo _expectedType{};
  uint64_t _expectedTypeHash{0};

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input value unchanged if it matches the expected type."); }

  static SHOptionalString help() {
    return SHCCSTR("Checks if the input value matches the expected type specified by the 'Type' parameter. The shard outputs the "
                   "input value unchanged if it is of "
                   "the appropriate type, the shard will trigger an error, preventing further execution. The 'Unsafe' parameter "
                   "can be set to skip deep type "
                   "hashing and comparison to improve performance.");
  }

  PARAM_VAR(_type, "Type", "The type to expect", {CoreInfo::TypeType});
  PARAM_VAR(_unsafe, "Unsafe",
            "When set to true, it will skip type comparison. Generally unsafe but it can improve performance. Only set to true "
            "if you are certain that the input type matches the expected type.",
            {CoreInfo::BoolType});
  PARAM_IMPL(PARAM_IMPL_FOR(_type), PARAM_IMPL_FOR(_unsafe));

  Expect() { _unsafe = Var(false); }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_type->valueType != SHType::Type) {
      throw std::logic_error("Expect shard type parameter must be a set");
    }

    _expectedType = *_type->payload.typeValue;
    _expectedTypeHash = deriveTypeHash64(_expectedType);

    return _expectedType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    expectTypeCheck(input, _expectedTypeHash, _expectedType, (bool)*_unsafe);
    return input;
  }
};

struct ExpectLike {
  SHTypeInfo _expectedType{};
  uint64_t _expectedTypeHash{0};
  bool _derived{false};

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the input value unchanged if it matches the expected type."); }

  static SHOptionalString help() {
    return SHCCSTR("Checks if the input value matches the type of the value provided in the TypeOf parameter or the output type "
                   "of the given expression in the OutputOf parameter. The shard outputs the input value unchanged if it is of "
                   "the appropriate type; otherwise, the shard will trigger an error, preventing further execution. Note that it "
                   "can only compare with either one of the "
                   "parameters, not both; an error will be thrown if both are provided. The 'Unsafe' parameter can be set to "
                   "skip deep type hashing and comparison to improve performance.");
  }

  PARAM_PARAMVAR(_typeOf, "TypeOf", "The type of the constant given here will be checked against this shard's input.",
                 {CoreInfo::AnyType});
  PARAM(ShardsVar, _outputOf, "OutputOf",
        "Evaluates the output type of the given expression. That type of the output will be checked against this shard's input.",
        {CoreInfo::ShardsOrNone});
  PARAM_VAR(_unsafe, "Unsafe",
            "When set to true, it will skip type comparison. Generally unsafe but it can improve performance. Only set to true "
            "if you are certain that the input type matches the expected type.",
            {CoreInfo::BoolType});
  PARAM_IMPL(PARAM_IMPL_FOR(_typeOf), PARAM_IMPL_FOR(_outputOf), PARAM_IMPL_FOR(_unsafe));

  ExpectLike() { _unsafe = Var(false); }

  void clearDerived() {
    if (_derived) {
      freeDerivedInfo(_expectedType);
      _expectedType = {};
      _derived = false;
    }
  }

  void destroy() { clearDerived(); }

  void warmup(SHContext *context) { _typeOf.warmup(context); }

  void cleanup(SHContext *context) { _typeOf.cleanup(context); }

  SHTypeInfo compose(const SHInstanceData &data) {
    bool haveOutputOf = _outputOf.shards().len > 0;
    if (_typeOf.isNotNullConstant() && haveOutputOf) {
      throw ComposeError("Only one of TypeOf or OutputOf is allowed");
    }

    if (_typeOf.isVariable()) {
      auto type = findExposedVariable(data.shared, _typeOf);
      if (!type.has_value())
        throw ComposeError(fmt::format("Can not derive type of variable {}, it was not found", SHSTRVIEW((*_typeOf))));
      _expectedType = type->exposedType;
    } else if (_typeOf.isNotNullConstant()) {
      clearDerived();
      _expectedType = deriveTypeInfo((SHVar &)_typeOf, data);
      _derived = true;
    } else if (haveOutputOf) {
      _expectedType = _outputOf.compose(data).outputType;
    } else {
      throw ComposeError("One of TypeOf or OutputOf is required");
    }

    _expectedTypeHash = deriveTypeHash64(_expectedType);

    return _expectedType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    expectTypeCheck(input, _expectedTypeHash, _expectedType, (bool)*_unsafe);
    return input;
  }
};

struct TypeOf {
  SHTypeInfo _expectedType;

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }

  static SHTypesInfo outputTypes() { return CoreInfo::TypeType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the type of the specified expression's output."); }

  static SHOptionalString help() {
    return SHCCSTR("Evaluates the output type of the given expression specified by the 'OutputOf' parameter and outputs that "
                   "type. No input is required for this shard.");
  }

  PARAM(ShardsVar, _outputOf, "OutputOf", "Evaluates the output type of the given expression.", {CoreInfo::ShardsOrNone});
  PARAM_IMPL(PARAM_IMPL_FOR(_outputOf));

  TypeOf() {}

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHTypeInfo compose(const SHInstanceData &data) {
    _expectedType = _outputOf.compose(data).outputType;
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    SHVar out{};
    out.valueType = SHType::Type;
    out.payload.typeValue = &_expectedType;
    return out;
  }
};

template <SHType ET> struct IsX {
  SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyType; }

  SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() {
    if constexpr (ET == SHType::Int) {
      return SHCCSTR("Outputs true if the input value is of type Int, and false otherwise.");
    } else if constexpr (ET == SHType::Int2) {
      return SHCCSTR("Outputs true if the input value is of type Int2, and false otherwise.");
    } else if constexpr (ET == SHType::Int3) {
      return SHCCSTR("Outputs true if the input value is of type Int3, and false otherwise.");
    } else if constexpr (ET == SHType::Int4) {
      return SHCCSTR("Outputs true if the input value is of type Int4, and false otherwise.");
    } else if constexpr (ET == SHType::Int8) {
      return SHCCSTR("Outputs true if the input value is of type Int8, and false otherwise.");
    } else if constexpr (ET == SHType::Int16) {
      return SHCCSTR("Outputs true if the input value is of type Int16, and false otherwise.");
    } else if constexpr (ET == SHType::Color) {
      return SHCCSTR("Outputs true if the input value is of type Color, and false otherwise.");
    } else if constexpr (ET == SHType::Float) {
      return SHCCSTR("Outputs true if the input value is of type Float, and false otherwise.");
    } else if constexpr (ET == SHType::Float2) {
      return SHCCSTR("Outputs true if the input value is of type Float2, and false otherwise.");
    } else if constexpr (ET == SHType::Float3) {
      return SHCCSTR("Outputs true if the input value is of type Float3, and false otherwise.");
    } else if constexpr (ET == SHType::Float4) {
      return SHCCSTR("Outputs true if the input value is of type Float4, and false otherwise.");
    } else if constexpr (ET == SHType::Bytes) {
      return SHCCSTR("Outputs true if the input value is of type Bytes, and false otherwise.");
    } else if constexpr (ET == SHType::Bool) {
      return SHCCSTR("Outputs true if the input value is of type Bool, and false otherwise.");
    } else if constexpr (ET == SHType::Image) {
      return SHCCSTR("Outputs true if the input value is of type Image, and false otherwise.");
    } else if constexpr (ET == SHType::String) {
      return SHCCSTR("Outputs true if the input value is of type String, and false otherwise.");
    } else if constexpr (ET == SHType::Wire) {
      return SHCCSTR("Outputs true if the input value is of type Wire, and false otherwise.");
    } else if constexpr (ET == SHType::Table) {
      return SHCCSTR("Outputs true if the input value is of type Table, and false otherwise.");
    } else if constexpr (ET == SHType::Audio) {
      return SHCCSTR("Outputs true if the input value is of type Audio, and false otherwise.");
    } else if constexpr (ET == SHType::None) {
      return SHCCSTR("Outputs true if the input value is of type None, and false otherwise.");
    } else {
      return SHCCSTR("Outputs true if the input value is of the specified type, and false otherwise.");
    }
  }

  static SHOptionalString help() {
    if constexpr (ET == SHType::Int) {
      return SHCCSTR("Checks the input value if it is of type Int. The shard will return true if the input value is of type Int, "
                     "and false otherwise.");
    } else if constexpr (ET == SHType::Int2) {
      return SHCCSTR("Checks the input value if it is a vector of 2 Int elements. The shard will return true if the input is of "
                     "the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Int3) {
      return SHCCSTR("Checks the input value if it is a vector of 3 Int elements. The shard will return true if the input is of "
                     "the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Int4) {
      return SHCCSTR("Checks the input value if it is a vector of 4 Int elements. The shard will return true if the input is of "
                     "the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Int8) {
      return SHCCSTR("Checks the input value if it is a vector of 8 Int elements. The shard will return true if the input is of "
                     "the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Int16) {
      return SHCCSTR("Checks the input value if it is a vector of 16 Int elements. The shard will return true if the input is of "
                     "the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Color) {
      return SHCCSTR("Checks the input value if it is vector of four color channels (RGBA). The shard will return true if the "
                     "input is of the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Float) {
      return SHCCSTR("Checks the input value if it is of type Float. The shard will return true if the input is of the "
                     "appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Float2) {
      return SHCCSTR("Checks the input value if it is a vector with two Float elements. The shard will return true if the input "
                     "is of the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Float3) {
      return SHCCSTR("Checks the input value if it is a vector with three Float elements. The shard will return true if the "
                     "input is of the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Float4) {
      return SHCCSTR("Checks the input value if it is a vector with float Float elements. The shard will return true if the "
                     "input is of the appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Bytes) {
      return SHCCSTR("Checks the input value if it is of type Bytes. The shard will return true if the input is of the "
                     "appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::Bool) {
      return SHCCSTR("Checks the input value if it is a Boolean. The shard will return true if the input is of the appropriate "
                     "type, and false otherwise.");
    } else if constexpr (ET == SHType::Image) {
      return SHCCSTR("Checks the input value if it is an Image file. The shard will return true if the input is of the "
                     "appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::String) {
      return SHCCSTR("Checks the input value if it is a String. The shard will return true if the input is of the appropriate "
                     "type, and false otherwise.");
    } else if constexpr (ET == SHType::Wire) {
      return SHCCSTR("Checks the input value if it is a Wire. The shard will return true if the input is of the appropriate "
                     "type, and false otherwise.");
    } else if constexpr (ET == SHType::Table) {
      return SHCCSTR("Checks the input value if it is a Table. The shard will return true if the input is of the appropriate "
                     "type, and false otherwise.");
    } else if constexpr (ET == SHType::Audio) {
      return SHCCSTR("Checks the input value if it is an Audio file. The shard will return true if the input is of the "
                     "appropriate type, and false otherwise.");
    } else if constexpr (ET == SHType::None) {
      return SHCCSTR("Checks the input value if it is none. The shard will return true if the input is of the appropriate type, "
                     "and false otherwise.");
    } else {
      return SHCCSTR("Checks the input value if it is of the type specified. The shard will return true if the input is of the "
                     "appropriate type, and false otherwise.");
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) { return Var(input.valueType == ET); }
};

struct ToBase64 {
  std::string output;
  static inline Types _inputTypes{{CoreInfo::BytesType, CoreInfo::StringType}};

  static SHTypesInfo inputTypes() { return _inputTypes; }
  static SHOptionalString inputHelp() { return SHCCSTR("Accepts a byte array or a string value as input."); }

  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the Base64 encoded string representation of the input value."); }

  static SHOptionalString help() {
    return SHCCSTR("Encodes the input bytes or string value to its Base64 string representation.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    output.clear();
    if (input.valueType == SHType::Bytes) {
      auto req = boost::beast::detail::base64::encoded_size(input.payload.bytesSize);
      output.resize(req);
      auto written = boost::beast::detail::base64::encode(output.data(), input.payload.bytesValue, input.payload.bytesSize);
      output.resize(written);
    } else {
      const auto len = input.payload.stringLen > 0 || input.payload.stringValue == nullptr ? input.payload.stringLen
                                                                                           : strlen(input.payload.stringValue);
      auto req = boost::beast::detail::base64::encoded_size(len);
      output.resize(req);
      auto written = boost::beast::detail::base64::encode(output.data(), input.payload.stringValue, len);
      output.resize(written);
    }
    return Var(output);
  }
};

struct FromBase64 {
  std::vector<uint8_t> output;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHOptionalString inputHelp() { return SHCCSTR("A Base64 encoded string to be decoded."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BytesType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The decoded bytes from the input Base64 string."); }

  static SHOptionalString help() { return SHCCSTR("Decodes a Base64 encoded string to its original byte representation."); }

  SHVar activate(SHContext *context, const SHVar &input) {
    output.clear();
    auto len = input.payload.stringLen > 0 || input.payload.stringValue == nullptr ? input.payload.stringLen
                                                                                   : strlen(input.payload.stringValue);
    auto req = boost::beast::detail::base64::decoded_size(len);
    output.resize(req);
    auto [written, _] = boost::beast::detail::base64::decode(output.data(), input.payload.stringValue, len);
    output.resize(written);
    return Var(output.data(), output.size());
  }
};

struct HexToBytes {
  std::vector<uint8_t> output;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("Accepts a hexadecimal string as input. The input may optionally start with '0x' or '0X'.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::BytesType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The decoded byte array from the input hexadecimal string."); }

  static SHOptionalString help() { return SHCCSTR("Converts a hexadecimal string to its original byte array representation."); }

  int convert(const char *hex_str, size_t hex_str_len, unsigned char *byte_array, size_t byte_array_max) {
    size_t i = 0, j = 0;

    // The output array size is half the hex_str length (rounded up)
    size_t byte_array_size = (hex_str_len + 1) / 2;

    if (byte_array_size > byte_array_max) {
      // Too big for the output array
      return -1;
    }

    if (hex_str_len % 2 == 1) {
      // hex_str is an odd length, so assume an implicit "0" prefix
      if (sscanf(&(hex_str[0]), "%1hhx", &(byte_array[0])) != 1) {
        return -1;
      }

      i = j = 1;
    }

    for (; i < hex_str_len; i += 2, j++) {
      if (sscanf(&(hex_str[i]), "%2hhx", &(byte_array[j])) != 1) {
        return -1;
      }
    }

    return byte_array_size;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto len = SHSTRLEN(input);
    if (len == 0)
      return Var((uint8_t *)nullptr, 0);

    auto src = input.payload.stringValue;
    // allow 0x prefix
    if (src[0] == '0' && (src[1] == 'x' || src[1] == 'X'))
      src += 2;

    auto input_view = std::string_view(src);
    output.clear();
    output.resize(input_view.size() / 2 + input_view.size() % 2);
    convert(input_view.data(), input_view.size(), output.data(), output.size());

    return Var(output.data(), output.size());
  }
};

struct StringToBytes {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHOptionalString inputHelp() { return SHCCSTR("A string to be converted to bytes."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BytesType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The byte representation of the input string."); }

  static SHOptionalString help() { return SHCCSTR("Converts a string to its byte representation."); }

  SHVar activate(SHContext *context, const SHVar &input) {
    const auto len = SHSTRLEN(input);
    return Var((uint8_t *)input.payload.stringValue, uint32_t(len));
  }
};

struct ImageToBytes {
  static SHTypesInfo inputTypes() { return CoreInfo::ImageType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Accepts an image as input."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BytesType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input image represented as a byte array."); }

  static SHOptionalString help() { return SHCCSTR("Converts an image into a byte array."); }

  SHVar activate(SHContext *context, const SHVar &input) {
    SHImage &image = *input.payload.imageValue;
    uint32_t imageDataLength = imageDeriveDataLength(&image);
    return Var((uint8_t *)image.data, imageDataLength);
  }
};

struct AudioToBytes {
  static SHTypesInfo inputTypes() { return CoreInfo::AudioType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Accepts an audio buffer as input."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BytesType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input audio buffer represented as a byte array."); }

  static SHOptionalString help() { return SHCCSTR("Converts an audio buffer into a byte array."); }

  std::vector<uint8_t> _output;

  size_t audioDeriveDataLength(const SHAudio &audio) { return audio.nsamples * audio.channels * sizeof(float); }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &audio = input.payload.audioValue;
    uint32_t audioDataLength = audioDeriveDataLength(audio);
    _output.resize(audioDataLength);
    memcpy(_output.data(), audio.samples, audioDataLength);
    return Var(_output.data(), audioDataLength);
  }
};

SHARDS_REGISTER_FN(casting) {
  REGISTER_SHARD("ToInt", ToNumber<SHType::Int>);
  REGISTER_SHARD("ToInt2", ToNumber<SHType::Int2>);
  REGISTER_SHARD("ToInt3", ToNumber<SHType::Int3>);
  REGISTER_SHARD("ToInt4", ToNumber<SHType::Int4>);
  REGISTER_SHARD("ToInt8", ToNumber<SHType::Int8>);
  REGISTER_SHARD("ToInt16", ToNumber<SHType::Int16>);
  REGISTER_SHARD("ToColor", ToNumber<SHType::Color>);
  REGISTER_SHARD("ToFloat", ToNumber<SHType::Float>);
  REGISTER_SHARD("ToFloat2", ToNumber<SHType::Float2>);
  REGISTER_SHARD("ToFloat3", ToNumber<SHType::Float3>);
  REGISTER_SHARD("ToFloat4", ToNumber<SHType::Float4>);

  REGISTER_SHARD("MakeInt2", MakeVector<SHType::Int2>);
  REGISTER_SHARD("MakeInt3", MakeVector<SHType::Int3>);
  REGISTER_SHARD("MakeInt4", MakeVector<SHType::Int4>);
  REGISTER_SHARD("MakeInt8", MakeVector<SHType::Int8>);
  REGISTER_SHARD("MakeInt16", MakeVector<SHType::Int16>);
  REGISTER_SHARD("MakeColor", MakeVector<SHType::Color>);
  REGISTER_SHARD("MakeFloat2", MakeVector<SHType::Float2>);
  REGISTER_SHARD("MakeFloat3", MakeVector<SHType::Float3>);
  REGISTER_SHARD("MakeFloat4", MakeVector<SHType::Float4>);

  REGISTER_SHARD("ToString", ToString);
  REGISTER_SHARD("ToHex", ToHex);
  REGISTER_SHARD("ToAny", ToAny);
  REGISTER_SHARD("VarAddr", VarAddr);
  REGISTER_SHARD("BitSwap32", BitSwap32);
  REGISTER_SHARD("BitSwap64", BitSwap64);

  REGISTER_SHARD("Expect", Expect);
  using ExpectNone = ExpectX<SHType::None>;
  REGISTER_SHARD("ExpectNone", ExpectNone);
  using ExpectInt = ExpectX<SHType::Int>;
  REGISTER_SHARD("ExpectInt", ExpectInt);
  using ExpectInt2 = ExpectX<SHType::Int2>;
  REGISTER_SHARD("ExpectInt2", ExpectInt2);
  using ExpectInt3 = ExpectX<SHType::Int3>;
  REGISTER_SHARD("ExpectInt3", ExpectInt3);
  using ExpectInt4 = ExpectX<SHType::Int4>;
  REGISTER_SHARD("ExpectInt4", ExpectInt4);
  using ExpectInt8 = ExpectX<SHType::Int8>;
  REGISTER_SHARD("ExpectInt8", ExpectInt8);
  using ExpectInt16 = ExpectX<SHType::Int16>;
  REGISTER_SHARD("ExpectInt16", ExpectInt16);
  using ExpectFloat = ExpectX<SHType::Float>;
  REGISTER_SHARD("ExpectFloat", ExpectFloat);
  using ExpectFloat2 = ExpectX<SHType::Float2>;
  REGISTER_SHARD("ExpectFloat2", ExpectFloat2);
  using ExpectFloat3 = ExpectX<SHType::Float3>;
  REGISTER_SHARD("ExpectFloat3", ExpectFloat3);
  using ExpectFloat4 = ExpectX<SHType::Float4>;
  REGISTER_SHARD("ExpectFloat4", ExpectFloat4);
  using ExpectBytes = ExpectX<SHType::Bytes>;
  REGISTER_SHARD("ExpectBytes", ExpectBytes);
  using ExpectString = ExpectX<SHType::String>;
  REGISTER_SHARD("ExpectString", ExpectString);
  using ExpectImage = ExpectX<SHType::Image>;
  REGISTER_SHARD("ExpectImage", ExpectImage);
  using ExpectBool = ExpectX<SHType::Bool>;
  REGISTER_SHARD("ExpectBool", ExpectBool);
  using ExpectColor = ExpectX<SHType::Color>;
  REGISTER_SHARD("ExpectColor", ExpectColor);
  using ExpectWire = ExpectX<SHType::Wire>;
  REGISTER_SHARD("ExpectWire", ExpectWire);
  using ExpectTable = ExpectX<SHType::Table>;
  REGISTER_SHARD("ExpectTable", ExpectTable);
  using ExpectAudio = ExpectX<SHType::Audio>;
  REGISTER_SHARD("ExpectAudio", ExpectAudio);
  REGISTER_SHARD("ExpectSeq", ExpectSeq);

  using ExpectFloatSeq = ExpectXComplex<CoreInfo::FloatSeqType>;
  REGISTER_SHARD("ExpectFloatSeq", ExpectFloatSeq);
  using ExpectFloat2Seq = ExpectXComplex<CoreInfo::Float2SeqType>;
  REGISTER_SHARD("ExpectFloat2Seq", ExpectFloat2Seq);
  using ExpectFloat3Seq = ExpectXComplex<CoreInfo::Float3SeqType>;
  REGISTER_SHARD("ExpectFloat3Seq", ExpectFloat3Seq);
  using ExpectFloat4Seq = ExpectXComplex<CoreInfo::Float4SeqType>;
  REGISTER_SHARD("ExpectFloat4Seq", ExpectFloat4Seq);
  using ExpectIntSeq = ExpectXComplex<CoreInfo::IntSeqType>;
  REGISTER_SHARD("ExpectIntSeq", ExpectIntSeq);
  using ExpectInt2Seq = ExpectXComplex<CoreInfo::Int2SeqType>;
  REGISTER_SHARD("ExpectInt2Seq", ExpectInt2Seq);
  using ExpectInt3Seq = ExpectXComplex<CoreInfo::Int3SeqType>;
  REGISTER_SHARD("ExpectInt3Seq", ExpectInt3Seq);
  using ExpectInt4Seq = ExpectXComplex<CoreInfo::Int4SeqType>;
  REGISTER_SHARD("ExpectInt4Seq", ExpectInt4Seq);
  using ExpectInt8Seq = ExpectXComplex<CoreInfo::Int8SeqType>;
  REGISTER_SHARD("ExpectInt8Seq", ExpectInt8Seq);
  using ExpectInt16Seq = ExpectXComplex<CoreInfo::Int16SeqType>;
  REGISTER_SHARD("ExpectInt16Seq", ExpectInt16Seq);
  using ExpectBytesSeq = ExpectXComplex<CoreInfo::BytesSeqType>;
  REGISTER_SHARD("ExpectBytesSeq", ExpectBytesSeq);
  using ExpectStringSeq = ExpectXComplex<CoreInfo::StringSeqType>;
  REGISTER_SHARD("ExpectStringSeq", ExpectStringSeq);
  using ExpectImageSeq = ExpectXComplex<CoreInfo::ImageSeqType>;
  REGISTER_SHARD("ExpectImageSeq", ExpectImageSeq);
  using ExpectBoolSeq = ExpectXComplex<CoreInfo::BoolSeqType>;
  REGISTER_SHARD("ExpectBoolSeq", ExpectBoolSeq);
  using ExpectColorSeq = ExpectXComplex<CoreInfo::ColorSeqType>;
  REGISTER_SHARD("ExpectColorSeq", ExpectColorSeq);
  using ExpectWireSeq = ExpectXComplex<CoreInfo::WireSeqType>;
  REGISTER_SHARD("ExpectWireSeq", ExpectWireSeq);
  using ExpectAudioSeq = ExpectXComplex<CoreInfo::AudioSeqType>;
  REGISTER_SHARD("ExpectAudioSeq", ExpectAudioSeq);

  REGISTER_SHARD("ExpectLike", ExpectLike);
  REGISTER_SHARD("TypeOf", TypeOf);

  // IsNone is implemented in Core
  using IsInt = IsX<SHType::Int>;
  REGISTER_SHARD("IsInt", IsInt);
  using IsInt2 = IsX<SHType::Int2>;
  REGISTER_SHARD("IsInt2", IsInt2);
  using IsInt3 = IsX<SHType::Int3>;
  REGISTER_SHARD("IsInt3", IsInt3);
  using IsInt4 = IsX<SHType::Int4>;
  REGISTER_SHARD("IsInt4", IsInt4);
  using IsInt8 = IsX<SHType::Int8>;
  REGISTER_SHARD("IsInt8", IsInt8);
  using IsInt16 = IsX<SHType::Int16>;
  REGISTER_SHARD("IsInt16", IsInt16);
  using IsFloat = IsX<SHType::Float>;
  REGISTER_SHARD("IsFloat", IsFloat);
  using IsFloat2 = IsX<SHType::Float2>;
  REGISTER_SHARD("IsFloat2", IsFloat2);
  using IsFloat3 = IsX<SHType::Float3>;
  REGISTER_SHARD("IsFloat3", IsFloat3);
  using IsFloat4 = IsX<SHType::Float4>;
  REGISTER_SHARD("IsFloat4", IsFloat4);
  using IsBytes = IsX<SHType::Bytes>;
  REGISTER_SHARD("IsBytes", IsBytes);
  using IsString = IsX<SHType::String>;
  REGISTER_SHARD("IsString", IsString);
  using IsImage = IsX<SHType::Image>;
  REGISTER_SHARD("IsImage", IsImage);
  using IsBool = IsX<SHType::Bool>;
  REGISTER_SHARD("IsBool", IsBool);
  using IsColor = IsX<SHType::Color>;
  REGISTER_SHARD("IsColor", IsColor);
  using IsSeq = IsX<SHType::Seq>;
  REGISTER_SHARD("IsSeq", IsSeq);
  using IsWire = IsX<SHType::Wire>;
  REGISTER_SHARD("IsWire", IsWire);
  using IsTable = IsX<SHType::Table>;
  REGISTER_SHARD("IsTable", IsTable);
  using IsAudio = IsX<SHType::Audio>;
  REGISTER_SHARD("IsAudio", IsAudio);

  using ImageToFloats = ToSeq<SHType::Image, SHType::Float>;
  using FloatsToImage = ToImage<SHType::Float>;
  REGISTER_SHARD("ImageToFloats", ImageToFloats);
  REGISTER_SHARD("FloatsToImage", FloatsToImage);

  using BytesToInts = ToSeq<SHType::Bytes, SHType::Int>;
  using BytesToString = ToString1<SHType::Bytes>;
  using IntsToBytes = ToBytes<SHType::Int>;
  REGISTER_SHARD("BytesToInts", BytesToInts);
  REGISTER_SHARD("BytesToString", BytesToString);
  REGISTER_SHARD("IntsToBytes", IntsToBytes);
  REGISTER_SHARD("StringToBytes", StringToBytes);
  REGISTER_SHARD("ImageToBytes", ImageToBytes);
  REGISTER_SHARD("AudioToBytes", AudioToBytes);

  REGISTER_SHARD("ToBase64", ToBase64);
  REGISTER_SHARD("FromBase64", FromBase64);
  REGISTER_SHARD("HexToBytes", HexToBytes);

  REGISTER_SHARD("VarPtr", VarPtr);
}
}; // namespace shards
