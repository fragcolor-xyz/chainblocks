#ifndef SH_CORE_BLOCKS_CASTING
#define SH_CORE_BLOCKS_CASTING

#include "number_types.hpp"
#include "shared.hpp"

namespace shards {
template <SHType ToType> struct ToNumber {
  const VectorTypeTraits *_outputVectorType{nullptr};
  const NumberTypeTraits *_outputNumberType{nullptr};

  // If known at compose time these are set
  // otherwise each element should be evaluated during activate
  const VectorTypeTraits *_inputVectorType{nullptr};
  const NumberConversion *_numberConversion{nullptr};

  SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  SHTypesInfo outputTypes() {
    if (!_outputVectorType)
      return CoreInfo::AnyType;
    return _outputVectorType->type;
  }

  static const NumberTypeTraits *getEnumNumberType() {
    static const NumberTypeTraits *result = nullptr;
    if (!result) {
      result = NumberTypeLookup::getInstance().get(NumberType::Int32);
      shassert(result);
    }
    return result;
  }

  const NumberTypeTraits *determineFixedSeqNumberType(const SHTypeInfo &typeInfo) {
    if (!typeInfo.fixedSize) {
      return nullptr;
    }

    const NumberTypeTraits *fixedElemNumberType = nullptr;
    for (size_t i = 0; i < typeInfo.seqTypes.len; i++) {
      const SHTypeInfo &elementType = typeInfo.seqTypes.elements[i];
      const NumberTypeTraits *currentElemNumberType = NumberTypeLookup::getInstance().get(elementType.basicType);
      if (!currentElemNumberType)
        return nullptr;
      if (i == 0) {
        fixedElemNumberType = currentElemNumberType;
      } else {
        if (currentElemNumberType != fixedElemNumberType) {
          return nullptr;
        }
      }
    }

    return fixedElemNumberType;
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    _outputVectorType = VectorTypeLookup::getInstance().get(ToType);
    if (!_outputVectorType) {
      throw ComposeError("Conversion not implemented for this type");
    }

    _outputNumberType = NumberTypeLookup::getInstance().get(_outputVectorType->numberType);
    _inputVectorType = VectorTypeLookup::getInstance().get(data.inputType.basicType);

    if (_inputVectorType) {
      _numberConversion =
          NumberTypeLookup::getInstance().getConversion(_inputVectorType->numberType, _outputVectorType->numberType);
      shassert(_numberConversion);
    } else if (data.inputType.basicType == Seq) {
      const NumberTypeTraits *fixedNumberType = determineFixedSeqNumberType(data.inputType);
      if (fixedNumberType) {
        _numberConversion = fixedNumberType->conversionTable.get(_outputNumberType->type);
        shassert(_numberConversion);

        OVERRIDE_ACTIVATE(data, activateSeqElementsFixed);
      }
    }

    return _outputVectorType->type;
  }

  void parseVector(SHVar &output, const SHVar &input) {
    const VectorTypeTraits *inputVectorType =
        _inputVectorType ? _inputVectorType : VectorTypeLookup::getInstance().get(input.valueType);
    if (!inputVectorType) {
      throw ActivationError("Cannot cast given input type.");
    }

    const NumberConversion *conversion =
        _numberConversion
            ? _numberConversion
            : NumberTypeLookup::getInstance().getConversion(inputVectorType->numberType, _outputVectorType->numberType);
    shassert(conversion);

    // Can reduce call overhead by adding a single ConvertMultiple function with
    // in/out lengths
    uint8_t *srcPtr = (uint8_t *)&input.payload;
    uint8_t *dstPtr = (uint8_t *)&output.payload;
    size_t numToConvert = std::min(_outputVectorType->dimension, inputVectorType->dimension);
    for (size_t i = 0; i < numToConvert; i++) {
      conversion->convertOne(srcPtr, dstPtr);
      srcPtr += conversion->inStride;
      dstPtr += conversion->outStride;
    }
  }

  void parseSeqElements(SHVar &output, const SHSeq &sequence) {
    uint8_t *dstPtr = (uint8_t *)&output.payload;
    for (size_t i = 0; i < sequence.len; i++) {
      const SHVar &elem = sequence.elements[i];

      const NumberTypeTraits *elemNumberType = NumberTypeLookup::getInstance().get(elem.valueType);
      if (!elemNumberType) {
        throw ActivationError("Cannot cast given sequence element type.");
      }

      const NumberConversion *conversion = elemNumberType->conversionTable.get(_outputVectorType->numberType);
      shassert(conversion);

      conversion->convertOne(&elem.payload, dstPtr);
      dstPtr += conversion->outStride;
    }
  }

  void skipStringSeparators(const char *&strPtr) {
    while (*strPtr && (std::isspace(strPtr[0]) || strPtr[0] == ',' || strPtr[0] == ';')) {
      ++strPtr;
    }
  }

  void parseStringElements(SHVar &output, const char *str, size_t length) {
    uint8_t *dstPtr = (uint8_t *)&output.payload;
    const char *strPtr = str;

    // Skip seq header
    while (*strPtr && (std::isspace(strPtr[0]) || strPtr[0] == '(' || strPtr[0] == '{' || strPtr[0] == '[')) {
      ++strPtr;
    }

    for (size_t i = 0; i < _outputVectorType->dimension; i++) {
      _outputNumberType->convertParse(dstPtr, strPtr, (char **)&strPtr);
      skipStringSeparators(strPtr);
      dstPtr += _outputNumberType->size;
    }
  }

  void parseEnumValue(SHVar &output, SHEnum value) {
    _numberConversion = getEnumNumberType()->conversionTable.get(_outputVectorType->numberType);
    shassert(_numberConversion);

    _numberConversion->convertOne(&value, &output.payload);
  }

  SHVar activateSeqElementsFixed(SHContext *context, const SHVar &input) {
    shassert(_outputVectorType);
    shassert(_numberConversion);

    const SHSeq &sequence = input.payload.seqValue;
    SHVar output{};
    output.valueType = _outputVectorType->shType;

    uint8_t *dstPtr = (uint8_t *)&output.payload;
    for (size_t i = 0; i < sequence.len; i++) {
      const SHVar &elem = sequence.elements[i];

      _numberConversion->convertOne(&elem.payload, dstPtr);
      dstPtr += _numberConversion->outStride;
    }

    return output;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    shassert(_outputVectorType);

    SHVar output{};
    output.valueType = _outputVectorType->shType;

    switch (input.valueType) {
    case Seq:
      parseSeqElements(output, input.payload.seqValue);
      break;
    case Enum:
      parseEnumValue(output, input.payload.enumValue);
      break;
    case String:
      parseStringElements(output, input.payload.stringValue, input.payload.stringLen);
      break;
    default:
      parseVector(output, input);
      break;
    }

    return output;
  }
};

template <SHType ToType> struct MakeVector {
  const VectorTypeTraits *_outputVectorType{nullptr};
  const NumberTypeTraits *_outputNumberType{nullptr};
  const NumberConversion *_componentConversion{nullptr};

  std::vector<ParamVar> params;

  MakeVector() {
    auto vectorType = VectorTypeLookup::getInstance().get(ToType);
    params.resize(vectorType->dimension);
  }

  SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  SHTypesInfo outputTypes() {
    if (!_outputVectorType)
      return CoreInfo::AnyType;
    return _outputVectorType->type;
  }

  void setParam(int index, const SHVar &value) {
    if (index >= params.size())
      return;
    params[index] = value;
  }

  SHVar getParam(int index) {
    if (index >= params.size())
      return Var::Empty;
    return params[index];
  }

  static inline std::vector<std::string> componentNames = []() {
    std::vector<std::string> result;
    for (size_t i = 0; i < 16; i++) {
      result.push_back(fmt::format("{}", i));
    }
    return result;
  }();

  static SHParametersInfo parameters() {
    static Parameters params = []() {
      Parameters params;
      auto vectorType = VectorTypeLookup::getInstance().get(ToType);
      auto paramType = getCompatibleUnitType(vectorType->numberType);

      for (size_t i = 0; i < vectorType->dimension; i++) {
        params._infos.emplace_back(componentNames[i].c_str(), Types({Type::VariableOf(paramType), paramType}));
      }

      for (auto &param : params._infos)
        params._pinfos.emplace_back(param);

      return params;
    }();

    return params;
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    _outputVectorType = VectorTypeLookup::getInstance().get(ToType);
    if (!_outputVectorType) {
      throw ComposeError("Conversion not implemented for this type");
    }

    auto &numberTypeLookup = NumberTypeLookup::getInstance();
    _outputNumberType = numberTypeLookup.get(_outputVectorType->numberType);

    NumberType componentNumberType = _outputNumberType->isInteger ? NumberType::Int64 : NumberType::Float64;
    _componentConversion = numberTypeLookup.getConversion(componentNumberType, _outputNumberType->type);

    return _outputVectorType->type;
  }

  void warmup(SHContext *context) {
    for (auto &param : params)
      param.warmup(context);
  }

  void cleanup() {
    for (auto &param : params)
      param.cleanup();
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    shassert(_outputVectorType);

    SHVar output{};
    output.valueType = _outputVectorType->shType;

    uint8_t *dstData = (uint8_t *)&output.payload.floatValue;
    for (size_t i = 0; i < _outputVectorType->dimension; i++) {
      auto &param = params[i].get();
      _componentConversion->convertOne(&param.payload.floatValue, dstData);
      dstData += _componentConversion->outStride;
    }

    return output;
  }
};

} // namespace shards
#endif // SH_CORE_BLOCKS_CASTING
