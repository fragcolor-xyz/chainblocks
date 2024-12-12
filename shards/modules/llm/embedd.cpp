#include <shards/core/module.hpp>
#include <shards/core/runtime.hpp>
#include <shards/shards.h>
#include <shards/core/shared.hpp>
#include <shards/utility.hpp>
#include <shards/core/params.hpp>

#include <llama.h>

namespace shards {
namespace llm {
struct ModelData {
  static inline std::atomic_uint32_t usageCounter;

  ModelData() {
    uint32_t expected = usageCounter.load(std::memory_order_acquire);
    uint32_t desired;
    do {
      desired = expected + 1;
    } while (!usageCounter.compare_exchange_weak(expected, desired, std::memory_order_release));

    if (desired == 1) {
      SHLOG_DEBUG("Initializing llama backend");
      llama_backend_init();
    }
  }

  ~ModelData() {
    uint32_t prev = usageCounter.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
      SHLOG_DEBUG("Freeing llama backend");
      llama_backend_free();
    }
  }

  std::shared_ptr<llama_model> model;
};

struct Model {
  Model() { _useMmap = Var(true); }

  static inline int32_t ObjectId = 'llam';
  static inline const char VariableName[] = "LLM.Model";
  static inline ::shards::Type Type = ::shards::Type::Object(CoreCC, ObjectId);
  static inline SHTypeInfo RawType = Type;
  static inline ::shards::Type VarType = ::shards::Type::VariableOf(Type);
  static inline shards::ObjectVar<ModelData> ObjectVar{VariableName, RawType.object.vendorId, RawType.object.typeId};

  ModelData *_data{};

  static SHTypesInfo inputTypes() { return shards::CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return Type; }

  PARAM_PARAMVAR(_useMmap, "UseMmap", "Use mmap to load the model", {shards::CoreInfo::BoolType, shards::CoreInfo::BoolVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_useMmap));

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    if (_data) {
      ObjectVar.Release(_data);
      _data = nullptr;
    }
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _data = ObjectVar.New();
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto path = SHSTRING_PREFER_SHSTRVIEW(input);

    auto params = llama_model_default_params();
    params.use_mmap = _useMmap.get().payload.boolValue;

    _data->model = std::shared_ptr<llama_model>(llama_load_model_from_file(path.c_str(), params), llama_free_model);

    return ObjectVar.Get(_data);
  }
};

struct Tokenize {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::IntSeqType; }

  PARAM_PARAMVAR(_model, "Model", "The model to use", {Model::VarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_model));

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);

    // free up all the memory explicitly
    _tokens = {};
    _tokensCache = {};
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SeqVar _tokens;
  std::vector<llama_token> _tokensCache;

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &data = varAsObjectChecked<ModelData>(_model.get(), Model::Type);
    auto model = data.model.get();

    auto text = SHSTRVIEW(input);

    _tokensCache.resize(text.size());
    auto nTokens = llama_tokenize(model, text.data(), text.size(), _tokensCache.data(), _tokensCache.size(), true, true);
    if (nTokens < 0) {
      throw ActivationError("Failed to tokenize input");
    }

    _tokens.clear();
    for (int i = 0; i < nTokens; i++) {
      _tokens.push_back(Var(_tokensCache[i]));
    }

    return _tokens;
  }
};

struct Detokenize {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::IntSeqType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::StringType; }

  PARAM_PARAMVAR(_model, "Model", "The model to use", {Model::VarType, Model::VarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_model));

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _text = {};
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  std::string _text;

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &data = varAsObjectChecked<ModelData>(_model.get(), Model::Type);
    auto model = data.model.get();

    std::vector<llama_token> tokens;
    for (const auto &token : input.payload.seqValue) {
      tokens.push_back(token.payload.intValue);
    }

    _text.resize(tokens.size() * 4); // Rough estimate for space needed
    auto result = llama_detokenize(model, tokens.data(), tokens.size(), _text.data(), _text.size(), true, false);
    if (result < 0) {
      throw ActivationError("Failed to detokenize input");
    }
    _text.resize(result);

    return Var(_text);
  }
};

struct ContextData {
  std::shared_ptr<llama_context> ctx;
  OwnedVar _modelData; // we need to keep a reference to the model data alive
  std::shared_ptr<std::mutex> _mutex;
};

struct Context {
  static inline int32_t ObjectId = 'llmc';
  static inline const char VariableName[] = "LLM.Context";
  static inline ::shards::Type Type = ::shards::Type::Object(CoreCC, ObjectId);
  static inline SHTypeInfo RawType = Type;
  static inline ::shards::Type VarType = ::shards::Type::VariableOf(Type);
  static inline shards::ObjectVar<ContextData> ObjectVar{VariableName, RawType.object.vendorId, RawType.object.typeId};

  static SHTypesInfo inputTypes() { return Model::Type; }
  static SHTypesInfo outputTypes() { return Type; }

  Context() { _embeddings = Var(false); }

  PARAM_PARAMVAR(_embeddings, "Embeddings", "Enable embeddings mode",
                 {shards::CoreInfo::BoolType, shards::CoreInfo::BoolVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_embeddings));

  ContextData *_data{};

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    if (_data) {
      ObjectVar.Release(_data);
      _data = nullptr;
    }
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _data = ObjectVar.New();
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &data = varAsObjectChecked<ModelData>(input, Model::Type);
    auto model = data.model.get();

    auto ctx_params = llama_context_default_params();
    ctx_params.embeddings = _embeddings.get().payload.boolValue;

    _data->ctx = std::shared_ptr<llama_context>(llama_new_context_with_model(model, ctx_params), llama_free);
    if (!_data->ctx) {
      throw ActivationError("Failed to create context");
    }
    _data->_modelData = input;
    _data->_mutex = std::make_shared<std::mutex>();
    return ObjectVar.Get(_data);
  }
};

struct Embed {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::IntSeqType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::FloatSeqType; }

  Embed() { _norm = Var(-1); }

  PARAM_PARAMVAR(_context, "Context", "The LLM context to use", {Context::VarType});
  PARAM_PARAMVAR(_norm, "Normalization", "Normalization type: -1=none, 0=max_abs, 2=euclidean, >2=p-norm",
                 {shards::CoreInfo::IntType});
  PARAM_IMPL(PARAM_IMPL_FOR(_context), PARAM_IMPL_FOR(_norm));

  static void normalize_embeddings(const float *inp, float *out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
    case -1: // no normalization
      sum = 1.0;
      break;
    case 0: // max absolute
      for (int i = 0; i < n; i++) {
        if (sum < std::abs(inp[i]))
          sum = std::abs(inp[i]);
      }
      sum /= 32760.0; // make an int16 range
      break;
    case 2: // euclidean
      for (int i = 0; i < n; i++) {
        sum += inp[i] * inp[i];
      }
      sum = std::sqrt(sum);
      break;
    default: // p-norm (euclidean is p-norm p=2)
      for (int i = 0; i < n; i++) {
        sum += std::pow(std::abs(inp[i]), embd_norm);
      }
      sum = std::pow(sum, 1.0 / embd_norm);
      break;
    }

    const float norm = sum > 0.0 ? 1.0f / sum : 0.0f;

    for (int i = 0; i < n; i++) {
      out[i] = inp[i] * norm;
    }
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _embeddings = {};
    _normalized_buffer = {};
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SeqVar _embeddings;
  std::vector<float> _normalized_buffer;

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &llmContext = varAsObjectChecked<ContextData>(_context.get(), Context::Type);

    std::lock_guard<std::mutex> lock(*llmContext._mutex);

    auto nUBatch = llama_n_ubatch(llmContext.ctx.get());
    auto model = llama_get_model(llmContext.ctx.get());
    auto nEmbd = llama_n_embd(model);
    auto batch = llama_batch_init(nUBatch, 0, 1);
    DEFER(llama_batch_free(batch));

    if (input.payload.seqValue.len > uint32_t(nUBatch)) {
      throw ActivationError(fmt::format("Input sequence is too long, maximum length is {}", nUBatch));
    }

    std::vector<llama_token> tokens;
    for (const auto &token : input.payload.seqValue) {
      tokens.push_back(token.payload.intValue);
    }

    for (size_t i = 0; i < tokens.size(); i++) {
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.seq_id[i][0] = 0;
      batch.n_seq_id[i] = 1;
      batch.logits[i] = true; // Enable logits for all tokens to get embeddings
    }
    batch.n_tokens = tokens.size();

    llama_kv_cache_clear(llmContext.ctx.get());

    if (llama_model_has_encoder(model)) {
      if (llama_encode(llmContext.ctx.get(), batch) < 0) {
        throw ActivationError("Failed to encode input");
      }
    } else {
      if (llama_decode(llmContext.ctx.get(), batch) < 0) {
        throw ActivationError("Failed to decode input");
      }
    }

    _embeddings.clear();
    _normalized_buffer.resize(nEmbd);

    auto pooling_type = llama_pooling_type(llmContext.ctx.get());
    std::set<int> processed_seq_ids;

    for (int i = 0; i < batch.n_tokens; i++) {
      if (!batch.logits[i]) {
        continue;
      }

      const float *embd = nullptr;

      if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        embd = llama_get_embeddings_ith(llmContext.ctx.get(), i);
        if (!embd) {
          throw ActivationError("Failed to get token embeddings for position " + std::to_string(i));
        }

        // Normalize and add token embedding
        normalize_embeddings(embd, _normalized_buffer.data(), nEmbd, _norm.get().payload.intValue);
        for (int j = 0; j < nEmbd; j++) {
          _embeddings.push_back(Var(_normalized_buffer[j]));
        }
      } else {
        // For sequence embeddings, only process each sequence ID once
        int seq_id = batch.seq_id[i][0];
        if (processed_seq_ids.find(seq_id) != processed_seq_ids.end()) {
          continue;
        }

        embd = llama_get_embeddings_seq(llmContext.ctx.get(), seq_id);
        if (!embd) {
          throw ActivationError("Failed to get sequence embeddings for seq_id " + std::to_string(seq_id));
        }

        // Normalize embeddings before adding them
        normalize_embeddings(embd, _normalized_buffer.data(), pooling_type == LLAMA_POOLING_TYPE_RANK ? 1 : nEmbd,
                             _norm.get().payload.intValue);

        if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
          _embeddings.push_back(Var(_normalized_buffer[0]));
        } else {
          for (int j = 0; j < nEmbd; j++) {
            _embeddings.push_back(Var(_normalized_buffer[j]));
          }
        }

        processed_seq_ids.insert(seq_id);
      }
    }

    if (_embeddings.empty()) {
      throw ActivationError("No embeddings were generated");
    }

    return _embeddings;
  }
};
} // namespace llm

SHARDS_REGISTER_FN(llm) {
  REGISTER_SHARD("LLM.Model", llm::Model);
  REGISTER_SHARD("LLM.Context", llm::Context);
  REGISTER_SHARD("LLM.Tokenize", llm::Tokenize);
  REGISTER_SHARD("LLM.Detokenize", llm::Detokenize);
  REGISTER_SHARD("LLM.Embed", llm::Embed);
}
}; // namespace shards
