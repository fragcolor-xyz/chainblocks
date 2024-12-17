#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include <shards/core/runtime.hpp>

struct EvalEnv;

struct Sequence;

extern "C" {

void shards_init(SHCore *core);

SHLAst shards_read(SHStringWithLen name, SHStringWithLen code, SHStringWithLen base_path);

SHLAst shards_load_ast(uint8_t *bytes, uint32_t size);

SHVar shards_save_ast(Sequence *ast);

EvalEnv *shards_create_env(SHStringWithLen namespace_);

void shards_forbid_shard(EvalEnv *env, SHStringWithLen name);

void shards_free_env(EvalEnv *env);

EvalEnv *shards_create_sub_env(EvalEnv *env, SHStringWithLen namespace_);

SHLError *shards_eval_env(EvalEnv *env, Sequence *ast);

/// It will consume the env
SHLWire shards_transform_env(EvalEnv *env, SHStringWithLen name);

SHLWire shards_transform_envs(EvalEnv **env, uintptr_t len, SHStringWithLen name);

SHLWire shards_eval(Sequence *sequence, SHStringWithLen name);

void shards_free_sequence(Sequence *sequence);

void shards_free_wire(SHWireRef *wire);

void shards_free_error(SHLError *error);

/// Please note it will consume `from` but not `to`
SHLError *shards_merge_envs(EvalEnv *from, EvalEnv *to);

int32_t shards_process_args(int32_t argc, const char *const *argv, bool no_cancellation);

} // extern "C"
