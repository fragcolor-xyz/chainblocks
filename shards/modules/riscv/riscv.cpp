#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>

#include <libriscv/machine.hpp>
#include <libriscv/prepared_call.hpp>

namespace shards {
namespace shriscv {
struct Types {
  static constexpr uint32_t VendorId = shards::CoreCC;

#define OBJECT(_id, _displayName, _definedAs, ...)                                                                              \
  static constexpr uint32_t SH_CONCAT(_definedAs, TypeId) = uint32_t(_id);                                                      \
  static inline Type _definedAs{{SHType::Object, {.object = {.vendorId = VendorId, .typeId = SH_CONCAT(_definedAs, TypeId)}}}}; \
  static inline Type _definedAs##Var = Type::VariableOf(_definedAs);                                                            \
  static inline ObjectVar<__VA_ARGS__> SH_CONCAT(_definedAs, ObjectVar){_displayName, VendorId, SH_CONCAT(_definedAs, TypeId)};

  using RiscVMachinePtr = std::shared_ptr<riscv::Machine<riscv::RISCV64>>;
  OBJECT('risc', "RISC-V.Machine", RiscVMachine, RiscVMachinePtr);
};

struct RiscVModule {
  Types::RiscVMachinePtr *_machine;
  SHVar _programHash;

  static SHTypesInfo inputTypes() { return shards::CoreInfo::BytesType; }
  static SHTypesInfo outputTypes() { return Types::RiscVMachine; }

  static SHOptionalString help() { return SHCCSTR("Creates a RISC-V machine from the given binary compiled program bytes."); }

  PARAM_VAR(_enableFS, "Filesystem", "Enable filesystem access.", {CoreInfo::BoolType});
  PARAM_VAR(_enableNet, "Network", "Enable network access.", {CoreInfo::BoolType});
  PARAM_PARAMVAR(_args, "Args", "The arguments to pass to the program.",
                 {CoreInfo::NoneType, CoreInfo::StringSeqType, CoreInfo::StringVarSeqType});
  PARAM_PARAMVAR(_env, "Env", "The environment variables to pass to the program.",
                 {CoreInfo::NoneType, CoreInfo::StringSeqType, CoreInfo::StringVarSeqType});
  PARAM_IMPL(PARAM_IMPL_FOR(_enableFS), PARAM_IMPL_FOR(_enableNet), PARAM_IMPL_FOR(_args), PARAM_IMPL_FOR(_env));

  RiscVModule() {
    _enableFS = Var(false);
    _enableNet = Var(false);
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) {
    if (_machine) {
      Types::RiscVMachineObjectVar.Release(_machine);
      _machine = nullptr;
    }
    PARAM_CLEANUP(context);
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto hash = shards::hash(input);
    if (_programHash != hash) {
      _programHash = hash;

      _machine = Types::RiscVMachineObjectVar.New();
      std::string_view program(reinterpret_cast<const char *>(input.payload.bytesValue), input.payload.bytesSize);
      *_machine = std::make_shared<riscv::Machine<riscv::RISCV64>>(program);

      std::vector<std::string> args;
      if (_args.get().valueType != SHType::None) {
        auto &argsSeq = asSeq(_args.get());
        for (auto &arg : argsSeq) {
          auto s = SHSTRVIEW(arg);
          args.emplace_back(s);
        }
      }
      std::vector<std::string> env;
      if (_env.get().valueType != SHType::None) {
        auto &envSeq = asSeq(_env.get());
        for (auto &envVar : envSeq) {
          auto s = SHSTRVIEW(envVar);
          env.emplace_back(s);
        }
      }
      (*_machine)->setup_linux(args, env);

      (*_machine)->setup_linux_syscalls(_enableFS.payload.boolValue, _enableNet.payload.boolValue);
    }
    return Types::RiscVMachineObjectVar.Get(_machine);
  }
};

struct RiscVSimulate {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }  // passthrough
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; } // passthrough
  static SHOptionalString help() { return SHCCSTR("Simulates the RISC-V machine."); }

  PARAM_PARAMVAR(_machine, "Machine", "The RISC-V machine to simulate.", {Types::RiscVMachine, Types::RiscVMachineVar});
  PARAM_PARAMVAR(_steps, "Instructions", "The optional number of instructions to simulate.",
                 {CoreInfo::NoneType, CoreInfo::IntType});
  PARAM_IMPL(PARAM_IMPL_FOR(_machine), PARAM_IMPL_FOR(_steps));

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &machine = varAsObjectChecked<Types::RiscVMachinePtr>(_machine.get(), Types::RiscVMachine);
    if (_steps.isNone())
      machine->simulate();
    else
      machine->simulate(int32_t(_steps.get().payload.intValue));
    return input;
  }
};

struct RiscVCall {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString help() { return SHCCSTR("Calls a function in the RISC-V machine."); }

  PARAM_PARAMVAR(_machine, "Machine", "The RISC-V machine to call the function in.",
                 {Types::RiscVMachine, Types::RiscVMachineVar});
  PARAM_VAR(_func, "Function",
            "The function to call. (has to be exported and with signature int32_t(const SHVar *input, SHVar *output))",
            {CoreInfo::StringType});
  PARAM_IMPL(PARAM_IMPL_FOR(_machine), PARAM_IMPL_FOR(_func));

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) {
    _call.reset();
    _machinePtr = nullptr;
    PARAM_CLEANUP(context);
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  std::optional<riscv::PreparedCall<riscv::RISCV64, int32_t(const SHVar *, SHVar *)>> _call;
  void *_machinePtr{nullptr};

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &machine = varAsObjectChecked<Types::RiscVMachinePtr>(_machine.get(), Types::RiscVMachine);
    if (_machinePtr != machine.get()) {
      _machinePtr = machine.get();
      auto func = SHSTRING_PREFER_SHSTRVIEW(_func);
      _call = riscv::PreparedCall<riscv::RISCV64, int32_t(const SHVar *, SHVar *)>(*machine, func);
    }

    SHVar output{};
    int res = (*_call)(&input, &output);
    if (res != 0) {
      throw shards::ActivationError(fmt::format("Failed to call function '{}' error: {}", _func, res));
    }

    return output;
  }
};

} // namespace shriscv
} // namespace shards

SHARDS_REGISTER_FN(shriscv) {
  REGISTER_SHARD("RISC-V.Load", shards::shriscv::RiscVModule);
  REGISTER_SHARD("RISC-V.Simulate", shards::shriscv::RiscVSimulate);
  REGISTER_SHARD("RISC-V.Call", shards::shriscv::RiscVCall);
}
