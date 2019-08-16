#ifndef CHAINBLOCKS_RUNTIME
#define CHAINBLOCKS_RUNTIME 1
#endif

#include "../runtime.hpp"

namespace chainblocks {
// Register Const
RUNTIME_CORE_BLOCK_FACTORY(Const);
RUNTIME_BLOCK_destroy(Const);
RUNTIME_BLOCK_inputTypes(Const);
RUNTIME_BLOCK_outputTypes(Const);
RUNTIME_BLOCK_parameters(Const);
RUNTIME_BLOCK_inferTypes(Const);
RUNTIME_BLOCK_setParam(Const);
RUNTIME_BLOCK_getParam(Const);
RUNTIME_BLOCK_activate(Const);
RUNTIME_BLOCK_END(Const);

// Register Sleep
RUNTIME_CORE_BLOCK_FACTORY(Sleep);
RUNTIME_BLOCK_inputTypes(Sleep);
RUNTIME_BLOCK_outputTypes(Sleep);
RUNTIME_BLOCK_parameters(Sleep);
RUNTIME_BLOCK_setParam(Sleep);
RUNTIME_BLOCK_getParam(Sleep);
RUNTIME_BLOCK_activate(Sleep);
RUNTIME_BLOCK_END(Sleep);

// Register Stop
RUNTIME_CORE_BLOCK_FACTORY(Stop);
RUNTIME_BLOCK_inputTypes(Stop);
RUNTIME_BLOCK_outputTypes(Stop);
RUNTIME_BLOCK_activate(Stop);
RUNTIME_BLOCK_END(Stop);

// Register Restart
RUNTIME_CORE_BLOCK_FACTORY(Restart);
RUNTIME_BLOCK_inputTypes(Restart);
RUNTIME_BLOCK_outputTypes(Restart);
RUNTIME_BLOCK_activate(Restart);
RUNTIME_BLOCK_END(Restart);

// Register Set
RUNTIME_CORE_BLOCK_FACTORY(Set);
RUNTIME_BLOCK_cleanup(Set);
RUNTIME_BLOCK_inputTypes(Set);
RUNTIME_BLOCK_outputTypes(Set);
RUNTIME_BLOCK_parameters(Set);
RUNTIME_BLOCK_inferTypes(Set);
RUNTIME_BLOCK_exposedVariables(Set);
RUNTIME_BLOCK_setParam(Set);
RUNTIME_BLOCK_getParam(Set);
RUNTIME_BLOCK_activate(Set);
RUNTIME_BLOCK_END(Set);

// Register Push
RUNTIME_CORE_BLOCK_FACTORY(Push);
RUNTIME_BLOCK_cleanup(Push);
RUNTIME_BLOCK_inputTypes(Push);
RUNTIME_BLOCK_outputTypes(Push);
RUNTIME_BLOCK_parameters(Push);
RUNTIME_BLOCK_inferTypes(Push);
RUNTIME_BLOCK_consumedVariables(Push);
RUNTIME_BLOCK_setParam(Push);
RUNTIME_BLOCK_getParam(Push);
RUNTIME_BLOCK_activate(Push);
RUNTIME_BLOCK_END(Push);

// Register Get
RUNTIME_CORE_BLOCK_FACTORY(Get);
RUNTIME_BLOCK_cleanup(Get);
RUNTIME_BLOCK_inputTypes(Get);
RUNTIME_BLOCK_outputTypes(Get);
RUNTIME_BLOCK_parameters(Get);
RUNTIME_BLOCK_inferTypes(Get);
RUNTIME_BLOCK_consumedVariables(Get);
RUNTIME_BLOCK_setParam(Get);
RUNTIME_BLOCK_getParam(Get);
RUNTIME_BLOCK_activate(Get);
RUNTIME_BLOCK_END(Get);

// Register Swap
RUNTIME_CORE_BLOCK_FACTORY(Swap);
RUNTIME_BLOCK_cleanup(Swap);
RUNTIME_BLOCK_inputTypes(Swap);
RUNTIME_BLOCK_outputTypes(Swap);
RUNTIME_BLOCK_parameters(Swap);
RUNTIME_BLOCK_consumedVariables(Swap);
RUNTIME_BLOCK_setParam(Swap);
RUNTIME_BLOCK_getParam(Swap);
RUNTIME_BLOCK_activate(Swap);
RUNTIME_BLOCK_END(Swap);

// Register Take
RUNTIME_CORE_BLOCK_FACTORY(Take);
RUNTIME_BLOCK_destroy(Take);
RUNTIME_BLOCK_inputTypes(Take);
RUNTIME_BLOCK_outputTypes(Take);
RUNTIME_BLOCK_parameters(Take);
RUNTIME_BLOCK_inferTypes(Take);
RUNTIME_BLOCK_setParam(Take);
RUNTIME_BLOCK_getParam(Take);
RUNTIME_BLOCK_activate(Take);
RUNTIME_BLOCK_END(Take);

// Register Take
RUNTIME_CORE_BLOCK_FACTORY(Repeat);
RUNTIME_BLOCK_inputTypes(Repeat);
RUNTIME_BLOCK_outputTypes(Repeat);
RUNTIME_BLOCK_parameters(Repeat);
RUNTIME_BLOCK_setParam(Repeat);
RUNTIME_BLOCK_getParam(Repeat);
RUNTIME_BLOCK_activate(Repeat);
RUNTIME_BLOCK_END(Repeat);

namespace Math {
MATH_BLOCK(Add);
MATH_BLOCK(Subtract);
MATH_BLOCK(Multiply);
MATH_BLOCK(Divide);
MATH_BLOCK(Xor);
MATH_BLOCK(And);
MATH_BLOCK(Or);
MATH_BLOCK(Mod);
MATH_BLOCK(LShift);
MATH_BLOCK(RShift);
}; // namespace Math

void registerBlocksCoreBlocks() {
  REGISTER_CORE_BLOCK(Const);
  REGISTER_CORE_BLOCK(Set);
  REGISTER_CORE_BLOCK(Push);
  REGISTER_CORE_BLOCK(Get);
  REGISTER_CORE_BLOCK(Swap);
  REGISTER_CORE_BLOCK(Sleep);
  REGISTER_CORE_BLOCK(Restart);
  REGISTER_CORE_BLOCK(Stop);
  REGISTER_CORE_BLOCK(Take);
  REGISTER_CORE_BLOCK(Repeat);

  REGISTER_BLOCK(Math, Add);
  REGISTER_BLOCK(Math, Subtract);
  REGISTER_BLOCK(Math, Multiply);
  REGISTER_BLOCK(Math, Divide);
  REGISTER_BLOCK(Math, Xor);
  REGISTER_BLOCK(Math, And);
  REGISTER_BLOCK(Math, Or);
  REGISTER_BLOCK(Math, Mod);
  REGISTER_BLOCK(Math, LShift);
  REGISTER_BLOCK(Math, RShift);
}
}; // namespace chainblocks