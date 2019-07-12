type Math* = object

when true:
  template calculateBinaryOp(typeName: untyped, shortName: string, op: untyped): untyped =
    type
      typeName* = object
        # INLINE BLOCK, CORE STUB PRESENT
        operand*: CBVar
        seqCache*: CBSeq
    
    template setup*(b: typename) = initSeq(b.seqCache)
    template destroy*(b: typename) = freeSeq(b.seqCache)
    template inputTypes*(b: typeName): CBTypesInfo = ({ Int, Int2, Int3, Int4, Float, Float2, Float3, Float4, Color}, true)
    template outputTypes*(b: typeName): CBTypesInfo = ({ Int, Int2, Int3, Int4, Float, Float2, Float3, Float4, Color }, true)
    template parameters*(b: typeName): CBParametersInfo =
      @[("Operand", { Int, Int2, Int3, Int4, Float, Float2, Float3, Float4, Color })]
    template setParam*(b: typeName; index: int; val: CBVar) = b.operand = val
    template getParam*(b: typeName; index: int): CBVar = b.operand
    template activate*(b: typeName; context: CBContext; input: CBVar): CBVar =
      # THIS CODE WON'T BE EXECUTED
      # THIS BLOCK IS OPTIMIZED INLINE IN THE C++ CORE
      if input.valueType == Seq:
        b.seqCache.clear()
        for val in input.seqValue:
          b.seqCache.push(op(val, b.operand))
        b.seqCache.CBVar
      else:
        op(input, b.operand)
    
    when defined blocksTesting:
      var blk: typeName
      blk.setParam(0, 77.CBVar)
      let res = blk.activate(nil, 10.CBVar).intValue
      echo "10", astToStr(op), "77 = ", res
      assert res == op(10.int64, 77.int64)
    
    chainblock typeName, shortName, "Math"

  calculateBinaryOp(CBMathAdd,      "Add",      `+`)
  calculateBinaryOp(CBMathSubtract, "Subtract", `-`)
  calculateBinaryOp(CBMathMultiply, "Multiply", `*`)
  calculateBinaryOp(CBMathDivide,   "Divide",   `/`)
  calculateBinaryOp(CBMathXor,      "Xor",      `xor`)
  calculateBinaryOp(CBMathAnd,      "And",      `and`)
  calculateBinaryOp(CBMathOr,       "Or",       `or`)
  calculateBinaryOp(CBMathMod,      "Mod",      `mod`)
  calculateBinaryOp(CBMathLShift,   "LShift",   `shl`)
  calculateBinaryOp(CBMathRShift,   "RShift",   `shr`)