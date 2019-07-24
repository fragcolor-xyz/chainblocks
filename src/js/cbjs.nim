import quicknimjs
import nimline
import config

importExtras()

var
  jsRuntime = newJSRuntime()
  jsContext = newJSContext(jsRuntime)

proc loadBuiltIns() =
  var echoProc = newProc(jsContext, proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
    for i in 0..<argc:
      echo argv[i].toStr(jsContext).getStr(jsContext)
    return Undefined
  , "echo", 1)
  jsContext.Global.setProperty(jsContext, "echo", echoProc)

  var sleepProc = newProc(jsContext, proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
    let sleepTime = argv[0].toFloat(jsContext)
    sleep(sleepTime)
    return Undefined
  , "sleep", 1)
  jsContext.Global.setProperty(jsContext, "sleep", sleepProc)
loadBuiltIns()

# Forwarding those here
var JsCBRuntimeBlock = JsClass(
  finalizer: proc(self: JSValue; classId: uint32) =
    var blk = cast[ptr CBRuntimeBlock](self.getPtr(classId))
    if blk != nil:
      blk[].cleanup(blk)
      blk[].destroy(blk),
  
  constructor: proc(self: var JSValue; argc: int; argv: ptr UncheckedArray[JSValue]) =
    let name = argv[0].getStr(jsContext)
    if name == nil:
      self = throwTypeError(jsContext, "Block constructor requires a name")
      return

    var blk = createBlock($name)
    if blk == nil:
      self = throwTypeError(jsContext, "Block construction failed, this block does not exist.")
      return
    
    blk[].setup(blk)
    self.attachPtr(blk),
  
  constructorMinArgs: 1
)
jsClass(jsContext, "Block", JsCBRuntimeBlock)

var JsCBChain = JsClass(
  finalizer: proc(self: JSValue; classId: uint32) =
    var chain = cast[CBChainPtr](self.getPtr(classId))
    if chain != nil:
      destroy(chain),
  
  constructor: proc(self: var JSValue; argc: int; argv: ptr UncheckedArray[JSValue]) =
    let name = argv[0].getStr(jsContext)
    if name == nil:
      self = throwTypeError(jsContext, "Chain constructor requires a name")
      return
    
    var chain = newChain($name)
    self.attachPtr(chain),
  
  constructorMinArgs: 1
)
jsClass(jsContext, "Chain", JsCBChain)

type 
  JsVarStorage {.pure.} = enum
    None,
    Copy
  
  JsCBVarBox = object
    value: CBVar
    storage: JsVarStorage

# A immutable CBVar
when true:
  var JsCBVarConst = JsClass(
    finalizer: proc(self: JSValue; classId: uint32) =
      discard,
    constructor: proc(self: var JSValue; argc: int; argv: ptr UncheckedArray[JSValue]) =
      discard,
    constructorMinArgs: 0
  )
  jsClass(jsContext, "ConstVar", JsCBVarConst)

  JsCBVarConst.jsClassGetSet(
    "valueType",
    proc(ctx: ptr JSContext; this_val: JSValue): JSValue {.cdecl.} =
      let
        pvar = cast[ptr CBVar](this_val.getPtr(JsCBVarConst.classId))
      
      if pvar != nil:
        return pvar[].valueType.int32.jsInt32(ctx)
      
      return Undefined
  )
  JsCBVarConst.jsClassCall(
    "toString",
    proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
      let
        pvar = cast[ptr CBVar](this_val.getPtr(JsCBVarConst.classId))
      
      if pvar != nil:
        var strVal = $pvar[]
        return newStr(strVal.cstring, jsContext)
      
      return Undefined,
  )

  proc toConstJSValue*(v: var CBVar; context: ptr JSContext): JSValue {.inline.} =
    result = newClassObj(context, JsCBVarConst.classId, JsCBVarConst.prototype)
    result.attachPtr(addr v)

# A mutable CBVar
when true:
  var JsCBVar = JsClass(
    finalizer: proc(self: JSValue; classId: uint32) =
      var cbvar = cast[ptr JsCBVarBox](self.getPtr(classId))
      if cbvar != nil:
        case cbvar.storage
        of JsVarStorage.None:
          discard
        of JsVarStorage.Copy:
          `~quickcopy` cbvar[].value
        
        cppdel(cbvar),
    
    constructor: proc(self: var JSValue; argc: int; argv: ptr UncheckedArray[JSValue]) =
      var newVar: ptr JsCBVarBox
      cppnew(newVar, JsCBVarBox, JsCBVarBox)
      
      # Add logic to init var from any value...
      if argc == 1:
        if argv[0].isObj:
          block findObj:
            let constVar = cast[ptr CBVar](argv[0].getPtr(JsCBVarConst.classId))
            if constVar != nil:
              newVar[].storage = JsVarStorage.Copy
              quickcopy(newVar[].value, constVar[])
              break
            
            let blockValue = cast[ptr CBRuntimeBlock](argv[0].getPtr(JsCBRuntimeBlock.classId))
            if blockValue != nil:
              newVar[].value.valueType = Block
              newVar[].value.blockValue = blockValue
              break
            
            let chainValue = cast[CBChainPtr](argv[0].getPtr(JsCBChain.classId))
            if blockValue != nil:
              newVar[].value.valueType = Chain
              newVar[].value.chainValue = chainValue
              break
        
        elif argv[0].isStr:
          let strVar = argv[0].getStr(jsContext)
          if strVar != nil:
            var strTemp: CBVar = strVar.CBString
            newVar[].storage = JsVarStorage.Copy
            quickcopy(newVar[].value, strTemp)
        
        elif argv[0].isInt:
          newVar[].value.valueType = Int
          newVar[].value.intValue = argv[0].toInt(jsContext)
        
        elif argv[0].isFloat:
          newVar[].value.valueType = Float
          newVar[].value.floatValue = argv[0].toFloat(jsContext)
      
      elif argc > 1:
        if argv[0].isInt: # infer from first arg
          case argc
          of 2:
            newVar[].value.valueType = Int2
            for i in 0..<argc: newVar[].value.int2Value[i] = argv[i].toInt(jsContext)
          of 3:
            newVar[].value.valueType = Int3
            for i in 0..<argc: newVar[].value.int3Value[i] = argv[i].toInt(jsContext)
          of 4:
            newVar[].value.valueType = Int4
            for i in 0..<argc: newVar[].value.int4Value[i] = argv[i].toInt(jsContext)
          of 8:
            newVar[].value.valueType = Int8
            for i in 0..<argc: newVar[].value.int8Value[i] = argv[i].toInt(jsContext)
          of 16:
            newVar[].value.valueType = Int16
            for i in 0..<argc: newVar[].value.int16Value[i] = argv[i].toInt(jsContext)
          else:
            self = throwTypeError(jsContext, "Int variable out of range, pass as an array instead.")
            return
        
        elif argv[0].isFloat: # infer from first arg
          case argc
          of 2:
            newVar[].value.valueType = Float2
            for i in 0..<argc: newVar[].value.float2Value[i] = argv[i].toFloat(jsContext)
          of 3:
            newVar[].value.valueType = Float3
            for i in 0..<argc: newVar[].value.float3Value[i] = argv[i].toFloat(jsContext)
          of 4:
            newVar[].value.valueType = Float4
            for i in 0..<argc: newVar[].value.float4Value[i] = argv[i].toFloat(jsContext)
          else:
            self = throwTypeError(jsContext, "Float variable out of range, pass as an array instead.")
            return
      
      self.attachPtr(newVar),
    
    constructorMinArgs: 0
  )
  jsClass(jsContext, "Var", JsCBVar)

  JsCBVar.jsClassCall(
    "toString",
    proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
      let
        pvar = cast[ptr CBVar](this_val.getPtr(JsCBVar.classId))
      
      if pvar != nil:
        var strVal = $pvar[]
        return newStr(strVal.cstring, jsContext)
      
      return Undefined,
  )

  proc toJSValue*(v: var CBVar; context: ptr JSContext): JSValue {.inline.} =
    result = newClassObj(context, JsCBVar.classId, JsCBVar.prototype)
    var newVar: ptr JsCBVarBox
    cppnew(newVar, JsCBVarBox, JsCBVarBox)
    newVar[].storage = JsVarStorage.Copy
    quickcopy(newVar[].value, v)
    result.attachPtr(newVar)

# CBRuntimeBlock
when true:
  JsCBRuntimeBlock.jsClassGetSet(
    "name",
    proc(ctx: ptr JSContext; this_val: JSValue): JSValue {.cdecl.} =
      let
        blk = cast[ptr CBRuntimeBlock](this_val.getPtr(JsCBRuntimeBlock.classId))
      
      if blk != nil:
        let text = blk[].name(blk)
        return text.newStr(ctx)
      
      return Undefined
  )
  JsCBRuntimeBlock.jsClassGetSet(
    "help",
    proc(ctx: ptr JSContext; this_val: JSValue): JSValue {.cdecl.} =
      let
        blk = cast[ptr CBRuntimeBlock](this_val.getPtr(JsCBRuntimeBlock.classId))
      
      if blk != nil:
        let text = blk[].help(blk)
        return text.newStr(ctx)
      
      return Undefined
  )
  JsCBRuntimeBlock.jsClassCall(
    "cleanup",
    proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
      let
        blk = cast[ptr CBRuntimeBlock](this_val.getPtr(JsCBRuntimeBlock.classId))
      
      if blk != nil:
        blk[].cleanup(blk)
      
      return Undefined,
  )
  JsCBRuntimeBlock.jsClassCall(
    "setParam",
    proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
      let
        blk = cast[ptr CBRuntimeBlock](this_val.getPtr(JsCBRuntimeBlock.classId))
        idx = argv[0].toInt32(jsContext)
        inVar = cast[ptr CBVar](argv[1].getPtr(JsCBVar.classId))
      if inVar == nil:
        return throwTypeError(jsContext, "Invalid Var value passed")
      
      if blk != nil:
        blk[].setParam(blk, idx, inVar[])
      
      return Undefined,
    2
  )
  JsCBRuntimeBlock.jsClassCall(
    "getParam",
    proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
      let
        blk = cast[ptr CBRuntimeBlock](this_val.getPtr(JsCBRuntimeBlock.classId))
        idx = argv[0].toInt32(jsContext)
      
      if blk != nil:
        var v = blk[].getParam(blk, idx)
        return v.toJSValue(jsContext)
      
      return Undefined,
    1
  )

# CBChain
when true:
  JsCBChain.jsClassCall(
    "addBlock",
    proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
      let
        chain = cast[CBChainPtr](this_val.getPtr(JsCBChain.classId))
        blkJs = argv[0]
        blk = cast[ptr CBRuntimeBlock](argv[0].getPtr(JsCBRuntimeBlock.classId))
      if chain == nil: return throwTypeError(ctx, "addBlock invalid chain")
      if blk == nil: return throwTypeError(ctx, "addBlock invalid block")
      
      # Chain will own the block, remove the ptr from the js side, so it won't destroy it!
      blkJs.attachPtr(nil)
      
      # now add to the chain
      chain.add(blk)
      
      return Undefined,
    1
  )

  JsCBChain.jsClassGetSet(
    "looped",
    proc(ctx: ptr JSContext; this_val: JSValue): JSValue {.cdecl.} =
      let
        chain = cast[CBChainPtr](this_val.getPtr(JsCBChain.classId))
      
      if chain != nil:
        return chain[].looped.jsBool(ctx)
      
      return Undefined,
    proc(ctx: ptr JSContext; this_val: JSValue; value: JSValue): JSValue {.cdecl.} =
      let
        chain = cast[CBChainPtr](this_val.getPtr(JsCBChain.classId))
      
      if chain != nil and value.isBool:
        chain[].looped = value.toBool(jsContext)
      
      return Undefined
  )
  JsCBChain.jsClassGetSet(
    "unsafe",
    proc(ctx: ptr JSContext; this_val: JSValue): JSValue {.cdecl.} =
      let
        chain = cast[CBChainPtr](this_val.getPtr(JsCBChain.classId))
      
      if chain != nil:
        return chain[].unsafe.jsBool(ctx)
      
      return Undefined,
    proc(ctx: ptr JSContext; this_val: JSValue; value: JSValue): JSValue {.cdecl.} =
      let
        chain = cast[CBChainPtr](this_val.getPtr(JsCBChain.classId))
      
      if chain != nil and value.isBool:
        chain[].unsafe = value.toBool(jsContext)
      
      return Undefined
  )

# CBNode
when true:
  var JsCBNode = JsClass(
    finalizer: proc(self: JSValue; classId: uint32) =
      var node = self.getPtr(classId)
      if node != nil:
        destroyNode(cast[ptr CBNode](node)),
    
    constructor: proc(self: var JSValue; argc: int; argv: ptr UncheckedArray[JSValue]) =
      var node = createNode()
      self.attachPtr(node)
  )
  jsClass(jsContext, "Node", JsCBNode)

  JsCBNode.jsClassCall(
    "schedule",
    proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
      let
        node = cast[ptr CBNode](this_val.getPtr(JsCBNode.classId))
        chain = cast[CBChainPtr](argv[0].getPtr(JsCBChain.classId))
      if node == nil: return throwTypeError(ctx, "schedule invalid node")
      if chain == nil: return throwTypeError(ctx, "schedule invalid chain")
      
      try:
        node.schedule(chain)
      except StdException as e:
        return throwInternalError(jsContext, e.what())

      return Undefined,
    1
  )
  JsCBNode.jsClassCall(
    "tick",
    proc(ctx: ptr JSContext; this_val: JSValue; argc: cint; argv: ptr UncheckedArray[JSValue]): JSValue {.cdecl.} =
      let
        node = cast[ptr CBNode](this_val.getPtr(JsCBNode.classId))
      if node == nil: return throwTypeError(ctx, "schedule invalid node")
      
      node.tick()

      return Undefined,
    0
  )

when isMainModule:
  import parseopt
  
  proc writeHelp() = discard
  proc writeVersion() = discard
  
  var filename: string
  
  for kind, key, val in getopt():
    case kind
    of cmdArgument:
      filename = key
    of cmdLongOption, cmdShortOption:
      case key
      of "help", "h": writeHelp()
      of "version", "v": writeVersion()
    of cmdEnd: assert(false) # cannot happen
  if filename == "":
    # no filename has been given, so we show the help
    writeHelp()
  else:
    let fileText = readFile(filename)
    eval(jsContext, fileText)
