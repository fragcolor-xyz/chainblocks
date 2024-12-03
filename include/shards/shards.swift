/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2023 Fragcolor Pte. Ltd. */

// WIP: This is a work in progress

/*
 final class MyShard1 : IShard {
     static var name: StaticString = "MyShard1"
     static var help: StaticString = ""

     var inputTypes: [SHTypeInfo] = [
         VarType.AnyValue.asSHTypeInfo()
     ]
     var outputTypes: [SHTypeInfo] = [
         VarType.AnyValue.asSHTypeInfo()
     ]

     var parameters: [SHParameterInfo] = []
     func setParam(idx: Int, value: SHVar) -> Result<Void, ShardError> {
         .failure(ShardError(message: "Not implemented"))
     }
     func getParam(idx: Int) -> SHVar {
         SHVar()
     }

     var exposedVariables: [SHExposedTypeInfo] = []
     var requiredVariables: [SHExposedTypeInfo] = []

     func compose(data: SHInstanceData) -> Result<SHTypeInfo, ShardError> {
         .success(data.inputType)
     }

     func warmup(context: Context) -> Result<Void, ShardError> {
         .success(())
     }

     func cleanup(context: Context) -> Result<Void, ShardError> {
         .success(())
     }

     func activate(context: Context, input: SHVar) -> Result<SHVar, ShardError> {
         guard input.type == .Int else {
             return .failure(ShardError(message: "Expected Int input"))
         }
         let result = input.int * 2
         return .success(SHVar(value: result))
     }

     // -- DON'T EDIT THE FOLLOWING --
     typealias ShardType = MyShard1
     static var inputTypesCFunc: SHInputTypesProc {{ bridgeInputTypes(ShardType.self, shard: $0) }}
     static var outputTypesCFunc: SHInputTypesProc {{ bridgeOutputTypes(ShardType.self, shard: $0) }}
     static var destroyCFunc: SHDestroyProc {{ bridgeDestroy(ShardType.self, shard: $0) }}
     static var nameCFunc: SHNameProc {{ _ in bridgeName(ShardType.self) }}
     static var hashCFunc: SHHashProc {{ _ in bridgeHash(ShardType.self) }}
     static var helpCFunc: SHHelpProc {{ _ in bridgeHelp(ShardType.self) }}
     static var parametersCFunc: SHParametersProc {{ bridgeParameters(ShardType.self, shard: $0) }}
     static var setParamCFunc: SHSetParamProc {{ bridgeSetParam(ShardType.self, shard: $0, idx: $1, input: $2)}}
     static var getParamCFunc: SHGetParamProc {{ bridgeGetParam(ShardType.self, shard: $0, idx: $1)}}
     static var exposedVariablesCFunc: SHExposedVariablesProc {{ bridgeExposedVariables(ShardType.self, shard: $0) }}
     static var requiredVariablesCFunc: SHRequiredVariablesProc {{ bridgeRequiredVariables(ShardType.self, shard: $0) }}
     static var composeCFunc: SHComposeProc {{ bridgeCompose(ShardType.self, shard: $0, data: $1) }}
     static var warmupCFunc: SHWarmupProc {{ bridgeWarmup(ShardType.self, shard: $0, ctx: $1) }}
     static var cleanupCFunc: SHCleanupProc {{ bridgeCleanup(ShardType.self, shard: $0, ctx: $1) }}
     static var activateCFunc: SHActivateProc {{ bridgeActivate(ShardType.self, shard: $0, ctx: $1, input: $2) }}
     var errorCache: ContiguousArray<CChar> = []
     var output: SHVar = SHVar()
 }

 and register with:
 RegisterShard(MyShard1.name.utf8Start.withMemoryRebound(to: Int8.self, capacity: 1) { $0 }, { createSwiftShard(MyShard1.self) })
 */

import Foundation
import shards

public struct Globals {
    public var Core: UnsafeMutablePointer<SHCore>

    init() {
        // Get current directory and check if writable
        let currentPath = FileManager.default.currentDirectoryPath
        let isWritable = FileManager.default.isWritableFile(atPath: currentPath)

        // If not writable, try to set to documents directory
        if !isWritable {
            if let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.path {
                FileManager.default.changeCurrentDirectoryPath(documentsPath)
            }
        }

        // Finally init Shards
        Core = shardsInterface(UInt32(SHARDS_CURRENT_ABI))
    }
}

public var G = Globals()

public var RegisterShard: SHRegisterShard = G.Core.pointee.registerShard

public enum VarType: UInt8, CustomStringConvertible, CaseIterable {
    case NoValue
    case AnyValue
    case Enum
    case Bool
    case Int // A 64bits int
    case Int2 // A vector of 2 64bits ints
    case Int3 // A vector of 3 32bits ints
    case Int4 // A vector of 4 32bits ints
    case Int8 // A vector of 8 16bits ints
    case Int16 // A vector of 16 8bits ints
    case Float // A 64bits float
    case Float2 // A vector of 2 64bits floats
    case Float3 // A vector of 3 32bits floats
    case Float4 // A vector of 4 32bits floats
    case Color // A vector of 4 uint8
    case Shard // a shard, useful for future introspection shards!
    case EndOfBlittableTypes = 50 // anything below this is not blittable (ish)
    case Bytes // pointer + size
    case String
    case Path // An OS filesystem path
    case ContextVar // A string label to find from SHContext variables
    case Image
    case Seq
    case Table
    case Wire
    case Object
    case Array // Notice: of just blittable types!

    public var description: String {
        switch self {
        case .NoValue:
            return "None"
        case .AnyValue:
            return "Any"
        case .Enum:
            return "Enum"
        case .Bool:
            return "Bool"
        case .Int:
            return "Int"
        case .Int2:
            return "Int2"
        case .Int3:
            return "Int3"
        case .Int4:
            return "Int4"
        case .Int8:
            return "Int8"
        case .Int16:
            return "Int16"
        case .Float:
            return "Float"
        case .Float2:
            return "Float2"
        case .Float3:
            return "Float3"
        case .Float4:
            return "Float4"
        case .Color:
            return "Color"
        case .Shard:
            return "Shard"
        case .Bytes:
            return "Bytes"
        case .String:
            return "String"
        case .Path:
            return "Path"
        case .ContextVar:
            return "ContextVar"
        case .Image:
            return "Image"
        case .Seq:
            return "Seq"
        case .Table:
            return "Table"
        case .Wire:
            return "Wire"
        case .Object:
            return "Object"
        case .Array:
            return "Array"
        default:
            fatalError("Type not found!")
        }
    }

    func uxInlineable() -> Bool {
        return self == VarType.NoValue
            || self == VarType.AnyValue
            || self == VarType.Int
            || self == VarType.Int2
            || self == VarType.Int3
            || self == VarType.Int4
            || self == VarType.Int8
            || self == VarType.Int16
            || self == VarType.Float
            || self == VarType.Float2
            || self == VarType.Float3
            || self == VarType.Float4
    }

    func asSHType() -> SHType {
        SHType(rawValue: rawValue)
    }

    func asSHTypeInfo() -> SHTypeInfo {
        var info = SHTypeInfo()
        info.basicType = asSHType()
        return info
    }
}

extension SHVar: CustomStringConvertible {
    public var description: String {
        switch type {
        case .NoValue:
            return "nil"
        case .AnyValue:
            return "Any"
        case .Enum:
            return "(Enum \(payload.enumVendorId) \(payload.enumTypeId) \(payload.enumValue))"
        case .Bool:
            return "\(payload.boolValue)"
        case .Int:
            return "\(payload.intValue)"
        case .Int2:
            return "(Int2 \(payload.int2Value.x) \(payload.int2Value.y))"
        case .Int3:
            return "(Int3 \(payload.int3Value.x) \(payload.int3Value.y) \(payload.int3Value.z))"
        case .Int4:
            return "\(payload.int4Value)"
        case .Int8:
            return "\(payload.int8Value)"
        case .Int16:
            return "\(payload.int16Value)"
        case .Float:
            return "\(payload.floatValue)"
        case .Float2:
            return "\(payload.float2Value)"
        case .Float3:
            return "\(payload.float3Value)"
        case .Float4:
            return "\(payload.float4Value)"
        case .Color:
            return "Color"
        case .Shard:
            return "Shard"
        case .Bytes:
            return "Bytes"
        case .String:
            return .init(cString: payload.stringValue)
        case .Path:
            return "Path"
        case .ContextVar:
            return "ContextVar"
        case .Image:
            return "Image"
        case .Seq:
            return "Seq"
        case .Table:
            return "Table"
        case .Wire:
            return "Wire"
        case .Object:
            return "Object"
        case .Array:
            return "Array"
        default:
            fatalError("Type not found!")
        }
    }

    public var typename: String {
        type.description
    }

    public var type: VarType {
        VarType(rawValue: valueType.rawValue)!
    }

    public mutating func Clone(dst: inout SHVar) {
        G.Core.pointee.cloneVar(&dst, &self)
    }

    public mutating func Clone() -> SHVar {
        var v = SHVar()
        G.Core.pointee.cloneVar(&v, &self)
        return v
    }

    public mutating func Destroy() {
        G.Core.pointee.destroyVar(&self)
    }

    init(value: Bool) {
        var v = SHVar()
        v.valueType = Bool
        v.payload.boolValue = SHBool(value)
        self = v
    }

    public var bool: Bool {
        get {
            assert(type == .Bool, "Bool variable expected!")
            return Bool(payload.boolValue)
        }
        set {
            self = .init(value: newValue)
        }
    }

    public init(value: Int) {
        var v = SHVar()
        v.valueType = Int
        v.payload.intValue = SHInt(value)
        self = v
    }

    public var int: Int {
        get {
            assert(type == .Int, "Int variable expected!")
            return Int(payload.intValue)
        } set {
            self = .init(value: newValue)
        }
    }

    init(x: Int64, y: Int64) {
        var v = SHVar()
        v.valueType = Int2
        v.payload.int2Value = SHInt2(x: x, y: y)
        self = v
    }

    init(value: SIMD2<Int64>) {
        var v = SHVar()
        v.valueType = Int2
        v.payload.int2Value = value
        self = v
    }

    public init(value: Float) {
        var v = SHVar()
        v.valueType = Float
        v.payload.floatValue = SHFloat(value)
        self = v
    }

    public var float: Float {
        get {
            assert(type == .Float, "Float variable expected!")
            return Float(payload.floatValue)
        } set {
            self = .init(value: newValue)
        }
    }

    init(value: inout ContiguousArray<CChar>) {
        var v = SHVar()
        v.valueType = String
        value.withUnsafeBufferPointer {
            v.payload.stringValue = $0.baseAddress
            v.payload.stringLen = UInt32(value.count - 1) // assumes \0 terminator
            v.payload.stringCapacity = UInt32(value.capacity)
        }
        self = v
    }
    
    init(string: StaticString) {
        var v = SHVar()
        v.valueType = String
        v.payload.stringValue = string.withUTF8Buffer { buffer in
            unsafeBitCast(buffer.baseAddress, to: UnsafePointer<CChar>.self)
        }
        v.payload.stringLen = UInt32(string.utf8CodeUnitCount)
        v.payload.stringCapacity = 0
        self = v
    }

    public var string: String {
        assert(type == .String, "String variable expected!")
        guard let stringPtr = payload.stringValue else {
            return ""
        }
        let length = Int(payload.stringLen)
        guard length > 0 else {
            return ""
        }
        // Cast `CChar` (Int8) to `UInt8` for decoding
        let buffer = UnsafeBufferPointer(start: stringPtr, count: length).map { UInt8(bitPattern: $0) }
        return String(decoding: buffer, as: UTF8.self)
    }

    init(value: ShardPtr) {
        var v = SHVar()
        v.valueType = SHType(rawValue: VarType.Shard.rawValue)
        v.payload.shardValue = value
        self = v
    }

    public var shard: ShardPtr {
        get {
            assert(type == .Shard, "Shard variable expected!")
            return payload.shardValue
        }
        set {
            self = .init(value: newValue)
        }
    }

    init(value: UnsafeMutableBufferPointer<SHVar>) {
        var v = SHVar()
        v.valueType = Seq
        v.payload.seqValue.elements = value.baseAddress
        v.payload.seqValue.len = UInt32(value.count)
        v.payload.seqValue.cap = 0
        self = v
    }

    public var seq: UnsafeMutableBufferPointer<SHVar> {
        get {
            assert(type == .Seq, "Seq variable expected!")
            return .init(start: payload.seqValue.elements, count: Int(payload.seqValue.len))
        } set {
            self = .init(value: newValue)
        }
    }

    init(pointer: UnsafeMutableRawPointer, vendorId: Int32, typeId: Int32) {
        var v = SHVar()
        v.valueType = Object
        v.payload.objectVendorId = vendorId
        v.payload.objectTypeId = typeId
        v.payload.objectValue = pointer
        self = v
    }
    
    func isNone() -> Bool {
        return type == .NoValue
    }
}

class OwnedVar {
    var v: SHVar

    init() {
        v = SHVar()
    }

    init(cloning: SHVar) {
        v = SHVar()

        withUnsafePointer(to: cloning) { ptr in
            G.Core.pointee.cloneVar(&v, UnsafeMutablePointer(mutating: ptr))
        }
    }

    init(owning: SHVar) {
        v = owning
    }

    init(string: String) {
        v = SHVar()
        set(string: string)
    }

    deinit {
        withUnsafeMutablePointer(to: &v) { ptr in
            G.Core.pointee.destroyVar(ptr)
        }
    }

    func ptr() -> UnsafeMutablePointer<SHVar> {
        return withUnsafeMutablePointer(to: &v) { ptr in
            UnsafeMutablePointer(mutating: ptr)
        }
    }

    func set(string: String) {
        string.withCString { cString in
            var tmp = SHVar()
            tmp.valueType = VarType.String.asSHType()
            tmp.payload.stringValue = cString
            let length = string.lengthOfBytes(using: .utf8)
            tmp.payload.stringLen = UInt32(length)
            G.Core.pointee.cloneVar(&v, &tmp)
        }
    }
}

class TableVar {
    var containerVar: SHVar

    init() {
        containerVar = SHVar()
        containerVar.valueType = VarType.Table.asSHType()
        containerVar.payload.tableValue = G.Core.pointee.tableNew()
    }

    init(cloning: SHVar) {
        assert(cloning.valueType == VarType.Table.asSHType())

        containerVar = SHVar()
        withUnsafePointer(to: cloning) { ptr in
            G.Core.pointee.cloneVar(&containerVar, UnsafeMutablePointer(mutating: ptr))
        }
    }

    deinit {
        withUnsafeMutablePointer(to: &containerVar) { ptr in
            G.Core.pointee.destroyVar(ptr)
        }
    }

    func insertOrUpdate(key: SHVar, cloning: SHVar) {
        let vPtr = containerVar.payload.tableValue.api.pointee.tableAt(containerVar.payload.tableValue, key)
        withUnsafePointer(to: cloning) { ptr in
            G.Core.pointee.cloneVar(vPtr, UnsafeMutablePointer(mutating: ptr))
        }
    }

    func get(key: SHVar) -> SHVar {
        let vPtr = containerVar.payload.tableValue.api.pointee.tableAt(containerVar.payload.tableValue, key)
        return vPtr!.pointee
    }
    
    func get(key: StaticString) -> SHVar {
        return get(key: SHVar(string: key))
    }

    func clear() {
        containerVar.payload.tableValue.api.pointee.tableClear(containerVar.payload.tableValue)
    }
}

class SeqVar {
    var containerVar: SHVar

    init() {
        containerVar = SHVar()
        containerVar.valueType = VarType.Seq.asSHType()
    }

    init(cloning: SHVar) {
        assert(cloning.valueType == VarType.Seq.asSHType())

        containerVar = SHVar()
        withUnsafePointer(to: cloning) { ptr in
            G.Core.pointee.cloneVar(&containerVar, UnsafeMutablePointer(mutating: ptr))
        }
    }

    deinit {
        withUnsafeMutablePointer(to: &containerVar) { ptr in
            G.Core.pointee.destroyVar(ptr)
        }
    }

    func get() -> SHVar {
        return containerVar
    }

    func ptr() -> UnsafeMutablePointer<SHVar> {
        return withUnsafeMutablePointer(to: &containerVar) { ptr in
            ptr
        }
    }

    func resize(size: Int) {
        withUnsafeMutablePointer(to: &containerVar.payload.seqValue) { ptr in
            G.Core.pointee.seqResize(ptr, UInt32(size))
        }
    }

    func clear() {
        resize(size: 0)
    }

    func size() -> Int {
        return Int(containerVar.payload.seqValue.len)
    }

    func pushRaw(value: SHVar) {
        let index = size()
        resize(size: index + 1)
        containerVar.payload.seqValue.elements[index] = value
    }

    func pushCloning(value: SHVar) {
        let index = size()
        resize(size: index + 1)
        withUnsafePointer(to: value) { ptr in
            G.Core.pointee.cloneVar(&containerVar.payload.seqValue.elements[index], UnsafeMutablePointer(mutating: ptr))
        }
    }

    func push(string: String) {
        string.utf8CString.withUnsafeBufferPointer { buffer in
            var tmp = SHVar()
            tmp.valueType = VarType.String.asSHType()
            tmp.payload.stringValue = buffer.baseAddress
            tmp.payload.stringLen = UInt32(buffer.count - 1) // Subtract 1 to exclude null terminator
            pushCloning(value: tmp)
        }
    }

    // Notice that the memory of the result is still owned by SeqVar as when we destroy we destroy capacity!
    // So a further push will reuse same memory!
    @discardableResult func popRaw() -> SHVar {
        assert(size() > 0)
        let index = size() - 1
        resize(size: index)
        return containerVar.payload.seqValue.elements[index]
    }

    func at(index: Int) -> SHVar {
        assert(index >= 0 && index < size())
        return containerVar.payload.seqValue.elements[index]
    }
}

public struct Context {
    public var context: OpaquePointer?

    public init(context: OpaquePointer?) {
        self.context = context
    }
}

public typealias ShardPtr = UnsafeMutablePointer<Shard>?

public final class ShardError: Error {
    public var message: String

    init(message: String) {
        self.message = message
    }
}

public protocol IShard: AnyObject {
    static var name: StaticString { get }
    static var help: StaticString { get }

    init()

    var inputTypes: [SHTypeInfo] { get }
    var outputTypes: [SHTypeInfo] { get }

    var parameters: [SHParameterInfo] { get }
    func setParam(idx: Int, value: SHVar) -> Result<Void, ShardError>
    func getParam(idx: Int) -> SHVar

    var exposedVariables: [SHExposedTypeInfo] { get }
    var requiredVariables: [SHExposedTypeInfo] { get }

    func compose(data: SHInstanceData) -> Result<SHTypeInfo, ShardError>

    func warmup(context: Context) -> Result<Void, ShardError>
    func cleanup(context: Context) -> Result<Void, ShardError>

    func activate(context: Context, input: SHVar) -> Result<SHVar, ShardError>

    // Need those... cos Swift generics are meh
    // I wasted a lot of time to find the optimal solution, don't think about wasting more
    // Could have used purely inherited classes but looked meh, could have done this that..
    // more here: https://chat.openai.com/share/023818da-89da-4f18-a79e-f46774e7fc8d (scoll down)
    static var inputTypesCFunc: SHInputTypesProc { get }
    static var outputTypesCFunc: SHInputTypesProc { get }
    static var destroyCFunc: SHDestroyProc { get }
    static var nameCFunc: SHNameProc { get }
    static var hashCFunc: SHHashProc { get }
    static var helpCFunc: SHHelpProc { get }
    static var parametersCFunc: SHParametersProc { get }
    static var setParamCFunc: SHSetParamProc { get }
    static var getParamCFunc: SHGetParamProc { get }
    static var exposedVariablesCFunc: SHExposedVariablesProc { get }
    static var requiredVariablesCFunc: SHRequiredVariablesProc { get }
    static var composeCFunc: SHComposeProc { get }
    static var warmupCFunc: SHWarmupProc { get }
    static var cleanupCFunc: SHCleanupProc { get }
    static var activateCFunc: SHActivateProc { get }
    var errorCache: ContiguousArray<CChar> { get set }
    var output: SHVar { get set }
}

public extension IShard {}

@inlinable public func bridgeParameters<T: IShard>(_: T.Type, shard: ShardPtr) -> SHParametersInfo {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var result = SHParametersInfo()
    let paramsPtr = b.parameters.withUnsafeBufferPointer {
        $0.baseAddress
    }
    result.elements = UnsafeMutablePointer<SHParameterInfo>(mutating: paramsPtr)
    result.len = UInt32(b.parameters.count)
    return result
}

@inlinable public func bridgeName<T: IShard>(_: T.Type) -> UnsafePointer<Int8>? {
    return T.name.utf8Start.withMemoryRebound(to: Int8.self, capacity: 1) { $0 }
}

@inlinable public func bridgeHash<T: IShard>(_: T.Type) -> UInt32 {
    return hashShard(T.self)
}

@inlinable public func bridgeSetParam<T: IShard>(_: T.Type, shard: ShardPtr, idx: Int32, input: UnsafePointer<SHVar>?) -> SHError {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var error = SHError()
    let result = b.setParam(idx: Int(idx), value: input!.pointee)
    switch result {
    case .success():
        return error
    case let .failure(err):
        error.code = 1
        b.errorCache = err.message.utf8CString
        error.message.string = b.errorCache.withUnsafeBufferPointer {
            $0.baseAddress
        }
        error.message.len = UInt64(b.errorCache.count - 1)
        return error
    }
}

@inlinable public func bridgeGetParam<T: IShard>(_: T.Type, shard: ShardPtr, idx: Int32) -> SHVar {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    return b.getParam(idx: Int(idx))
}

@inlinable public func bridgeHelp<T: IShard>(_: T.Type) -> SHOptionalString {
    var result = SHOptionalString()
    result.string = T.help.utf8Start.withMemoryRebound(to: Int8.self, capacity: 1) { $0 }
    return result
}

@inlinable public func bridgeDestroy<T: IShard>(_: T.Type, shard: ShardPtr) {
    let reboundShard = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self)
    _ = Unmanaged<T>.fromOpaque(reboundShard.pointee.swiftClass).takeRetainedValue()
    shard!.deallocate()
}

@inlinable public func bridgeInputTypes<T: IShard>(_: T.Type, shard: ShardPtr) -> SHTypesInfo {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var result = SHTypesInfo()
    let ptr = b.inputTypes.withUnsafeBufferPointer {
        $0.baseAddress
    }
    result.elements = UnsafeMutablePointer<SHTypeInfo>(mutating: ptr)
    result.len = UInt32(b.inputTypes.count)
    return result
}

@inlinable public func bridgeOutputTypes<T: IShard>(_: T.Type, shard: ShardPtr) -> SHTypesInfo {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var result = SHTypesInfo()
    let ptr = b.outputTypes.withUnsafeBufferPointer {
        $0.baseAddress
    }
    result.elements = UnsafeMutablePointer<SHTypeInfo>(mutating: ptr)
    result.len = UInt32(b.outputTypes.count)
    return result
}

@inlinable public func bridgeCompose<T: IShard>(_: T.Type, shard: ShardPtr, data: UnsafeMutablePointer<SHInstanceData>?) -> SHShardComposeResult {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var value = SHShardComposeResult()
    let result = b.compose(data: data!.pointee)
    switch result {
    case let .success(typ):
        value.result = typ
        return value
    case let .failure(err):
        var error = SHError()
        error.code = 1
        b.errorCache = err.message.utf8CString
        error.message.string = b.errorCache.withUnsafeBufferPointer {
            $0.baseAddress
        }
        error.message.len = UInt64(b.errorCache.count - 1)
        value.error = error
        return value
    }
}

@inlinable public func bridgeWarmup<T: IShard>(_: T.Type, shard: ShardPtr, ctx: OpaquePointer?) -> SHError {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var error = SHError()
    let result = b.warmup(context: Context(context: ctx))
    switch result {
    case .success():
        return error
    case let .failure(err):
        error.code = 1
        b.errorCache = err.message.utf8CString
        error.message.string = b.errorCache.withUnsafeBufferPointer {
            $0.baseAddress
        }
        error.message.len = UInt64(b.errorCache.count - 1)
        return error
    }
}

@inlinable public func bridgeCleanup<T: IShard>(_: T.Type, shard: ShardPtr, ctx: OpaquePointer?) -> SHError {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var error = SHError()
    let result = b.cleanup(context: Context(context: ctx))
    switch result {
    case .success():
        return error
    case let .failure(err):
        error.code = 1
        b.errorCache = err.message.utf8CString
        error.message.string = b.errorCache.withUnsafeBufferPointer {
            $0.baseAddress
        }
        error.message.len = UInt64(b.errorCache.count - 1)
        return error
    }
}

@inlinable public func bridgeActivate<T: IShard>(_: T.Type, shard: ShardPtr, ctx: OpaquePointer?, input: UnsafePointer<SHVar>?) -> UnsafePointer<SHVar>? {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    let result = b.activate(context: Context(context: ctx), input: input!.pointee)
    switch result {
    case let .success(res):
        b.output = res
        return withUnsafePointer(to: b.output) { $0 }
    case let .failure(error):
        var errorMsg = SHStringWithLen()
        let error = error.message.utf8CString
        errorMsg.string = error.withUnsafeBufferPointer {
            $0.baseAddress
        }
        errorMsg.len = UInt64(error.count - 1)
        G.Core.pointee.abortWire(ctx, errorMsg)
        return withUnsafePointer(to: b.output) { $0 }
    }
}

@inlinable public func bridgeExposedVariables<T: IShard>(_: T.Type, shard: ShardPtr) -> SHExposedTypesInfo {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var result = SHExposedTypesInfo()
    let ptr = b.exposedVariables.withUnsafeBufferPointer {
        $0.baseAddress
    }
    result.elements = UnsafeMutablePointer<SHExposedTypeInfo>(mutating: ptr)
    result.len = UInt32(b.exposedVariables.count)
    return result
}

@inlinable public func bridgeRequiredVariables<T: IShard>(_: T.Type, shard: ShardPtr) -> SHExposedTypesInfo {
    let a = UnsafeRawPointer(shard!).assumingMemoryBound(to: SwiftShard.self).pointee
    let b = Unmanaged<T>.fromOpaque(a.swiftClass).takeUnretainedValue()
    var result = SHExposedTypesInfo()
    let ptr = b.requiredVariables.withUnsafeBufferPointer {
        $0.baseAddress
    }
    result.elements = UnsafeMutablePointer<SHExposedTypeInfo>(mutating: ptr)
    result.len = UInt32(b.requiredVariables.count)
    return result
}

@inlinable public func hashShard<T: IShard>(_: T.Type) -> UInt32 {
    let name = T.name
    let namePtr = name.utf8Start.withMemoryRebound(to: UInt8.self, capacity: name.utf8CodeUnitCount) { $0 }
    let nameData = Data(bytes: namePtr, count: name.utf8CodeUnitCount)

    // Create a buffer with the shard name and SHARDS_CURRENT_ABI
    var buffer = [UInt8]()
    buffer.append(contentsOf: nameData)

    var abi = UInt32(SHARDS_CURRENT_ABI)
    withUnsafeBytes(of: &abi) { buffer.append(contentsOf: $0) }

    // Compute the hash using a simple algorithm (FNV-1a in this case)
    var hash: UInt32 = 2_166_136_261
    for byte in buffer {
        hash ^= UInt32(byte)
        hash &*= 16_777_619
    }

    return hash
}

func createSwiftShard<T: IShard>(_: T.Type) -> UnsafeMutablePointer<Shard>? {
    #if DEBUG
        print("Creating swift shard: \(T.name)")
    #endif
    let shard = T()
    let cwrapper = UnsafeMutablePointer<SwiftShard>.allocate(capacity: 1)
    cwrapper.initialize(to: SwiftShard())

    cwrapper.pointee.header.name = T.nameCFunc
    cwrapper.pointee.header.hash = T.hashCFunc
    cwrapper.pointee.header.help = T.helpCFunc
    cwrapper.pointee.header.inputHelp = { _ in SHOptionalString() }
    cwrapper.pointee.header.outputHelp = { _ in SHOptionalString() }
    cwrapper.pointee.header.properties = { _ in nil }
    cwrapper.pointee.header.setup = { _ in }
    cwrapper.pointee.header.destroy = T.destroyCFunc
    cwrapper.pointee.header.activate = T.activateCFunc
    cwrapper.pointee.header.parameters = T.parametersCFunc
    cwrapper.pointee.header.setParam = T.setParamCFunc
    cwrapper.pointee.header.getParam = T.getParamCFunc
    cwrapper.pointee.header.inputTypes = T.inputTypesCFunc
    cwrapper.pointee.header.outputTypes = T.outputTypesCFunc
    cwrapper.pointee.header.warmup = T.warmupCFunc
    cwrapper.pointee.header.cleanup = T.cleanupCFunc
    cwrapper.pointee.header.compose = T.composeCFunc
    cwrapper.pointee.header.exposedVariables = T.exposedVariablesCFunc
    cwrapper.pointee.header.requiredVariables = T.requiredVariablesCFunc

    cwrapper.pointee.swiftClass = Unmanaged<T>.passRetained(shard).toOpaque()

    // Cast to Shard pointer without rebinding
    return UnsafeMutableRawPointer(cwrapper).assumingMemoryBound(to: Shard.self)
}

struct ShardParameter {
    private weak var owner: ShardController?
    var info: ParameterInfo

    public var value: SHVar {
        get {
            owner!.getParam(index: info.index)
        }
        set {
            _ = owner!.setParam(index: info.index, value: newValue)
        }
    }

    public init(shard: ShardController, info: ParameterInfo) {
        owner = shard
        self.info = info
    }
}

class ParameterInfo {
    init(name: String, types: SHTypesInfo, index: Int) {
        self.name = name
        help = ""
        self.types = types
        self.index = index
    }

    init(name: String, help: String, types: SHTypesInfo, index: Int) {
        self.name = name
        self.help = help
        self.types = types
        self.index = index
    }

    var name: String
    var help: String
    var types: SHTypesInfo
    var index: Int
}

class ShardController: Equatable, Identifiable {
    var id: Int {
        nativeShard.hashValue
    }

    convenience init(name: String) {
        let n = name.utf8CString
        var cname = SHStringWithLen()
        cname.string = n.withUnsafeBufferPointer {
            $0.baseAddress
        }
        cname.len = UInt64(n.count - 1)
        self.init(native: G.Core.pointee.createShard(cname)!)
    }

    init(native: ShardPtr) {
        nativeShard = native

        let blkname = nativeShard?.pointee.name(nativeShard!)
        let nparams = nativeShard?.pointee.parameters(nativeShard)
        if (nparams?.len ?? 0) > 0 {
            for i in 0 ..< nparams!.len {
                let nparam = nparams!.elements[Int(i)]
                let nparamId = "\(blkname!)-\(nparam.name!)"
                if let info = ShardController.infos[nparamId] {
                    params.append(ShardParameter(shard: self, info: info))
                } else {
                    let info = ParameterInfo(
                        name: .init(cString: nparam.name!),
                        // help: .init(cString: nparam.help!),
                        types: nparam.valueTypes,
                        index: Int(i)
                    )
                    ShardController.infos[nparamId] = info
                    params.append(ShardParameter(shard: self, info: info))
                }
            }
        }
    }

    deinit {
        if nativeShard != nil {
            if !nativeShard!.pointee.owned {
                nativeShard!.pointee.destroy(nativeShard)
            }
        }
    }

    static func == (lhs: ShardController, rhs: ShardController) -> Bool {
        return lhs.nativeShard == rhs.nativeShard
    }

    var inputTypes: UnsafeMutableBufferPointer<SHTypeInfo> {
        let infos = nativeShard!.pointee.inputTypes(nativeShard!)
        return .init(start: infos.elements, count: Int(infos.len))
    }

    var noInput: Bool {
        inputTypes.allSatisfy { info in
            info.basicType == None
        }
    }

    var outputTypes: UnsafeMutableBufferPointer<SHTypeInfo> {
        let infos = nativeShard!.pointee.outputTypes(nativeShard!)
        return .init(start: infos.elements, count: Int(infos.len))
    }

    var name: String {
        .init(cString: nativeShard!.pointee.name(nativeShard)!)
    }

    var parameters: [ShardParameter] {
        params
    }

    func getParam(index: Int) -> SHVar {
        nativeShard!.pointee.getParam(nativeShard, Int32(index))
    }

    func setParam(index: Int, value: SHVar) -> ShardController {
        #if DEBUG
            print("setParam(\(index), \(value))")
        #endif
        var mutableValue = value
        let error = nativeShard!.pointee.setParam(nativeShard, Int32(index), &mutableValue)
        if error.code != SH_ERROR_NONE {
            print("Error setting parameter: \(error)")
        }
        return self
    }

    func setParam(name: String, value: SHVar) -> ShardController {
        for idx in params.indices {
            if params[idx].info.name == name {
                params[idx].value = value
                return self
            }
        }
        fatalError("Parameter not found! \(name)")
    }

    func setParam(name: String, string: String) -> ShardController {
        for idx in params.indices {
            if params[idx].info.name == name {
                var ustr = string.utf8CString
                params[idx].value = SHVar(value: &ustr)
                return self
            }
        }
        fatalError("Parameter not found! \(name)")
    }

    func setParam(index: Int, string: String) -> ShardController {
        var ustr = string.utf8CString
        params[index].value = SHVar(value: &ustr)
        return self
    }

    func setParam(name: String, @ShardsBuilder _ contents: () -> [ShardController]) -> ShardController {
        let shards = contents()
        for idx in params.indices {
            if params[idx].info.name == name {
                var vshards = shards.map {
                    SHVar(value: $0.nativeShard!)
                }
                vshards.withUnsafeMutableBufferPointer {
                    params[idx].value.seq = $0
                }
                return self
            }
        }
        fatalError("Parameter not found! \(name)")
    }

    var nativeShard: ShardPtr
    var params = [ShardParameter]()
    static var infos: [String: ParameterInfo] = [:]
}

@resultBuilder struct ShardsBuilder {
    static func buildBlock(_ components: ShardController...) -> [ShardController] {
        components
    }

    static func buildBlock(_ component: ShardController) -> [ShardController] {
        [component]
    }

    static func buildOptional(_ component: [ShardController]?) -> [ShardController] {
        component ?? []
    }

    static func buildEither(first: [ShardController]) -> [ShardController] {
        first
    }

    static func buildEither(second: [ShardController]) -> [ShardController] {
        second
    }

    static func buildArray(_ components: [[ShardController]]) -> [ShardController] {
        components.flatMap { $0 }
    }

    static func buildExpression(_ expression: ShardController) -> [ShardController] {
        [expression]
    }

    static func buildExpression(_ expression: [ShardController]) -> [ShardController] {
        expression
    }

    static func buildLimitedAvailability(_ component: [ShardController]) -> [ShardController] {
        component
    }

    static func buildFinalResult(_ component: [ShardController]) -> [ShardController] {
        component
    }
}

class WireController {
    init() {
        let cname = SHStringWithLen()
        nativeRef = G.Core.pointee.createWire(cname)
    }

    init(native: SHWireRef) {
        nativeRef = native
    }

    convenience init(shards: [ShardController]) {
        self.init()
        for item in shards {
            add(shard: item)
        }
    }

    convenience init(@ShardsBuilder _ contents: () -> [ShardController]) {
        self.init(shards: contents())
    }

    deinit {
        for variable in variables {
            G.Core.pointee.releaseVariable(variable.value)
        }
        
        if nativeRef != nil {
            G.Core.pointee.destroyWire(nativeRef)
        }
    }

    func add(shard: ShardController) {
        G.Core.pointee.addShard(nativeRef, shard.nativeShard)
    }

    var looped: Bool = false {
        didSet {
            G.Core.pointee.setWireLooped(nativeRef, looped)
        }
    }

    var unsafe: Bool = false {
        didSet {
            G.Core.pointee.setWireUnsafe(nativeRef, unsafe)
        }
    }

    private func addExternalVar(name: String, varPtr: UnsafeMutablePointer<SHVar>) {
        varPtr.pointee.flags |= UInt16(SHVAR_FLAGS_EXTERNAL)
        var ev = SHExternalVariable()
        ev.var = varPtr

        name.withCString { cString in
            var cname = SHStringWithLen()
            cname.string = cString
            let length = name.lengthOfBytes(using: .utf8)
            cname.len = UInt64(length)
            G.Core.pointee.setExternalVariable(nativeRef, cname, &ev)
        }
    }

    func addExternal(name: String, owned: inout OwnedVar) {
        addExternalVar(name: name, varPtr: owned.ptr())
    }

    func addExternal(name: String, sequence: inout SeqVar) {
        addExternalVar(name: name, varPtr: sequence.ptr())
    }

    func addExternal(name: String, raw: inout SHVar) {
        addExternalVar(name: name, varPtr: &raw)
    }
    
    func referenceVariable(name: String) -> UnsafeMutablePointer<SHVar> {
        if variables[name] != nil {
            return variables[name]!
        } else {
            return name.withCString { cString in
                var cname = SHStringWithLen()
                cname.string = cString
                let length = name.lengthOfBytes(using: .utf8)
                cname.len = UInt64(length)
                let ptr = G.Core.pointee.referenceWireVariable(nativeRef, cname)!
                variables[name] = ptr
                return ptr
            }
        }
    }
    
    func setVariable(name: String, value: SHVar, tracked: Bool = false) {
        let dst = referenceVariable(name: name)
        if tracked {
            dst.pointee.flags |= UInt16(SHVAR_FLAGS_TRACKED) // make sure this is there!
        }
        withUnsafePointer(to: value) { src in
            G.Core.pointee.cloneVar(dst, src)
        }
    }
    
    func setVariable(name: String, value: String, tracked: Bool = false) {
        value.withCString { cString in
            var tmp = SHVar()
            tmp.valueType = VarType.String.asSHType()
            tmp.payload.stringValue = cString
            let length = value.lengthOfBytes(using: .utf8)
            tmp.payload.stringLen = UInt32(length)
            setVariable(name: name, value: tmp, tracked: tracked)
        }
    }

    func isRunning() -> Bool {
        G.Core.pointee.isWireRunning(nativeRef)
    }
    
    func setPriority(_ priority: Int) {
        G.Core.pointee.setWirePriority(nativeRef, Int32(priority))
    }

    var nativeRef = SHWireRef(bitPattern: 0)
    var variables: [String: UnsafeMutablePointer<SHVar>] = [:]
}

class MeshController {
    init() {
        nativeRef = G.Core.pointee.createMesh()
    }

    deinit {
        G.Core.pointee.destroyMesh(nativeRef)
    }

    func schedule(wire: WireController) {
        G.Core.pointee.schedule(nativeRef, wire.nativeRef, true)
    }

    func unschedule(wire: WireController) {
        G.Core.pointee.unschedule(nativeRef, wire.nativeRef)
    }

    func tick() -> Bool {
        G.Core.pointee.tick(nativeRef)
    }

    func isEmpty() -> Bool {
        G.Core.pointee.isEmpty(nativeRef)
    }

    var nativeRef = SHMeshRef(bitPattern: 0)
}

extension SHStringWithLen {
    // Create SHStringWithLen from array of CChar
    static func from(_ chars: ContiguousArray<CChar>) -> SHStringWithLen {
        var result = SHStringWithLen()
        result.string = chars.withUnsafeBufferPointer { $0.baseAddress }
        result.len = UInt64(chars.count - 1) // Subtract 1 to exclude null terminator
        return result
    }

    // Create SHStringWithLen from a static compile-time string
    static func fromStatic(_ staticString: StaticString) -> SHStringWithLen {
        var result = SHStringWithLen()
        result.string = staticString.withUTF8Buffer { buffer in
            unsafeBitCast(buffer.baseAddress, to: UnsafePointer<CChar>.self)
        }
        result.len = UInt64(staticString.utf8CodeUnitCount)
        return result
    }

    // Convert SHStringWithLen to Swift String
    func toString() -> String? {
        guard let cString = string else { return nil }
        return String(cString: cString)
    }

    // Create an empty SHStringWithLen
    static var empty: SHStringWithLen {
        var result = SHStringWithLen()
        result.string = nil
        result.len = 0
        return result
    }

    // Check if SHStringWithLen is empty
    var isEmpty: Bool {
        return len == 0 || string == nil
    }
}

class SwiftSWL {
    var chars: ContiguousArray<CChar> // store the CChar array directly

    init(_ string: String) {
        chars = string.utf8CString
    }

    func asSHStringWithLen() -> SHStringWithLen {
        SHStringWithLen.from(chars)
    }
}

class Shards {
    static func log(_ message: String) {
        message.withCString { cString in
            var shString = SHStringWithLen()
            shString.string = cString
            let length = message.lengthOfBytes(using: .utf8)
            shString.len = UInt64(length)
            G.Core.pointee.log(shString)
        }
    }

    static func evalWire(_ name: String, _ code: String, _ basePath: String) -> WireController? {
        // Create SHStringWithLen instances
        let nameStr = SwiftSWL(name)
        let codeStr = SwiftSWL(code)
        let basePathStr = SwiftSWL(basePath)

        // Read the AST
        var ast = G.Core.pointee.read(nameStr.asSHStringWithLen(), codeStr.asSHStringWithLen(), basePathStr.asSHStringWithLen(), nil, 0)
        guard ast.error == nil else {
            return nil
        }

        // Create evaluation environment
        let emptyStr = SHStringWithLen.fromStatic("")
        let env = G.Core.pointee.createEvalEnv(emptyStr)

        // Evaluate the AST
        let error = G.Core.pointee.eval(env, &ast.ast) // consumes ast
        guard error == nil else {
            G.Core.pointee.freeEvalEnv(env)
            return nil
        }

        // Transform environment into a wire
        var wire = G.Core.pointee.transformEnv(env, nameStr.asSHStringWithLen()) // consumes env
        guard wire.error == nil else {
            G.Core.pointee.freeWire(&wire)
            return nil
        }

        // Create WireController from the resulting wire
        let wireController = WireController(native: wire.wire.pointee!)
        return wireController
    }

    static func evalWire(_ name: String, _ ast: [UInt8]) -> WireController? {
        // Create SHStringWithLen instances
        let nameStr = SwiftSWL(name)

        // Read the AST
        var ast = ast.withUnsafeBufferPointer { buffer in
            G.Core.pointee.loadAst(buffer.baseAddress!, UInt32(buffer.count))
        }
        guard ast.error == nil else {
            return nil
        }

        // Create evaluation environment
        let emptyStr = SHStringWithLen.fromStatic("")
        let env = G.Core.pointee.createEvalEnv(emptyStr)

        // Evaluate the AST
        let error = G.Core.pointee.eval(env, &ast.ast) // consumes ast
        guard error == nil else {
            G.Core.pointee.freeEvalEnv(env)
            return nil
        }

        // Transform environment into a wire
        var wire = G.Core.pointee.transformEnv(env, nameStr.asSHStringWithLen()) // consumes env
        guard wire.error == nil else {
            G.Core.pointee.freeWire(&wire)
            return nil
        }

        // Create WireController from the resulting wire
        let wireController = WireController(native: wire.wire.pointee!)
        return wireController
    }
}

#if canImport(UIKit)
    import UIKit

    extension UIView {
        var safeArea: UIEdgeInsets {
            if #available(iOS 11, *) {
                if let window = (UIApplication.shared.connectedScenes.first as? UIWindowScene)?.windows.first {
                    return window.safeAreaInsets
                }
            }
            return UIEdgeInsets(top: 0.0, left: 0.0, bottom: 0.0, right: 0.0)
        }
    }

    @_cdecl("shards_get_uiview_safe_area")
    public func getViewSafeArea(uiEdgeInsets: UnsafeMutablePointer<UIEdgeInsets>, viewPtr: UnsafeMutableRawPointer?) {
        let view = Unmanaged<UIView>.fromOpaque(viewPtr!).takeUnretainedValue()
        uiEdgeInsets.pointee = view.safeArea
    }

    extension WireController {
        public func wait() async {
            // Check copier status every 100ms
            while isRunning() {
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }
    }
#endif
