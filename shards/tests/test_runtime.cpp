/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

#include <random>

#include <shards/langffi/bindings.h>
#include <shards/ops.hpp>
#include <shards/utility.hpp>
#include <shards/common_types.hpp>
#include <shards/core/async.hpp>
#include <shards/core/runtime.hpp>
#include <shards/core/serialization.hpp>
#include <shards/linalg_shim.hpp>
#include <shards/wire_dsl.hpp>
#include "shards/core/wire_doppelganger_pool.hpp"

#undef CHECK

#define CATCH_CONFIG_RUNNER

#include <catch2/catch_all.hpp>

// Defined in the gfx rust crate
//   used to initialize tracy on the rust side, since it required special intialization (C++ doesn't)
//   but since we link to the dll, we can use it from C++ too
extern "C" void gfxTracyInit();

int main(int argc, char *argv[]) {
#ifdef TRACY_ENABLE
  gfxTracyInit();
#endif

  shards::GetGlobals().RootPath = "./";
  (void)shardsInterface(SHARDS_CURRENT_ABI);
  int result = Catch::Session().run(argc, argv);

#ifdef TRACY_ENABLE
  shards::sleep(1.0);
#endif
  return result;
}

using namespace shards;

#define TEST_SERIALIZATION(_source_)            \
  Serialization ws;                             \
  std::vector<uint8_t> buffer;                  \
  BufferRefWriter w{buffer};                    \
  ws.serialize(_source_, w);                    \
  Var serialized(buffer.data(), buffer.size()); \
  SHVar output{};                               \
  Serialization rs;                             \
  VarReader r(serialized);                      \
  rs.reset();                                   \
  rs.deserialize(r, output);                    \
  REQUIRE(_source_ == output);                  \
  destroyVar(output)

TEST_CASE("SHType-type2Name", "[ops]") {
  REQUIRE_NOTHROW(type2Name(SHType::EndOfBlittableTypes));
  REQUIRE(type2Name(SHType::None) == "None");
  REQUIRE(type2Name(SHType::Any) == "Any");
  REQUIRE(type2Name(SHType::Object) == "Object");
  REQUIRE(type2Name(SHType::Enum) == "Enum");
  REQUIRE(type2Name(SHType::Bool) == "Bool");
  REQUIRE(type2Name(SHType::Bytes) == "Bytes");
  REQUIRE(type2Name(SHType::Color) == "Color");
  REQUIRE(type2Name(SHType::Int) == "Int");
  REQUIRE(type2Name(SHType::Int2) == "Int2");
  REQUIRE(type2Name(SHType::Int3) == "Int3");
  REQUIRE(type2Name(SHType::Int4) == "Int4");
  REQUIRE(type2Name(SHType::Int8) == "Int8");
  REQUIRE(type2Name(SHType::Int16) == "Int16");
  REQUIRE(type2Name(SHType::Float) == "Float");
  REQUIRE(type2Name(SHType::Float2) == "Float2");
  REQUIRE(type2Name(SHType::Float3) == "Float3");
  REQUIRE(type2Name(SHType::Float4) == "Float4");
  REQUIRE(type2Name(SHType::ShardRef) == "Shard");
  REQUIRE(type2Name(SHType::String) == "String");
  REQUIRE(type2Name(SHType::ContextVar) == "ContextVar");
  REQUIRE(type2Name(SHType::Path) == "Path");
  REQUIRE(type2Name(SHType::Image) == "Image");
  REQUIRE(type2Name(SHType::Seq) == "Seq");
  REQUIRE(type2Name(SHType::Table) == "Table");
  REQUIRE(type2Name(SHType::Audio) == "Audio");
}

TEST_CASE("SHVar-comparison", "[ops]") {
  SECTION("None, Any, EndOfBlittableTypes") {
    auto n1 = Var::Empty;
    auto n2 = Var::Empty;
    auto n3 = Var(100);
    REQUIRE(n1 == n2);
    REQUIRE(n1 != n3);
    auto hash1 = hash(n1);
    auto hash2 = hash(n2);
    REQUIRE(hash1 == hash2);
    SHLOG_INFO(n1); // logging coverage
    TEST_SERIALIZATION(n1);
  }

  SECTION("Bool") {
    auto t = Var::True;
    auto f = Var::False;
    auto t1 = Var(true);
    auto f1 = Var(false);
    REQUIRE(t == t1);
    REQUIRE(f == f1);
    REQUIRE(true == bool(t1));
    REQUIRE_FALSE(true == bool(f1));
    REQUIRE(false == bool(f1));
    REQUIRE(t > f);
    REQUIRE(t >= f);
    REQUIRE(f < t);
    REQUIRE(f <= t);
    auto hash1 = hash(t);
    auto hash2 = hash(t1);
    REQUIRE(hash1 == hash2);
    SHLOG_INFO(t); // logging coverage
    TEST_SERIALIZATION(t);
  }

  SECTION("Object") {
    int x = 10;
    int y = 20;
    auto o1 = Var::Object(&x, 10, 20);
    auto o2 = Var::Object(&x, 10, 20);
    auto o3 = Var::Object(&y, 10, 20);
    auto o4 = Var::Object(&y, 11, 20);
    auto empty = Var::Empty;
    REQUIRE(o1 == o2);
    REQUIRE(o1 != o3);
    REQUIRE(o3 != o4);
    REQUIRE_NOTHROW(o3 > o4);
    REQUIRE_NOTHROW(o3 >= o4);
    REQUIRE(o1 != empty);
    auto hash1 = hash(o1);
    auto hash2 = hash(o2);
    auto hash3 = hash(o3);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);
    SHLOG_INFO(o1); // logging coverage
  }

  SECTION("Audio") {
    std::vector<float> b1(1024);
    std::vector<float> b2(1024);
    std::vector<float> b3(1024);
    SHVar x{.payload = {.audioValue = SHAudio{44100, 512, 2, b1.data()}}, .valueType = SHType::Audio};
    SHVar y{.payload = {.audioValue = SHAudio{44100, 512, 2, b2.data()}}, .valueType = SHType::Audio};
    SHVar z{.payload = {.audioValue = SHAudio{44100, 1024, 1, b3.data()}}, .valueType = SHType::Audio};
    auto empty = Var::Empty;
    REQUIRE(x == y);
    REQUIRE(x != z);
    REQUIRE(y != z);
    REQUIRE_NOTHROW(x > y);
    REQUIRE_NOTHROW(x >= z);
    REQUIRE(x != empty);
    auto hash1 = hash(x);
    auto hash2 = hash(y);
    auto hash3 = hash(z);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);
    SHLOG_INFO(x); // logging coverage
    SHVar xx{};
    cloneVar(xx, x);
    TEST_SERIALIZATION(xx);
    destroyVar(xx);
  }

  SECTION("Enum") {
    int x = 10;
    int y = 20;
    auto o1 = Var::Enum(x, 10, 20);
    auto o2 = Var::Enum(x, 10, 20);
    auto o3 = Var::Enum(y, 10, 20);
    auto o4 = Var::Enum(y, 11, 20);
    auto empty = Var::Empty;
    REQUIRE(o1 == o2);
    REQUIRE(o1 != o3);
    REQUIRE(o3 != o4);
    REQUIRE(o1 < o3);
    REQUIRE_NOTHROW(o3 < o4);
    REQUIRE(o1 <= o3);
    REQUIRE_NOTHROW(o3 <= o4);
    REQUIRE_FALSE(o1 >= o3);
    REQUIRE(o3 >= o1);
    REQUIRE_NOTHROW(o3 >= o4);
    REQUIRE_NOTHROW(o4 >= o3);
    REQUIRE(o1 != empty);
    auto hash1 = hash(o1);
    auto hash2 = hash(o2);
    auto hash3 = hash(o3);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);
    SHLOG_INFO(o1); // logging coverage
    TEST_SERIALIZATION(o1);
  }

  SECTION("Float") {
    auto f1 = Var(10.0);
    REQUIRE(f1.valueType == SHType::Float);
    float ff = float(f1);
    REQUIRE(ff == 10);
    REQUIRE_THROWS(int(f1));
    auto f2 = Var(10.0);
    auto f3 = Var(22.0);
    auto i1 = Var(10);
    REQUIRE(f1 == f2);
    REQUIRE(f1 != f3);
    REQUIRE(f1 != i1);
    REQUIRE(f1 <= f2);
    REQUIRE(f1 <= f3);
    REQUIRE(f1 >= f2);
    REQUIRE_FALSE(f1 >= f3);
    REQUIRE(f1 < f3);
    REQUIRE((f3 > f1 and f3 > f2));
    REQUIRE((f3 >= f1 and f3 >= f2));
    auto hash1 = hash(f1);
    auto hash2 = hash(f2);
    auto hash3 = hash(f3);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);
    SHLOG_INFO(f1); // logging coverage
    TEST_SERIALIZATION(f1);
  }

#define SHVECTOR_TESTS              \
  REQUIRE(f1 == f2);                \
  REQUIRE(v1 == v2);                \
  REQUIRE(f1 != f3);                \
  REQUIRE(v1 != v3);                \
  REQUIRE(f1 != i1);                \
  REQUIRE_NOTHROW(f1 <= i1);        \
  REQUIRE(f1 <= f2);                \
  REQUIRE(v1 <= v2);                \
  REQUIRE_FALSE(f3 <= f1);          \
  REQUIRE_FALSE(v3 <= v1);          \
  REQUIRE(f1 < f4);                 \
  REQUIRE(v1 < v4);                 \
  REQUIRE(f1 >= f2);                \
  REQUIRE(v1 >= v2);                \
  REQUIRE(f1 < f3);                 \
  REQUIRE(v1 < v3);                 \
  REQUIRE((f3 > f1 and f3 > f2));   \
  REQUIRE((v3 > v1 and v3 > v2));   \
  REQUIRE((f4 >= f1 and f4 >= f2)); \
  REQUIRE((v4 >= v1 and v4 >= v2)); \
  REQUIRE_FALSE(f3 < f1);           \
  REQUIRE_FALSE(v3 < v1);           \
  auto hash1 = hash(f1);            \
  auto hash2 = hash(f2);            \
  auto hash3 = hash(f3);            \
  REQUIRE(hash1 == hash2);          \
  REQUIRE_FALSE(hash1 == hash3);    \
  SHLOG_INFO(f1);                   \
  TEST_SERIALIZATION(f1)

  SECTION("Float2") {
    auto f1 = Var(10.0, 2.0);
    std::vector<double> v1{10.0, 2.0};
    auto f2 = Var(10.0, 2.0);
    std::vector<double> v2{10.0, 2.0};
    auto f3 = Var(22.0, 2.0);
    std::vector<double> v3{22.0, 2.0};
    auto f4 = Var(22.0, 2.0);
    std::vector<double> v4{22.0, 2.0};
    auto i1 = Var(10);
    REQUIRE(f1.valueType == SHType::Float2);
    SHVECTOR_TESTS;
  }

  SECTION("Float3") {
    auto f1 = Var(10.0, 2.0, 3.0);
    std::vector<float> v1{10.0, 2.0, 3.0};
    auto f2 = Var(10.0, 2.0, 3.0);
    std::vector<float> v2{10.0, 2.0, 3.0};
    auto f3 = Var(22.0, 2.0, 1.0);
    std::vector<float> v3{22.0, 2.0, 1.0};
    auto f4 = Var(22.0, 2.0, 4.0);
    std::vector<float> v4{22.0, 2.0, 4.0};
    auto i1 = Var(10);
    REQUIRE(f1.valueType == SHType::Float3);
    SHVECTOR_TESTS;
  }

  SECTION("Float4") {
    auto f1 = Var(10.0, 2.0, 3.0, 3.0);
    std::vector<float> v1{10.0, 2.0, 3.0, 3.0};
    auto f2 = Var(10.0, 2.0, 3.0, 3.0);
    std::vector<float> v2{10.0, 2.0, 3.0, 3.0};
    auto f3 = Var(22.0, 2.0, 1.0, 3.0);
    std::vector<float> v3{22.0, 2.0, 1.0, 3.0};
    auto f4 = Var(22.0, 2.0, 4.0, 3.0);
    std::vector<float> v4{22.0, 2.0, 4.0, 3.0};
    auto i1 = Var(10);
    REQUIRE(f1.valueType == SHType::Float4);
    SHVECTOR_TESTS;
  }

  SECTION("Int") {
    auto f1 = Var(10);
    REQUIRE(f1.valueType == SHType::Int);
    int ff = int(f1);
    REQUIRE(ff == 10);
    REQUIRE_NOTHROW(float(f1)); // will convert to float automatically
    auto f2 = Var(10);
    auto f3 = Var(22);
    auto i1 = Var(10.0);
    REQUIRE(f1 == f2);
    REQUIRE(f1 != f3);
    REQUIRE(f1 != i1);
    REQUIRE(f1 <= f2);
    REQUIRE(f1 <= f3);
    REQUIRE(f1 >= f2);
    REQUIRE_FALSE(f1 >= f3);
    REQUIRE(f1 < f3);
    REQUIRE((f3 > f1 and f3 > f2));
    REQUIRE((f3 >= f1 and f3 >= f2));
    auto hash1 = hash(f1);
    auto hash2 = hash(f2);
    auto hash3 = hash(f3);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);
    SHLOG_INFO(f1); // logging coverage
  }

  SECTION("Int2") {
    auto f1 = Var(10, 2);
    std::vector<int64_t> v1{10, 2};
    auto f2 = Var(10, 2);
    std::vector<int64_t> v2{10, 2};
    auto f3 = Var(22, 2);
    std::vector<int64_t> v3{22, 2};
    auto f4 = Var(22, 2);
    std::vector<int64_t> v4{22, 2};
    auto i1 = Var(10.0);
    REQUIRE(f1.valueType == SHType::Int2);
    SHVECTOR_TESTS;
  }

  SECTION("Int3") {
    auto f1 = Var(10, 2, 3);
    std::vector<int32_t> v1{10, 2, 3};
    auto f2 = Var(10, 2, 3);
    std::vector<int32_t> v2{10, 2, 3};
    auto f3 = Var(22, 2, 1);
    std::vector<int32_t> v3{22, 2, 1};
    auto f4 = Var(22, 2, 4);
    std::vector<int32_t> v4{22, 2, 4};
    auto i1 = Var(10);
    REQUIRE(f1.valueType == SHType::Int3);
    SHVECTOR_TESTS;
  }

  SECTION("Int4") {
    auto f1 = Var(10, 2, 3, 0);
    std::vector<int32_t> v1{10, 2, 3, 0};
    auto f2 = Var(10, 2, 3, 0);
    std::vector<int32_t> v2{10, 2, 3, 0};
    auto f3 = Var(22, 2, 1, 0);
    std::vector<int32_t> v3{22, 2, 1, 0};
    auto f4 = Var(22, 2, 4, 0);
    std::vector<int32_t> v4{22, 2, 4, 0};
    auto i1 = Var(10);
    REQUIRE(f1.valueType == SHType::Int4);
    SHVECTOR_TESTS;
  }

  SECTION("Int8") {
    auto f1 = Var(10, 2, 3, 0, 1, 2, 3, 4);
    std::vector<int16_t> v1{10, 2, 3, 0, 1, 2, 3, 4};
    auto f2 = Var(10, 2, 3, 0, 1, 2, 3, 4);
    std::vector<int16_t> v2{10, 2, 3, 0, 1, 2, 3, 4};
    auto f3 = Var(22, 2, 1, 0, 1, 2, 3, 4);
    std::vector<int16_t> v3{22, 2, 1, 0, 1, 2, 3, 4};
    auto f4 = Var(22, 2, 4, 0, 1, 2, 3, 4);
    std::vector<int16_t> v4{22, 2, 4, 0, 1, 2, 3, 4};
    auto i1 = Var(10);
    REQUIRE(f1.valueType == SHType::Int8);
    SHVECTOR_TESTS;
  }

  SECTION("Int16") {
    auto f1 = Var(10, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2);
    std::vector<int8_t> v1{10, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2};
    auto f2 = Var(10, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2);
    std::vector<int8_t> v2{10, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2};
    auto f3 = Var(22, 2, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2);
    std::vector<int8_t> v3{22, 2, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2};
    auto f4 = Var(22, 2, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2);
    std::vector<int8_t> v4{22, 2, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2};
    auto i1 = Var(10);
    REQUIRE(f1.valueType == SHType::Int16);
    SHVECTOR_TESTS;
  }

#undef SHVECTOR_TESTS

  SECTION("Color") {
    auto c1 = Var(SHColor{20, 20, 20, 255});
    auto c2 = Var(SHColor{20, 20, 20, 255});
    auto c3 = Var(SHColor{20, 20, 20, 0});
    REQUIRE(c1 == c2);
    REQUIRE(c2 >= c1);
    REQUIRE(c1 != c3);
    REQUIRE(c1 > c3);
    REQUIRE(c1 >= c3);
    REQUIRE(c3 < c1);
    REQUIRE(c3 <= c1);
    auto hash1 = hash(c1);
    auto hash2 = hash(c2);
    auto hash3 = hash(c3);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);
    SHLOG_INFO(c1);
  }

  SECTION("String") {
    auto s1 = Var("Hello world");
    REQUIRE(s1.valueType == SHType::String);
    auto s2 = Var("Hello world");
    auto s3 = Var("Hello world 2");
    auto s5 = SHVar();
    s5.valueType = SHType::String;
    s5.payload.stringValue = "Hello world";
    std::string q("qwerty");
    auto s4 = Var(q);
    REQUIRE_THROWS(float(s4));
    REQUIRE(s1 == s2);
    REQUIRE(s1 == s5);
    REQUIRE(s2 == s5);
    REQUIRE(s2 <= s5);
    REQUIRE_FALSE(s2 < s5);
    REQUIRE(s1 != s3);
    REQUIRE(s1 != s4);
    REQUIRE(s1 < s3);
    REQUIRE(s1 <= s3);
    REQUIRE(s3 >= s2);
    REQUIRE(s3 > s2);
    auto hash1 = hash(s1);
    auto hash2 = hash(s2);
    auto hash3 = hash(s3);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);
    SHLOG_INFO(s1); // logging coverage
  }

  SECTION("std::vector-Var") {
    std::vector<Var> s1{Var(10), Var(20), Var(30), Var(40), Var(50)};
    Var v1(s1);
    REQUIRE(v1.valueType == SHType::Seq);
    std::vector<Var> s2{Var(10), Var(20), Var(30), Var(40), Var(50)};
    Var v2(s2);
    REQUIRE(v2.valueType == SHType::Seq);
    std::vector<Var> s3{Var(100), Var(200), Var(300)};
    Var v3(s3);
    REQUIRE(v3.valueType == SHType::Seq);
    std::vector<Var> s4{Var(10), Var(20), Var(30)};
    Var v4(s4);
    REQUIRE(v4.valueType == SHType::Seq);
    std::vector<Var> s5{Var(1), Var(2), Var(3), Var(4), Var(5), Var(6)};
    Var v5(s5);
    REQUIRE(v5.valueType == SHType::Seq);
    std::vector<Var> s6{Var(10), Var(20), Var(30), Var(45), Var(55)};
    Var v6(s6);
    REQUIRE(v6.valueType == SHType::Seq);
    std::vector<Var> s7{Var(10), Var(20), Var(30), Var(25), Var(35)};
    Var v7(s7);
    std::vector<Var> s77 = std::vector<Var>(v7);
    std::vector<int> i77 = std::vector<int>(v7);
    REQUIRE_THROWS(([&]() {
      std::vector<bool> f77 = std::vector<bool>(v7);
      return true;
    }()));
    REQUIRE_THROWS(([&]() {
      Var empty{};
      std::vector<Var> f77 = std::vector<Var>(empty);
      return true;
    }()));
    REQUIRE(i77[2] == 30);
    REQUIRE(s7 == s77);
    REQUIRE(v7.valueType == SHType::Seq);
    std::vector<Var> s8{Var(10)};
    Var v8(s8);
    REQUIRE(v8.valueType == SHType::Seq);
    std::vector<Var> s9{Var(1)};
    Var v9(s9);
    REQUIRE(v9.valueType == SHType::Seq);

    REQUIRE_FALSE(v8 < v9);
    REQUIRE_FALSE(v8 <= v9);
    REQUIRE_FALSE(s8 < s9);
    REQUIRE_FALSE(s8 <= s9);
    REQUIRE(s1 == s2);
    REQUIRE(v1 == v2);
    REQUIRE_FALSE(v1 < v2);
    REQUIRE(v1 <= v2);
    REQUIRE_FALSE(s4 <= s8);
    REQUIRE_FALSE(v4 <= v8);
    REQUIRE(s1 <= s2);
    REQUIRE(s1 <= s2);
    REQUIRE(s1 != s5);
    REQUIRE(v1 != v5);
    REQUIRE(s1 != s6);
    REQUIRE(v1 != v6);
    REQUIRE(s1 != s3);
    REQUIRE(v1 != v3);
    REQUIRE(s2 != s3);
    REQUIRE(v2 != v3);
    REQUIRE(s3 > s1);
    REQUIRE(v3 > v1);
    REQUIRE(s1 < s3);
    REQUIRE(v1 < v3);
    REQUIRE(s2 > s4);
    REQUIRE(v2 > v4);
    REQUIRE(s2 > s5);
    REQUIRE(v2 > v5);
    REQUIRE(s3 >= s1);
    REQUIRE(v3 >= v1);
    REQUIRE(s1 <= s3);
    REQUIRE(v1 <= v3);
    REQUIRE(s2 >= s4);
    REQUIRE(v2 >= v4);
    REQUIRE(s2 >= s5);
    REQUIRE(v2 >= v5);
    REQUIRE(v6 > v1);
    REQUIRE(s6 > s1);
    REQUIRE(v6 >= v1);
    REQUIRE(s6 >= s1);
    REQUIRE(v1 < v6);
    REQUIRE(s1 < s6);
    REQUIRE(v1 <= v6);
    REQUIRE(s1 <= s6);
    REQUIRE_FALSE(v6 < v1);
    REQUIRE_FALSE(s6 < s1);
    REQUIRE_FALSE(v6 <= v1);
    REQUIRE_FALSE(s6 <= s1);
    REQUIRE_FALSE(v1 > v6);
    REQUIRE_FALSE(s1 > s6);
    REQUIRE_FALSE(v1 >= v6);
    REQUIRE_FALSE(s1 >= s6);
    REQUIRE_FALSE(v7 > v1);
    REQUIRE_FALSE(s7 > s1);
    REQUIRE_FALSE(v7 >= v1);
    REQUIRE_FALSE(s7 >= s1);
    REQUIRE_FALSE(v1 < v7);
    REQUIRE_FALSE(s1 < s7);
    REQUIRE_FALSE(v1 <= v7);
    REQUIRE_FALSE(s1 <= s7);
    REQUIRE(v7 < v1);
    REQUIRE(s7 < s1);
    REQUIRE(v7 <= v1);
    REQUIRE(s7 <= s1);
    REQUIRE(v1 > v7);
    REQUIRE(s1 > s7);
    REQUIRE(v1 >= v7);
    REQUIRE(s1 >= s7);

    auto hash1 = hash(v1);
    auto hash2 = hash(v2);
    auto hash3 = hash(v3);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);

    SHLOG_INFO(v1);
  }

  SECTION("std::vector-SHVar") {
    std::vector<SHVar> s1{Var(10), Var(20), Var(30), Var(40), Var(50)};
    Var v1(s1);
    REQUIRE(v1.valueType == SHType::Seq);
    std::vector<SHVar> s2{Var(10), Var(20), Var(30), Var(40), Var(50)};
    Var v2(s2);
    REQUIRE(v2.valueType == SHType::Seq);
    std::vector<SHVar> s3{Var(100), Var(200), Var(300)};
    Var v3(s3);
    REQUIRE(v3.valueType == SHType::Seq);
    std::vector<SHVar> s4{Var(10), Var(20), Var(30)};
    Var v4(s4);
    REQUIRE(v4.valueType == SHType::Seq);
    std::vector<SHVar> s5{Var(1), Var(2), Var(3), Var(4), Var(5), Var(6)};
    Var v5(s5);
    REQUIRE(v5.valueType == SHType::Seq);
    std::vector<SHVar> s6{Var(1), Var(2.2), Var(3), Var(4), Var("Hello"), Var(6)};
    Var v6(s6);
    REQUIRE(v6.valueType == SHType::Seq);

    REQUIRE(s1 == s2);
    REQUIRE(v1 == v2);
    REQUIRE(s1 != s3);
    REQUIRE(v1 != v3);
    REQUIRE(s2 != s3);
    REQUIRE(v2 != v3);
    REQUIRE(s3 > s1);
    REQUIRE(v3 > v1);
    REQUIRE(s1 < s3);
    REQUIRE(v1 < v3);
    REQUIRE(s2 > s4);
    REQUIRE(v2 > v4);
    REQUIRE(s2 > s5);
    REQUIRE(v2 > v5);
    REQUIRE(s3 >= s1);
    REQUIRE(v3 >= v1);
    REQUIRE(s1 <= s3);
    REQUIRE(v1 <= v3);
    REQUIRE(s2 >= s4);
    REQUIRE(v2 >= v4);
    REQUIRE(s2 >= s5);
    REQUIRE(v2 >= v5);

    auto h1 = deriveTypeHash64(v1);
    auto h2 = deriveTypeHash64(v2);
    REQUIRE(h1 == h2);
    auto h6 = deriveTypeHash64(v6);
    REQUIRE(h1 != h6);

    SHLOG_INFO(v1);
  }

  SECTION("std::array-Var") {
    std::array<Var, 5> s1{Var(10), Var(20), Var(30), Var(40), Var(50)};
    Var v1(s1);
    REQUIRE(v1.valueType == SHType::Seq);
    std::array<Var, 5> s2{Var(10), Var(20), Var(30), Var(40), Var(50)};
    Var v2(s2);
    REQUIRE(v2.valueType == SHType::Seq);
    std::array<Var, 3> s3{Var(100), Var(200), Var(300)};
    Var v3(s3);
    REQUIRE(v3.valueType == SHType::Seq);
    std::array<Var, 3> s4{Var(10), Var(20), Var(30)};
    Var v4(s4);
    REQUIRE(v4.valueType == SHType::Seq);
    std::array<Var, 6> s5{Var(1), Var(2), Var(3), Var(4), Var(5), Var(6)};
    Var v5(s5);
    REQUIRE(v5.valueType == SHType::Seq);

    REQUIRE(v1 == v2);
    REQUIRE(v1 != v3);
    REQUIRE(v2 != v3);
    REQUIRE(v3 > v1);
    REQUIRE(v1 < v3);
    REQUIRE(v2 > v4);
    REQUIRE(v2 > v5);
    REQUIRE(v3 >= v1);
    REQUIRE(v1 <= v3);
    REQUIRE(v2 >= v4);
    REQUIRE(v2 >= v5);

    SHLOG_INFO(v1);
  }

  SECTION("std::array-SHVar") {
    std::array<SHVar, 5> s1{Var(10), Var(20), Var(30), Var(40), Var(50)};
    Var v1(s1);
    REQUIRE(v1.valueType == SHType::Seq);
    std::array<SHVar, 5> s2{Var(10), Var(20), Var(30), Var(40), Var(50)};
    Var v2(s2);
    REQUIRE(v2.valueType == SHType::Seq);
    std::array<SHVar, 3> s3{Var(100), Var(200), Var(300)};
    Var v3(s3);
    REQUIRE(v3.valueType == SHType::Seq);
    std::array<SHVar, 3> s4{Var(10), Var(20), Var(30)};
    Var v4(s4);
    REQUIRE(v4.valueType == SHType::Seq);
    std::array<SHVar, 6> s5{Var(1), Var(2), Var(3), Var(4), Var(5), Var(6)};
    Var v5(s5);
    REQUIRE(v5.valueType == SHType::Seq);

    REQUIRE(v1 == v2);
    REQUIRE(v1 != v3);
    REQUIRE(v2 != v3);
    REQUIRE(v3 > v1);
    REQUIRE(v1 < v3);
    REQUIRE(v2 > v4);
    REQUIRE(v2 > v5);
    REQUIRE(v3 >= v1);
    REQUIRE(v1 <= v3);
    REQUIRE(v2 >= v4);
    REQUIRE(v2 >= v5);

    SHLOG_INFO(v1);
  }

  SECTION("Bytes") {
    std::random_device rd;
    std::uniform_int_distribution<uint32_t> dist(0, 0x07);

    std::vector<uint8_t> data1(1024);
    for (auto &b : data1) {
      b = dist(rd);
    }
    Var v1{&data1.front(), uint32_t(data1.size())};

    auto data2 = data1;
    Var v2{&data2.front(), uint32_t(data2.size())};

    std::vector<uint8_t> data3(1024);
    for (auto &b : data3) {
      b = dist(rd);
    }
    Var v3{&data3.front(), uint32_t(data3.size())};

    std::vector<uint8_t> data4(1050);
    for (auto &b : data4) {
      b = dist(rd);
    }
    Var v4{&data4.front(), uint32_t(data4.size())};

    std::vector<uint8_t> data5(200);
    for (auto &b : data5) {
      b = dist(rd);
    }
    Var v5{&data5.front(), uint32_t(data5.size())};

    REQUIRE((&data1[0]) != (&data2[0]));
    REQUIRE(data1.size() == 1024);

    REQUIRE(v1.valueType == SHType::Bytes);

    REQUIRE(data1 == data2);
    REQUIRE(v1 == v2);

    REQUIRE(data1 != data3);
    REQUIRE(v1 != v3);

    REQUIRE(data1 != data3);
    REQUIRE(v1 != v3);

    REQUIRE(data1 >= data2);
    REQUIRE(v1 >= v2);

    REQUIRE_FALSE(data1 < data2);
    REQUIRE_FALSE(v1 < v2);

    REQUIRE((v1 > v5) == (data1 > data5));
    REQUIRE((v1 > v4) == (data1 > data4));

    REQUIRE(v1 == v1);
    REQUIRE(v1 <= v1);
    REQUIRE_FALSE(v1 < v1);

    auto hash1 = hash(v1);
    auto hash2 = hash(v2);
    auto hash3 = hash(v3);
    REQUIRE(hash1 == hash2);
    REQUIRE_FALSE(hash1 == hash3);

    SHLOG_INFO(v1);
  }

  SECTION("Shard") {
    Shard foo;
    Shard bar;
    auto b1 = SHVar();
    b1.valueType = SHType::ShardRef;
    b1.payload.shardValue = &foo;
    auto b2 = SHVar();
    b2.valueType = SHType::ShardRef;
    b2.payload.shardValue = &bar;
    REQUIRE_FALSE(b1 == b2);
    REQUIRE(b1 != b2);
  }
}

TEST_CASE("VarPayload") {
  SECTION("Int2") {
    VarPayload a1 = Int2VarPayload({2, 1});
    Var v1(std::get<Int2VarPayload>(a1));
    Int2VarPayload a2({2, 1});
    Var v2(a2);
    Var v3(2, 1);
    Int2VarPayload a4 = {2, 1};
    Var v4(a4);

    REQUIRE(v1 == v2);
    REQUIRE(v1 == v3);
    REQUIRE(v2 == v3);
    REQUIRE(v1 == v4);
  }

  SECTION("Float4") {
    VarPayload a1 = Float4VarPayload({2.0, 1.0, 3.0, 9.0});
    Var v1(std::get<Float4VarPayload>(a1));
    Float4VarPayload a2({2.0, 1.0, 3.0, 9.0});
    Var v2(a2);
    Var v3(2.0, 1.0, 3.0, 9.0);
    Float4VarPayload a4 = {2.0, 1.0, 3.0, 9.0};
    Var v4(a4);

    REQUIRE(v1 == v2);
    REQUIRE(v1 == v3);
    REQUIRE(v2 == v3);
    REQUIRE(v1 == v4);
  }
}

#define SET_TABLE_COMMON_TESTS                                 \
  SHInstanceData data{};                                       \
  auto hash1 = hash(vxx);                                      \
  auto hash2 = hash(vx);                                       \
  auto hash3 = hash(vyy);                                      \
  REQUIRE(hash1 == hash2);                                     \
  REQUIRE(hash1 == hash3);                                     \
  auto typeHash1 = deriveTypeHash64(vxx);                      \
  auto typeInfo1 = deriveTypeInfo(vxx, data);                  \
  auto typeInfo2 = deriveTypeInfo(vx, data);                   \
  auto typeInfo3 = deriveTypeInfo(Var::Empty, data);           \
  REQUIRE(deriveTypeHash64(typeInfo1) == typeHash1);           \
  auto stypeHash = std::hash<SHTypeInfo>()(typeInfo1);         \
  REQUIRE(stypeHash != 0);                                     \
  REQUIRE(typeInfo1 == typeInfo2);                             \
  REQUIRE(typeInfo1 != typeInfo3);                             \
  freeDerivedInfo(typeInfo1);                                  \
  freeDerivedInfo(typeInfo2);                                  \
  freeDerivedInfo(typeInfo3);                                  \
  auto typeHash2 = deriveTypeHash64(vx);                       \
  auto typeHash3 = deriveTypeHash64(vyy);                      \
  SHLOG_INFO("{} - {} - {}", typeHash1, typeHash2, typeHash3); \
  REQUIRE(typeHash1 == typeHash2);                             \
  REQUIRE(typeHash1 == typeHash3);                             \
  vyy = vz;                                                    \
  REQUIRE(vyy != vy);                                          \
  auto typeHash4 = deriveTypeHash64(vyy);                      \
  REQUIRE(typeHash4 == typeHash3);                             \
  auto typeHash5 = deriveTypeHash64(vz);                       \
  REQUIRE(typeHash4 == typeHash5)

TEST_CASE("SHMap") {
  SHMap x;
  x.emplace(Var("x"), Var(10));
  x.emplace(Var("y"), Var("Hello Set"));
  SHVar vx{};
  vx.valueType = SHType::Table;
  vx.payload.tableValue.opaque = &x;
  vx.payload.tableValue.api = &GetGlobals().TableInterface;

  SHMap y;
  y.emplace(Var("y"), Var("Hello Set"));
  y.emplace(Var("x"), Var(10));

  SHVar vy{};
  vy.valueType = SHType::Table;
  vy.payload.tableValue.opaque = &y;
  vy.payload.tableValue.api = &GetGlobals().TableInterface;

  SHMap z;
  z.emplace(Var("y"), Var("Hello Set"));
  z.emplace(Var("x"), Var(11));

  SHVar vz{};
  vz.valueType = SHType::Table;
  vz.payload.tableValue.opaque = &z;
  vz.payload.tableValue.api = &GetGlobals().TableInterface;

  REQUIRE(vx == vy);

  OwnedVar vxx = vx;
  OwnedVar vyy = vy;
  REQUIRE(vxx == vyy);

  int items = 0;
  ForEach(vxx.payload.tableValue, [&](auto k, auto &v) {
    REQUIRE(vx.payload.tableValue.api->tableContains(vx.payload.tableValue, k));
    items++;
  });
  REQUIRE(items == 2);

  SET_TABLE_COMMON_TESTS;

  REQUIRE(x.size() == 2);
  REQUIRE(x.count(Var("x")) == 1);
  x.erase(Var("x"));
  REQUIRE(x.count(Var("x")) == 0);
  REQUIRE(x.count(Var("y")) == 1);
  x.erase(Var("y"));
  REQUIRE(x.count(Var("y")) == 0);

  REQUIRE(vx != vy);
}

TEST_CASE("CXX-Wire-DSL") {
  // TODO, improve this
  auto wire = shards::Wire("test-wire").looped(true).let(1).shard("Log").shard("Math.Add", 2).shard("Assert.Is", 3, true);
  assert(wire->shards.size() == 4);
}

TEST_CASE("DynamicArray") {
  SHSeq ts{};
  Var a{0}, b{1}, c{2}, d{3}, e{4}, f{5};
  arrayPush(ts, a);
  assert(ts.len == 1);
  assert(ts.cap == 4);
  arrayPush(ts, b);
  arrayPush(ts, c);
  arrayPush(ts, d);
  arrayPush(ts, e);
  assert(ts.len == 5);
  assert(ts.cap == 8);

  assert(ts.elements[0] == Var(0));
  assert(ts.elements[1] == Var(1));
  assert(ts.elements[2] == Var(2));
  assert(ts.elements[3] == Var(3));
  assert(ts.elements[4] == Var(4));

  arrayInsert(ts, 1, f);

  assert(ts.elements[0] == Var(0));
  assert(ts.elements[1] == Var(5));
  assert(ts.elements[2] == Var(1));
  assert(ts.elements[3] == Var(2));
  assert(ts.elements[4] == Var(3));

  arrayDel(ts, 2);

  assert(ts.elements[0] == Var(0));
  assert(ts.elements[1] == Var(5));
  assert(ts.elements[2] == Var(2));
  assert(ts.elements[3] == Var(3));
  assert(ts.elements[4] == Var(4));

  arrayDelFast(ts, 2);

  assert(ts.elements[0] == Var(0));
  assert(ts.elements[1] == Var(5));
  assert(ts.elements[2] == Var(4));
  assert(ts.elements[3] == Var(3));

  assert(ts.len == 4);

  arrayFree(ts);
  assert(ts.elements == nullptr);
  assert(ts.len == 0);
  assert(ts.cap == 0);
}

TEST_CASE("Type") {
  SECTION("TableOf") {
    Types VerticesSeqTypes{{CoreInfo::FloatType, CoreInfo::Float2Type, CoreInfo::Float3Type, CoreInfo::ColorType}};
    Type VerticesSeq = Type::SeqOf(VerticesSeqTypes);
    Types IndicesSeqTypes{{
        CoreInfo::IntType,  // Triangle strip
        CoreInfo::Int2Type, // Line list
        CoreInfo::Int3Type  // Triangle list
    }};
    Type IndicesSeq = Type::SeqOf(IndicesSeqTypes);
    Types InputTableTypes{{VerticesSeq, IndicesSeq}};
    std::array<SHVar, 2> InputTableKeys{Var("Vertices"), Var("Indices")};
    Type InputTable = Type::TableOf(InputTableTypes, InputTableKeys);

    SHTypeInfo t1 = InputTable;
    REQUIRE(t1.basicType == SHType::Table);
    REQUIRE(t1.table.types.len == 2);
    REQUIRE(t1.table.keys.len == 2);
    REQUIRE(t1.table.keys.elements[0] == Var("Vertices"));
    REQUIRE(t1.table.keys.elements[1] == Var("Indices"));
    SHTypeInfo verticesSeq = VerticesSeq;
    SHTypeInfo indicesSeq = IndicesSeq;
    REQUIRE(t1.table.types.elements[0] == verticesSeq);
    REQUIRE(t1.table.types.elements[1] == indicesSeq);
  }
}

TEST_CASE("ObjectVar") {
  SECTION("WireUse") {
    ObjectVar<int> myobject{"ObjectVarTestObject1", 100, 1};
    int *o1 = myobject.New();
    *o1 = 1000;
    SHVar v1 = myobject.Get(o1);
    REQUIRE(v1.valueType == SHType::Object);
    int *vo1 = reinterpret_cast<int *>(v1.payload.objectValue);
    REQUIRE(*vo1 == 1000);
    REQUIRE(vo1 == o1);

    // check internal ref count
    // this is internal magic but needed for testing
    struct ObjectRef {
      int shared;
      uint32_t refcount;
    };
    auto or1 = reinterpret_cast<ObjectRef *>(o1);
    REQUIRE(or1->refcount == 1);

    auto wire = shards::Wire("test-wire-ObjectVar")
                    .let(v1)
                    .shard("Set", "v1"_sv)
                    .shard("Set", "v2"_sv)
                    .shard("Get", "v1"_sv)
                    .shard("Is", Var::ContextVar("v2"_sv))
                    .shard("Assert.Is", true);
    myobject.Release(o1);
    auto mesh = SHMesh::make();
    mesh->schedule(wire);
    REQUIRE(mesh->tick());       // false is wire errors happened
    REQUIRE(or1->refcount == 1); // will be 0 when wire goes out of scope
  }
}

TEST_CASE("linalg compatibility") {
  static_assert(sizeof(SHVar) == sizeof(padded::Float4));
  static_assert(sizeof(SHVar) == sizeof(padded::Float3));
  static_assert(sizeof(SHVar) == sizeof(padded::Float2));
  static_assert(sizeof(SHVar) == sizeof(padded::Float));
  static_assert(sizeof(SHVar) == sizeof(padded::Int4));
  static_assert(sizeof(SHVar) == sizeof(padded::Int3));
  static_assert(sizeof(SHVar) == sizeof(padded::Int2));
  static_assert(sizeof(SHVar) == sizeof(padded::Int));
  static_assert(sizeof(SHVar) * 4 == sizeof(Mat4));
  static_assert(sizeof(padded::Float4) * 4 == sizeof(Mat4));
  static_assert(offsetof(SHVar, valueType) == offsetof(padded::Float4, valueType));
  static_assert(offsetof(SHVar, valueType) == offsetof(padded::Float3, valueType));
  static_assert(offsetof(SHVar, valueType) == offsetof(padded::Float2, valueType));
  static_assert(offsetof(SHVar, valueType) == offsetof(padded::Float, valueType));
  static_assert(offsetof(SHVar, valueType) == offsetof(padded::Int4, valueType));
  static_assert(offsetof(SHVar, valueType) == offsetof(padded::Int3, valueType));
  static_assert(offsetof(SHVar, valueType) == offsetof(padded::Int2, valueType));
  static_assert(offsetof(SHVar, valueType) == offsetof(padded::Int, valueType));
  static_assert(0 == offsetof(Mat4, x));
  static_assert(sizeof(SHVar) * 1 == offsetof(Mat4, y));
  static_assert(sizeof(SHVar) * 2 == offsetof(Mat4, z));
  static_assert(sizeof(SHVar) * 3 == offsetof(Mat4, w));
  static_assert(alignof(SHVar) == alignof(padded::Float4));
  static_assert(alignof(SHVar) == alignof(padded::Float3));

  Var a{1.0, 2.0, 3.0, 4.0};
  Var b{4.0, 3.0, 2.0, 1.0};
  const padded::Float4 *va = reinterpret_cast<padded::Float4 *>(&a.payload.float4Value);
  const padded::Float4 *vb = reinterpret_cast<padded::Float4 *>(&b.payload.float4Value);
  auto c = (**va) + (**vb);
  linalg::aliases::float4 rc{5.0, 5.0, 5.0, 5.0};
  REQUIRE(c == rc);
  c = (**va) * (**vb);
  rc = {4.0, 6.0, 6.0, 4.0};
  REQUIRE(c == rc);

  std::vector<Var> ad{Var(1.0, 2.0, 3.0, 4.0), Var(1.0, 2.0, 3.0, 4.0), Var(1.0, 2.0, 3.0, 4.0), Var(1.0, 2.0, 3.0, 4.0)};
  Var d(ad);
  const Mat4 *md = reinterpret_cast<Mat4 *>(&d.payload.seqValue.elements[0]);
  linalg::aliases::float4x4 rd{{1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}};
  REQUIRE(linalg::aliases::float4x4(*md) == rd);

  Mat4 sm{rd};
  REQUIRE(*sm.x == rd.x);
  REQUIRE(*sm.y == rd.y);
}

enum class XRHand { Left, Right };

struct GamePadTable : public TableVar {
  GamePadTable()
      : TableVar(),                           //
        buttons(get<SeqVar>(Var("buttons"))), //
        sticks(get<SeqVar>(Var("sticks"))),   //
        id((*this)[Var("id")]),               //
        connected((*this)[Var("connected")]) {
    connected = Var(false);
  }

  SeqVar &buttons;
  SeqVar &sticks;
  SHVar &id;
  SHVar &connected;
};

struct HandTable : public GamePadTable {
  DECL_ENUM_INFO(XRHand, XrHand, "XR hand", 'xrha');

  HandTable()
      : GamePadTable(),                           //
        handedness((*this)[Var("handedness")]),   //
        transform(get<SeqVar>(Var("transform"))), //
        inverseTransform(get<SeqVar>(Var("inverseTransform"))) {
    handedness = Var::Enum(XRHand::Left, XrHandEnumInfo::VendorId, XrHandEnumInfo::TypeId);
  }

  SHVar &handedness;
  SeqVar &transform;
  SeqVar &inverseTransform;
};

TEST_CASE("TableVar") {
  HandTable hand;
  hand.buttons.push_back(Var(0.6, 1.0, 1.0));
  SHLOG_INFO(hand);
}

#include <shards/number_types.hpp>

TEST_CASE("Number Types") {
  NumberTypeLookup typeLookup;

  CHECK(typeLookup.get(NumberType::Invalid) == nullptr);

  CHECK(typeLookup.get(NumberType::UInt8)->size == 1);
  CHECK(typeLookup.get(NumberType::Int8)->size == 1);
  CHECK(typeLookup.get(NumberType::Int16)->size == 2);
  CHECK(typeLookup.get(NumberType::Int32)->size == 4);
  CHECK(typeLookup.get(NumberType::Int64)->size == 8);
  CHECK(typeLookup.get(NumberType::Float32)->size == 4);
  CHECK(typeLookup.get(NumberType::Float64)->size == 8);

  CHECK(typeLookup.get(NumberType::UInt8)->isInteger);
  CHECK(typeLookup.get(NumberType::Int8)->isInteger);
  CHECK(typeLookup.get(NumberType::Int16)->isInteger);
  CHECK(typeLookup.get(NumberType::Int32)->isInteger);
  CHECK(typeLookup.get(NumberType::Int64)->isInteger);
  CHECK(!typeLookup.get(NumberType::Float32)->isInteger);
  CHECK(!typeLookup.get(NumberType::Float64)->isInteger);

  CHECK(typeLookup.get(SHType::Int) == typeLookup.get(NumberType::Int64));
  CHECK(typeLookup.get(SHType::Int2) == typeLookup.get(NumberType::Int64));
  CHECK(typeLookup.get(SHType::Int3) == typeLookup.get(NumberType::Int32));
  CHECK(typeLookup.get(SHType::Int4) == typeLookup.get(NumberType::Int32));
  CHECK(typeLookup.get(SHType::Int8) == typeLookup.get(NumberType::Int16));
  CHECK(typeLookup.get(SHType::Int16) == typeLookup.get(NumberType::Int8));
  CHECK(typeLookup.get(SHType::Color) == typeLookup.get(NumberType::UInt8));
  CHECK(typeLookup.get(SHType::Float) == typeLookup.get(NumberType::Float64));
  CHECK(typeLookup.get(SHType::Float2) == typeLookup.get(NumberType::Float64));
  CHECK(typeLookup.get(SHType::Float3) == typeLookup.get(NumberType::Float32));
  CHECK(typeLookup.get(SHType::Float4) == typeLookup.get(NumberType::Float32));

  double d = 3.14;
  float f = 0.f;
  typeLookup.getConversion(NumberType::Float64, NumberType::Float32)->convertOne(&d, &f);
  CHECK(f == 3.14f);

  int32_t i32 = 0;
  typeLookup.getConversion(NumberType::Float32, NumberType::Int32)->convertOne(&f, &i32);
  CHECK(i32 == 3);

  uint8_t u8 = 0;
  typeLookup.getConversion(NumberType::Float32, NumberType::UInt8)->convertOne(&f, &u8);
  CHECK(u8 == 3);

  typeLookup.getConversion(NumberType::UInt8, NumberType::Float64)->convertOne(&u8, &d);
  CHECK(d == 3.0);

  SECTION("Vector take") {
    float float4v[4] = {1.1f, 2.2f, 3.3f, 4.4f};
    double float2v[2] = {0};

    std::vector<Var> vec{Var(0)};
    Var seq = Var(vec);
    CHECK_NOTHROW(typeLookup.getConversion(NumberType::Float32, NumberType::Float64)
                      ->convertMultipleSeq(float4v, float2v, 4, seq.payload.seqValue));
    CHECK(float2v[0] == (double)1.1f);
    CHECK(float2v[1] == 0.0);

    vec = {Var(3), Var(1)};
    seq = Var(vec);
    CHECK_NOTHROW(typeLookup.getConversion(NumberType::Float32, NumberType::Float64)
                      ->convertMultipleSeq(float4v, float2v, 4, seq.payload.seqValue));
    CHECK(float2v[0] == (double)4.4f);
    CHECK(float2v[1] == (double)2.2f);
  }

  SECTION("Vector take (out of range)") {
    float float2v0[2] = {1.f, 2.f};
    float float2v1[2] = {0};

    std::vector<Var> vec{Var(3)};
    Var seq = Var(vec);

    // Should return false
    CHECK_THROWS_AS(typeLookup.getConversion(NumberType::Float32, NumberType::Float32)
                        ->convertMultipleSeq(float2v0, float2v1, 2, seq.payload.seqValue),
                    NumberConversionOutOfRangeEx);
    CHECK(float2v1[0] == 0.f);
    CHECK(float2v1[1] == 0.f);
  }
}

TEST_CASE("Vector types") {
  VectorTypeLookup Typelookup;

  SHType typesToCheck[] = {
      SHType::Int,   SHType::Int2,   SHType::Int3,   SHType::Int4,   SHType::Int8,  SHType::Int16,
      SHType::Float, SHType::Float2, SHType::Float3, SHType::Float4, SHType::Color,
  };

  for (const SHType &typeToCheck : typesToCheck) {
    const VectorTypeTraits *typeTraits = Typelookup.get(typeToCheck);
    CHECK(typeTraits);
    CHECK(typeTraits->dimension > 0);
  }

  for (size_t i = 1; i < 4; i++) {
    const VectorTypeTraits *compatibleType = Typelookup.findCompatibleType(false, i);
    CHECK(compatibleType);
    CHECK(compatibleType->dimension >= i);
    CHECK(!compatibleType->isInteger);
  }

  for (size_t i = 1; i < 16; i++) {
    const VectorTypeTraits *compatibleType = Typelookup.findCompatibleType(true, i);
    CHECK(compatibleType);
    CHECK(compatibleType->dimension >= i);
    CHECK(compatibleType->isInteger);
  }
}

TEST_CASE("UnsafeActivate-shard") {
  std::function<SHVar(SHContext *, const SHVar &)> f = [](SHContext *ctx, const SHVar &input) -> SHVar { return Var(77); };
  auto fVar = Var(reinterpret_cast<int64_t>(&f));
  auto b1 = createShard("UnsafeActivate!");
  DEFER(b1->destroy(b1));
  b1->setParam(b1, 0, &fVar);
  SHVar input{};
  CHECK(b1->activate(b1, nullptr, &input)->payload.intValue == 77);
}

TEST_CASE("AWAIT/AWAITNE") {
#if HAS_ASYNC_SUPPORT && SH_ENABLE_TIDE_POOL
  struct TestWork : TidePool::Work {
    TestWork() { testString = "test-aaaaaabbbbbbbcccccccccddddddeeeeeffff"; }
    virtual ~TestWork() {}
    virtual void call() {
      testString.append("test-aaaaaabbbbbbbcccccccccddddddeeeeeffff");
      std::cout << testString << "\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(4000));
    }

    std::string testString;
  };

  std::vector<TestWork> works(40);
  for (auto &work : works) {
    getTidePool().schedule(&work);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  // number of workers should be increased
  CHECK(getTidePool()._workers.size() > TidePool::NumWorkers);

  std::this_thread::sleep_for(std::chrono::milliseconds(15000));
  // number should be back now to normal
  CHECK(getTidePool()._workers.size() == TidePool::NumWorkers);
#endif
}

TEST_CASE("TTableVar initialization", "[TTableVar]") {
  SECTION("Default construction") {
    TableVar tv;
    REQUIRE(tv.valueType == SHType::Table);
    REQUIRE(tv.payload.tableValue.opaque != nullptr);
  }

  SECTION("Copy construction") {
    TableVar tv1;
    tv1.insert("key1", Var("value1"));
    TableVar tv2(tv1);
    REQUIRE(tv2["key1"] == Var("value1"));
  }

  SECTION("Move construction") {
    TableVar tv1;
    tv1.insert("key1", Var("value1"));
    TableVar tv2(std::move(tv1));
    REQUIRE(tv2["key1"] == Var("value1"));
  }

  SECTION("Initializer list construction") {
    TableVar tv1{{Var("key1"), Var("value1")}, {Var("key2"), Var("value2")}};
    REQUIRE(tv1["key1"] == Var("value1"));
    REQUIRE(tv1["key2"] == Var("value2"));
  }
}

TEST_CASE("TTableVar operations", "[TTableVar]") {
  SECTION("Assignment operator") {
    TableVar tv1;
    tv1.insert("key1", Var("value1"));
    TableVar tv2;
    tv2 = tv1;
    REQUIRE(tv2["key1"] == Var("value1"));
  }

  SECTION("Move assignment operator") {
    TableVar tv1;
    tv1.insert("key1", Var("value1"));
    TableVar tv2;
    tv2 = std::move(tv1);
    REQUIRE(tv2["key1"] == Var("value1"));
  }

  SECTION("Access operator") {
    TableVar tv;
    tv.insert("key1", Var("value1"));
    REQUIRE(tv["key1"] == Var("value1"));
  }

  SECTION("Check key existence") {
    TableVar tv;
    tv.insert("key1", Var("value1"));
    REQUIRE(tv.hasKey("key1"));
    REQUIRE_FALSE(tv.hasKey("key2"));
  }

  SECTION("Remove key") {
    TableVar tv;
    tv.insert("key1", Var("value1"));
    REQUIRE(tv.size() == 1);
    tv.remove("key1");
    REQUIRE(tv.size() == 0);
    REQUIRE_FALSE(tv.hasKey("key1"));
  }
}

#include <shards/core/function.hpp>
static int staticVal = 0;
extern shards::Function<void()> getFunc(int &output, int someValue) {
  return [&, someValue]() { output += someValue; };
}

extern void staticFunc() { staticVal = 1; }

TEST_CASE("Function") {
  shards::Function<void()> f = []() { staticVal = 2; };
  f();
  CHECK(staticVal == 2);

  decltype(f) f1 = getFunc(staticVal, 0x100);
  f1();
  CHECK(staticVal == (0x100 + 2));

  decltype(f) f2 = f1;
  f2();
  CHECK(staticVal == (0x100 * 2 + 2));

  auto f3 = getFunc(staticVal, 0x200);
  f3();
  CHECK(staticVal == (0x100 * 2 + 0x200 + 2));

  shards::Function<void()> f4;
  CHECK(!f4);
  f4 = []() {};
  CHECK(f4);

  f4();
  CHECK(staticVal == (0x100 * 2 + 0x200 + 2)); // Unchanged

  f4 = &staticFunc;
  CHECK(f4);
  f4();
  CHECK(staticVal == 1);

  f4.reset();
  CHECK(!f4);
}

#define TEST_SUCCESS_CASE(testName, code)                                   \
  SECTION(testName) {                                                       \
    auto seq = readHelper(code);                                            \
    shards::OwnedVar ast{seq.ast};                                          \
    REQUIRE(ast.valueType == SHType::Object);                               \
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")}); \
    REQUIRE(wire.wire);                                                     \
    auto mesh = SHMesh::make();                                             \
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));                    \
    mesh->tick();                                                           \
    shards_free_wire(wire);                                                 \
  }

#define TEST_EVAL_ERROR_CASE(testName, code, expectedErrorMessage)          \
  SECTION(testName) {                                                       \
    auto seq = readHelper(code);                                            \
    shards::OwnedVar ast{seq.ast};                                          \
    REQUIRE(ast.valueType == SHType::Object);                               \
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")}); \
    REQUIRE(wire.error);                                                    \
    std::string a(wire.error->message);                                     \
    std::string b(expectedErrorMessage);                                    \
    REQUIRE(a == b);                                                        \
    shards_free_error(wire.error);                                          \
  }

inline SHLAst readHelper(const char *code) {
  return shards_read(SHStringWithLen{}, SHStringWithLen{code, strlen(code)}, SHStringWithLen{}, nullptr, 0);
}

TEST_CASE("shards-lang") {
  // initialize shards
  shards_init(shardsInterface(SHARDS_CURRENT_ABI));

  TEST_SUCCESS_CASE("Simple", "1 | Math.Add(2) | Assert.Is(3) | Log");
  TEST_SUCCESS_CASE("Assign 1", "100 = x\n x | Log | Assert.Is(100)");
  TEST_SUCCESS_CASE("SubFlow Shards 1", "1 | Math.Add(2) | SubFlow({Assert.Is(3) | Log}) | Log");
  TEST_SUCCESS_CASE("SubFlow Shards 2", "1 | Math.Add(2) | SubFlow({Assert.Is(Value: 3) | Log}) | Log");
  TEST_EVAL_ERROR_CASE("SubFlow Shards 3", "1 | Math.Add(2) | SubFlow({Assert.Is(LOL: 3) | Log}) | Log",
                       "Unknown parameter 'LOL'");
  TEST_SUCCESS_CASE("Exp 1", "1 | Log | (2 | Log (3 | Log))");
  TEST_SUCCESS_CASE("Exp 2", "[(2 | Math.Multiply(3)) (2 | Math.Multiply(6)) (2 | Math.Multiply(12))] | Log")
  TEST_SUCCESS_CASE("Exp 3", "[(2 | Math.Multiply((3 | Math.Add(6)))) (2 | Math.Multiply(6)) (2 | Math.Multiply(12))] | Log")
  TEST_EVAL_ERROR_CASE("Failed EvalExpr", "#(false | Assert.Is(true))", "Assert failed - Is");
  TEST_SUCCESS_CASE("EvalExpr 1", "2 | Math.Multiply(#(1 | Math.Add(2) | Log)) | Log | Assert.Is(6)");
  TEST_SUCCESS_CASE("SeqTake 1", "[1 2] | Log = s s:0 | Log | Assert.Is(1) s:1 | Log | Assert.Is(2)");
  TEST_SUCCESS_CASE("SeqTake 2", "[1 2] | Log = s 1 | Math.Add(s:0) | Assert.Is(2)");
  TEST_SUCCESS_CASE("TableTake 1", "{a: 1 b: 2} | Log = t t:a | Log | Assert.Is(1) t:b | Log | Assert.Is(2)");

  SECTION("TableTake 2") {
    auto code = "{a: 1 b: 2} | Log = t 1 | Math.Add(t:a) | Assert.Is(2)";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("SubFlow 1") {
    auto code = "{a: 1 b: 2} | {ToString | Assert.Is(\"{a: 1, b: 2}\") | Log} | Log = t t:a | Log | Assert.Is(1) t:b "
                "| Log | Assert.Is(2)";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("Enums 1") {
    auto code = "Msg(Enum::NotExisting)";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.error);
    DEFER(shards_free_error(wire.error));
    std::string a(wire.error->message);
    std::string b("Enum Enum not found");
    REQUIRE(a == b);
  }

  SECTION("Enums 2") {
    auto code = "Const(Type::String) | Log";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("Enums 3") {
    auto code = "Type::String | Log";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("Namespaces 1") {
    auto code1 = "10 = n";
    auto seq1 = readHelper(code1);
    shards::OwnedVar ast1{seq1.ast};
    REQUIRE(ast1.valueType == SHType::Object);

    auto code2 = "x/n | Math.Add(2) | Log";
    auto seq2 = readHelper(code2);
    shards::OwnedVar ast2{seq2.ast};
    REQUIRE(ast2.valueType == SHType::Object);

    auto env = shards_create_env(SHStringWithLen{"x", strlen("x")});

    auto err = shards_eval_env(env, &ast1);
    REQUIRE_FALSE(err);

    auto sub_env = shards_create_env(SHStringWithLen{});

    err = shards_eval_env(sub_env, &ast2);
    REQUIRE_FALSE(err);

    SHLEvalEnv *envs[] = {env, sub_env};

    auto wire = shards_transform_envs(&envs[0], 2, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("Namespaces 2") {
    auto code1 = "10 = n";
    auto seq1 = readHelper(code1);
    shards::OwnedVar ast1{seq1.ast};
    REQUIRE(ast1.valueType == SHType::Object);

    auto code2 = "x/n | Math.Add(2) | Log = r2"; // access still needs to be explicit
    auto seq2 = readHelper(code2);
    shards::OwnedVar ast2{seq2.ast};
    REQUIRE(ast2.valueType == SHType::Object);

    auto code3 = "r2 | Math.Add(2) | Log";
    auto seq3 = readHelper(code3);
    shards::OwnedVar ast3{seq3.ast};
    REQUIRE(ast3.valueType == SHType::Object);

    auto code4 = "x/n | Math.Add(x/y/r2) | Log";
    auto seq4 = readHelper(code4);
    shards::OwnedVar ast4{seq4.ast};
    REQUIRE(ast4.valueType == SHType::Object);

    auto env = shards_create_env(SHStringWithLen{"x", strlen("x")});
    auto err = shards_eval_env(env, &ast1);
    REQUIRE_FALSE(err);

    auto sub_env1 = shards_create_sub_env(env, SHStringWithLen{"y", strlen("y")});
    err = shards_eval_env(sub_env1, &ast2);
    REQUIRE_FALSE(err);

    auto sub_env2 = shards_create_sub_env(sub_env1, SHStringWithLen{});
    err = shards_eval_env(sub_env2, &ast3);
    REQUIRE_FALSE(err);

    auto another_env = shards_create_env(SHStringWithLen{});
    err = shards_eval_env(another_env, &ast4);
    REQUIRE_FALSE(err);

    SHLEvalEnv *envs[] = {env, sub_env1, sub_env2, another_env};

    auto wire = shards_transform_envs(&envs[0], 4, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("@wire 1") {
    auto code = R"(
      @wire(wire1 {
        Log | Math.Add(1) | Log
      })
      2 | Do(wire1) | Assert.Is(3)
    )";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    auto pWire = SHWire::sharedFromRef(*(wire.wire));
    mesh->schedule(pWire);
    // cover getherWires
    shards::printWireGraph(pWire.get());
    mesh->tick();
  }

  SECTION("String 1") {
    auto code = R"(
      "Hello" | Log >= s
      " " | AppendTo(s)
      "World" | AppendTo(s)
      s | Log
    )";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("Test @template 1") {
    auto code = R"(
      @template(group1 [n] {
        {n = n1} ; this will be made unique internally!
        Log | Math.Add(n1) | Log
      })
      2 | @group1(1) | Assert.Is(3)
      3 | @group1(2) | Assert.Is(5)
    )";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("Test @template 2") {
    auto code = R"(
      @template(range [from to] {
        [] >= s
        from >= n
        Repeat({
          n >> s
          n | Math.Add(1) > n
        } Times: to)
        s
      })
      #(@range(0 5)) | Log
    )";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});

    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("Test @mesh & @schedule & @run") {
    auto code = R"(
      @wire(wire1 {
        Msg("Hello")
      } Looped: true)
      @mesh(main)
      @schedule(main wire1)
      @run(main 1.0 5)
    )";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);
    auto wire = shards_eval(&ast, SHStringWithLen{"root", strlen("root")});

    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  TEST_SUCCESS_CASE("Color Log Test", "@color(0x112233) | Log \n @color(0x1122) | Log \n @color(0x11223344) | Log \n @color(0.1 "
                                      "0.3 0.7 1.0) | Log \n @color(10 30 60 255) | Log \n @color(1.0) | Log")

  TEST_SUCCESS_CASE("Log Various Data Test", "@i2(1) | Log \n @i2(1 2) | Log \n @i3(1) | Log \n @i3(1 2 3) | Log \n @i4(1) | Log "
                                             "\n @i4(1 2 3 4) | Log \n @f2(1) | Log \n @f2(1 2) | Log")

  TEST_SUCCESS_CASE("Shards Test", "@template(base [texture] { \"TEST\" = texture }) \n @base(test-shards-1) \n test-shards-1 | "
                                   "Log | Assert.Is(\"TEST\") \n @base(test-shards-2) | Log | Assert.Is(\"TEST\")")

  SECTION("Namespaces 3") {
    auto code = R"(
      @wire(w {
        Msg("Running")
      })

      Do(w)
    )";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);

    auto env = shards_create_env("x"_swl);
    auto err = shards_eval_env(env, &ast);
    REQUIRE_FALSE(err);

    auto wire = shards_transform_env(env, "x"_swl);
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }

  SECTION("Namespaces 4") {
    auto code = R"(
      @template(wt [name param] {
        @wire(name {
          Msg("Running")
          "xyz" | String.Contains(param) | Assert.Is(true)
        })
      })

      @wt(w1 "x")
      @wt(w2 "y")

      Do(w1)
      Do(w2)
    )";
    auto seq = readHelper(code);
    shards::OwnedVar ast{seq.ast};
    REQUIRE(ast.valueType == SHType::Object);

    auto env = shards_create_env("x"_swl);
    auto err = shards_eval_env(env, &ast);

    REQUIRE_FALSE(err);

    auto wire = shards_transform_env(env, "x"_swl);
    REQUIRE(wire.wire);
    DEFER(shards_free_wire(wire));
    auto mesh = SHMesh::make();
    mesh->schedule(SHWire::sharedFromRef(*(wire.wire)));
    mesh->tick();
  }
}

TEST_CASE("meshThreadTask") {
  shards::pushThreadName("Main Thread");
  auto mesh = SHMesh::make();

  auto currentThreadId = std::this_thread::get_id();

  int called1 = 0;
  int called1Activate = 0;
  int called1bis = 0;
  int called1bisActivate = 0;
  int called2 = 0;
  int called2Activate = 0;

  auto testWire = SHWire::make("test-wire");
  auto testWire2 = SHWire::make("test-wire2");
  auto steppedWire = SHWire::make("stepped-wire");

  std::function<SHVar(SHContext *, const SHVar &)> f1 = [&](SHContext *context, const SHVar &input) {
    shards::callOnMeshThread(context, [&]() {
      // required called on mesh thread
      REQUIRE(currentThreadId == std::this_thread::get_id());
      called1++;
    });
    called1Activate++;
    return input;
  };

  std::function<SHVar(SHContext *, const SHVar &)> f1bis = [&](SHContext *context, const SHVar &input) {
    shards::callOnMeshThread(context, [&]() {
      // required called on mesh thread
      REQUIRE(currentThreadId == std::this_thread::get_id());
      called1bis++;
    });
    called1bisActivate++;
    return input;
  };

  std::function<SHVar(SHContext *, const SHVar &)> f2 = [&](SHContext *context, const SHVar &input) {
    shards::callOnMeshThread(context, [&]() {
      // required called on mesh thread
      REQUIRE(currentThreadId == std::this_thread::get_id());
      called2++;
    });
    called2Activate++;
    return input;
  };

  { // run 2 wires with 2 onMeshThread calls
    auto unsafeActivate1 = createShard("UnsafeActivate!");
    auto vf1 = Var(reinterpret_cast<int64_t>(&f1));
    unsafeActivate1->setParam(unsafeActivate1, 0, &vf1);
    testWire->addShard(unsafeActivate1);

    auto unsafeActivate1bis = createShard("UnsafeActivate!");
    auto vf1bis = Var(reinterpret_cast<int64_t>(&f1bis));
    unsafeActivate1bis->setParam(unsafeActivate1bis, 0, &vf1bis);
    testWire2->addShard(unsafeActivate1bis);
  }

  // adds a onMeshThread call inside the following Step
  auto unsafeActivate2 = createShard("UnsafeActivate!");
  auto vf2 = Var(reinterpret_cast<int64_t>(&f2));
  unsafeActivate2->setParam(unsafeActivate2, 0, &vf2);
  steppedWire->addShard(unsafeActivate2);

  // Step and adds it to the second wire
  auto stepShard = createShard("Step");
  stepShard->setup(stepShard);
  auto steppedWireVar = Var(steppedWire);
  stepShard->setParam(stepShard, 0, &steppedWireVar);
  testWire2->addShard(stepShard);

  auto pauseShard = createShard("Pause");
  testWire2->addShard(pauseShard);

  mesh->schedule(testWire);
  mesh->schedule(testWire2);

  for (int i = 0; i < 10; i++) {
    mesh->tick();
    REQUIRE(called1Activate == 1);
    REQUIRE(called1bisActivate == 1);
    REQUIRE(called2Activate == 1);
    REQUIRE(called1 == 1);
    REQUIRE(called1bis == 1);
    REQUIRE(called2 == 1);
  }
}

TEST_CASE("meshThreadTask-looped") {
  shards::pushThreadName("Main Thread");
  auto mesh = SHMesh::make();

  auto currentThreadId = std::this_thread::get_id();

  int called1 = 0;
  int called1Activate = 0;
  int called1bis = 0;
  int called1bisActivate = 0;
  int called2 = 0;
  int called2Activate = 0;

  auto testWire = SHWire::make("test-wire");
  testWire->looped = true;
  auto testWire2 = SHWire::make("test-wire2");
  testWire2->looped = true;
  auto steppedWire = SHWire::make("stepped-wire");
  steppedWire->looped = true;

  std::function<SHVar(SHContext *, const SHVar &)> f1 = [&](SHContext *context, const SHVar &input) {
    shards::callOnMeshThread(context, [&]() {
      // required called on mesh thread
      REQUIRE(currentThreadId == std::this_thread::get_id());
      called1++;
    });
    called1Activate++;
    return input;
  };

  std::function<SHVar(SHContext *, const SHVar &)> f1bis = [&](SHContext *context, const SHVar &input) {
    shards::callOnMeshThread(context, [&]() {
      // required called on mesh thread
      REQUIRE(currentThreadId == std::this_thread::get_id());
      called1bis++;
    });
    called1bisActivate++;
    return input;
  };

  std::function<SHVar(SHContext *, const SHVar &)> f2 = [&](SHContext *context, const SHVar &input) {
    shards::callOnMeshThread(context, [&]() {
      // required called on mesh thread
      REQUIRE(currentThreadId == std::this_thread::get_id());
      called2++;
    });
    called2Activate++;
    return input;
  };

  auto unsafeActivate1 = createShard("UnsafeActivate!");
  auto vf1 = Var(reinterpret_cast<int64_t>(&f1));
  unsafeActivate1->setParam(unsafeActivate1, 0, &vf1);
  testWire->addShard(unsafeActivate1);

  auto unsafeActivate1bis = createShard("UnsafeActivate!");
  auto vf1bis = Var(reinterpret_cast<int64_t>(&f1bis));
  unsafeActivate1bis->setParam(unsafeActivate1bis, 0, &vf1bis);
  testWire2->addShard(unsafeActivate1bis);

  auto unsafeActivate2 = createShard("UnsafeActivate!");
  auto vf2 = Var(reinterpret_cast<int64_t>(&f2));
  unsafeActivate2->setParam(unsafeActivate2, 0, &vf2);
  steppedWire->addShard(unsafeActivate2);

  auto stepShard = createShard("Step");
  stepShard->setup(stepShard);
  auto steppedWireVar = Var(steppedWire);
  stepShard->setParam(stepShard, 0, &steppedWireVar);
  testWire2->addShard(stepShard);

  mesh->schedule(testWire);
  mesh->schedule(testWire2);

  for (int i = 1; i < 10; i++) {
    mesh->tick();
    REQUIRE(called1Activate == i);
    REQUIRE(called1bisActivate == i);
    REQUIRE(called2Activate == i);
    REQUIRE(called1 == i);
    REQUIRE(called1bis == i);
    REQUIRE(called2 == i);
  }

  mesh->terminate();
  mesh.reset();
}

#include <shards/core/pool.hpp>

struct TestPoolItem {
  size_t refCount{};
  std::string tag;

  void inc() { refCount++; }
  void dec() {
    if (--refCount == 0) {
      delete this;
    }
  }
};

namespace shards {
template <> struct PoolItemTraits<TestPoolItem *> {
  TestPoolItem *newItem() { return new TestPoolItem{.refCount = 1}; }
  void release(TestPoolItem *&ptr) { ptr->dec(); }
  size_t canRecycle(TestPoolItem *&v) { return v->refCount == 1; }
  void recycled(TestPoolItem *&v) {}
};
} // namespace shards

TEST_CASE("Pool") {
  TestPoolItem *t3{};
  {
    Pool<TestPoolItem *> pool;
    pool.recycle();
    auto t0 = pool.newValue([&](TestPoolItem *item) { item->tag = "item1"; });
    t0->inc();
    pool.recycle();
    auto t1 = pool.newValue([&](TestPoolItem *item) { item->tag = "item2"; });
    t1->inc();
    pool.recycle();
    CHECK(t0->tag == "item1");
    CHECK(t1->tag == "item2");

    t0->dec();

    pool.recycle();
    auto t2 = pool.newValue([&](TestPoolItem *item) {});
    t2->inc();
    CHECK(t2->tag == "item1"); // Check that this has been reused

    pool.recycle();
    t3 = pool.newValue([&](TestPoolItem *item) {});
    t3->inc();
    CHECK(t3->tag.empty()); // Check that this is a new item

    t1->dec();
    t2->dec();
    pool.recycle();

    CHECK(pool.freeList.size() == 2);
    CHECK(pool.inUseList.size() == 1); // Only t3 should be left
  }

  // Test this to be the last item, and that the pool has released it's reference
  CHECK(t3->refCount == 1);
  t3->dec();
}

#include <shards/core/hash.inl>

struct SimpleExtType {
  static inline shards::Type Type{shards::Type::Object('abcd', 'efgh')};
  size_t index{};

  static void hash(const SimpleExtType &self, void *outDigest, size_t digestSize) {
    static thread_local shards::HashState<XXH128_hash_t> hs;
    hs.reset();

    XXH3_state_t state;
    shards::hashReset<XXH128_hash_t>(&state);
    XXH3_128bits_update(&state, &self.index, sizeof(size_t));
    XXH128_hash_t digest = shards::hashDigest<XXH128_hash_t>(&state);
    memcpy(outDigest, &digest, std::min(sizeof(XXH128_hash_t), digestSize));
  }

  static bool match(const SimpleExtType &self, const SimpleExtType &other) { return self.index == other.index; }
};
using SimpleExtTypeInfo = shards::TExtendedObjectType<shards::InternalCore, SimpleExtType>;

TEST_CASE("ExtType") {
  static thread_local shards::HashState<XXH128_hash_t> hs;
  hs.reset();

  shards::TypeInfo t0, t1;
  {
    auto &e = SimpleExtTypeInfo::makeExtended(t0);
    e.index = 1;
  }
  {
    t1 = (SHTypeInfo &)t0; // Attempt extend from existing
    (void)SimpleExtTypeInfo::makeExtended(t1);
  }

  auto h0 = hs.deriveTypeHash((SHTypeInfo &)t0);
  auto h1 = hs.deriveTypeHash((SHTypeInfo &)t1);
  REQUIRE(h0.low64 == h1.low64);
  REQUIRE(h0.high64 == h1.high64);

  bool matched = matchTypes(t1, t0, true, true, true);
  CHECK(matched);

  shards::TypeInfo t2;
  auto &e2 = SimpleExtTypeInfo::makeExtended(t2);
  e2.index = 0;

  auto h2 = hs.deriveTypeHash((SHTypeInfo &)t2);
  REQUIRE((h0.low64 != h2.low64 || h0.high64 != h2.high64));

  bool matched2 = matchTypes(t1, t2, true, true, true);
  CHECK(!matched2);
}

struct TestWireData {
  std::shared_ptr<SHWire> wire;
};

TEST_CASE("WireDoppelgangerPool acquire and release", "[WireDoppelgangerPool]") {
  // Create a master wire
  auto masterWire = SHWire::make("MasterWire");
  shards::WireDoppelgangerPool<TestWireData> pool(SHWire::weakRef(masterWire));

  // Test acquire
  auto [item1, cached1] = pool.acquire();
  REQUIRE(item1 != nullptr);
  REQUIRE_FALSE(cached1);

  auto [item2, cached2] = pool.acquire();
  REQUIRE(item2 != nullptr);
  REQUIRE_FALSE(cached2);

  REQUIRE(item1 != item2);

  // Release item1
  pool.release(item1);

  // Acquire again, should get item1 back
  auto [item3, cached3] = pool.acquire();
  REQUIRE(item3 == item1);
  REQUIRE(cached3);

  // Release all items
  pool.release(item2);
  pool.release(item3);

  // Acquire multiple items
  constexpr size_t numItems = 5;
  std::vector<TestWireData *> items;
  for (size_t i = 0; i < numItems; ++i) {
    auto [item, cached] = pool.acquire();
    REQUIRE(item != nullptr);
    items.push_back(item);
  }

  // Check that all items are unique
  std::unordered_set<TestWireData *> uniqueItems(items.begin(), items.end());
  REQUIRE(uniqueItems.size() == numItems);

  // Release all items
  for (auto item : items) {
    pool.release(item);
  }

  // Acquire again, should get all cached items
  for (size_t i = 0; i < numItems; ++i) {
    auto [item, cached] = pool.acquire();
    REQUIRE(cached);
    REQUIRE(uniqueItems.find(item) != uniqueItems.end());
  }
}

TEST_CASE("Mesh-schedule-wire-running") {
  auto mesh = SHMesh::make();
  std::shared_ptr<SHWire> wire = Wire("main").looped(false).let(Var("Hello")).shard("Log");

  mesh->schedule(wire);
  REQUIRE(shards::isRunning(wire.get()));
  mesh->tick();
  REQUIRE(!shards::isRunning(wire.get()));
  mesh->tick();
  REQUIRE(!shards::isRunning(wire.get()));
  mesh->tick();
  REQUIRE(!shards::isRunning(wire.get()));
  mesh->tick();
  REQUIRE(!shards::isRunning(wire.get()));
}

#ifndef SHARDS_VALGRIND
TEST_CASE("Mesh restart wire") {
  auto mesh = SHMesh::make();
  std::shared_ptr<SHWire> wire = Wire("main")
                                     .looped(true) //
                                     .shard("Math.Inc", Var::ContextVar("v"))
                                     .let(Var::ContextVar("v"))
                                     .shard("Log");
  auto &extVars = wire->getExternalVariables();
  Var v(0);
  v.flags |= SHVAR_FLAGS_EXTERNAL;
  extVars[OwnedVar::Foreign("v")] = SHExternalVariable{.var = &v};

  mesh->schedule(wire);
  mesh->tick();
  REQUIRE(v.payload.intValue == 1);
  REQUIRE(mesh->scheduledSetCount() == 1);
  REQUIRE(mesh->scheduledCount() == 1);

  mesh->tick();
  REQUIRE(v.payload.intValue == 2);
  REQUIRE(mesh->scheduledSetCount() == 1);
  REQUIRE(mesh->scheduledCount() == 1);

  mesh->remove(wire);
  REQUIRE(mesh->scheduledSetCount() == 0);
  mesh->schedule(wire);
  REQUIRE(mesh->scheduledSetCount() == 1);

  mesh->tick();
  REQUIRE(v.payload.intValue == 3);
  REQUIRE(mesh->scheduledSetCount() == 1);
  REQUIRE(mesh->scheduledCount() == 1);

  mesh->tick();
  REQUIRE(v.payload.intValue == 4);
  REQUIRE(mesh->scheduledSetCount() == 1);
  REQUIRE(mesh->scheduledCount() == 1);

  mesh->remove(wire);

  mesh->tick();
  REQUIRE(v.payload.intValue == 4);
  REQUIRE(mesh->scheduledSetCount() == 0);
  REQUIRE(mesh->scheduledCount() == 0);

  mesh->tick();
  REQUIRE(v.payload.intValue == 4);
  REQUIRE(mesh->scheduledSetCount() == 0);
  REQUIRE(mesh->scheduledCount() == 0);

  mesh->schedule(wire);
  mesh->tick();
  REQUIRE(v.payload.intValue == 5);
  REQUIRE(mesh->scheduledSetCount() == 1);
  REQUIRE(mesh->scheduledCount() == 1);

  mesh->tick();
  REQUIRE(v.payload.intValue == 6);
  REQUIRE(mesh->scheduledSetCount() == 1);
  REQUIRE(mesh->scheduledCount() == 1);

  wire.reset();
  mesh.reset();
}
#endif
