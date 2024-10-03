#include <catch2/catch_all.hpp>
#include <gfx/texture_serde.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>

TEST_CASE("Serialization", "[Serializer]") {
  gfx::TextureFormat format{
      .resolution = {1024, 1024},
      .dimension = gfx::TextureDimension::D2,
  };
  auto bytes = shards::toByteArray(format);
  auto format2 = shards::fromByteArray<gfx::TextureFormat>(bytes);
  CHECK(format2.dimension == format.dimension);
  CHECK(format2.resolution == format.resolution);
}

TEST_CASE("String", "[Serializer]") {
  std::string str = "Hello, world!";
  auto bytes = shards::toByteArray(str);
  auto str2 = shards::fromByteArray<std::string>(bytes);
  CHECK(str2 == str);
}

TEST_CASE("Random vector", "[Serializer]") {
  std::vector<int> vec = {1, 2, 3, 4, 5};
  auto bytes = shards::toByteArray(vec);
  auto vec2 = shards::fromByteArray<std::vector<int>>(bytes);
  CHECK(vec2 == vec);

  std::vector<std::string, boost::container::pmr::polymorphic_allocator<std::string>> vec3 = {"Hello", "world", "this",
                                                                                              "is",    "a",     "test"};
  auto bytes2 = shards::toByteArray(vec3);
  auto vec4 = shards::fromByteArray<std::vector<std::string>>(bytes2);
  CHECK(vec3.size() == vec4.size());
  CHECK(vec3[0] == vec4[0]);
}

TEST_CASE("Variant", "[Serializer]") {
  using T0 = std::variant<int, float, uint32_t>;
  {
    T0 q = 10;
    auto bytes = shards::toByteArray(q);
    auto q1 = shards::fromByteArray<decltype(q)>(bytes);
    CHECK(q1 == q);
  }
  {
    T0 q = 5.0f;
    auto bytes = shards::toByteArray(q);
    auto q1 = shards::fromByteArray<decltype(q)>(bytes);
    CHECK(q1 == q);
  }
}
