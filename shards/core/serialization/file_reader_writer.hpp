#ifndef C4F4130D_93E8_4646_9AF3_521516233383
#define C4F4130D_93E8_4646_9AF3_521516233383

#include <fstream>
#include <stdexcept>
#include "reader_writer.hpp"

namespace shards {

struct FileWriter {
  static constexpr IOMode Mode = IOMode::Write;

private:
  std::ofstream _file;

public:
  FileWriter(const std::string &filename) : _file(filename, std::ios::binary) {
    if (!_file.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }
  }

  void operator()(const uint8_t *buf, size_t size) {
    _file.write(reinterpret_cast<const char *>(buf), size);
    if (_file.fail()) {
      throw std::runtime_error("Failed to write to file");
    }
  }
};

struct FileReader {
  static constexpr IOMode Mode = IOMode::Read;

private:
  std::ifstream _file;

public:
  FileReader(const std::string &filename) : _file(filename, std::ios::binary) {
    if (!_file.is_open()) {
      throw std::runtime_error("Failed to open file for reading: " + filename);
    }
  }

  void operator()(uint8_t *buf, size_t size) {
    _file.read(reinterpret_cast<char *>(buf), size);
    if (_file.fail() && !_file.eof()) {
      throw std::runtime_error("Failed to read from file");
    }
    if (_file.gcount() != static_cast<std::streamsize>(size)) {
      throw std::runtime_error("Unexpected end of file");
    }
  }
};

} // namespace shards

#endif /* C4F4130D_93E8_4646_9AF3_521516233383 */
