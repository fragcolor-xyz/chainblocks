/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>

#include <stdlib.h>

#ifdef _WIN32
#include "winsock2.h"
#define SHAlloc SHAlloc1
#define SHFree SHFree1
#include <windows.h>
#include <ShlObj.h>
#undef SHAlloc
#undef SHFree
#endif

// workaround for a boost bug..
#ifndef __kernel_entry
#define __kernel_entry
#endif

#pragma clang diagnostic push
// Disable warning inside boost process posix implementation
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#include <boost/asio.hpp>
#include <boost/process.hpp>
#include <boost/stacktrace.hpp>
#include <sstream>
#include <string>
#pragma clang diagnostic pop

#include <shards/core/async.hpp>

namespace shards {
namespace Process {
struct Run {
  std::array<SHExposedTypeInfo, 1> _requiring;
  std::string _outBuf;
  std::string _errBuf;
  std::optional<boost::process::child *> _cmd;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  PARAM_PARAMVAR(_executable, "Executable", "The executable to run.",
                 {CoreInfo::PathType, CoreInfo::PathVarType, CoreInfo::StringType, CoreInfo::StringVarType});
  PARAM_PARAMVAR(_arguments, "Arguments", "The arguments to pass to the executable.",
                 {CoreInfo::NoneType, CoreInfo::StringSeqType, CoreInfo::StringVarSeqType});
  PARAM_VAR(_timeout, "Timeout", "The maximum time to wait for the executable to finish in seconds.",
            {CoreInfo::NoneType, CoreInfo::IntType});
  PARAM_IMPL(PARAM_IMPL_FOR(_executable), PARAM_IMPL_FOR(_arguments), PARAM_IMPL_FOR(_timeout));

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar run(std::string moduleName, std::vector<std::string> argsArray, const SHVar &input) {
    // use async asio to avoid deadlocks
    boost::asio::io_service ios;
    std::future<std::string> ostr;
    std::future<std::string> estr;
    boost::process::opstream ipipe;

    // try PATH first
    auto exePath = boost::filesystem::path(moduleName);
    if (!boost::filesystem::exists(exePath)) {
      // fallback to searching PATH
      exePath = boost::process::search_path(moduleName);
    }

    if (exePath.empty()) {
      throw ActivationError("Executable not found");
    }

    exePath = exePath.make_preferred();

    boost::process::child cmd(exePath, argsArray, boost::process::std_out > ostr, boost::process::std_err > estr,
                              boost::process::std_in < ipipe, ios);

    _cmd = &cmd;

    if (!ipipe) {
      throw ActivationError("Failed to open streams for child process");
    }

    ipipe << SHSTRVIEW(input) << std::endl;
    ipipe.pipe().close(); // send EOF

    SHLOG_TRACE("Process started");

    auto timeout = std::chrono::seconds(_timeout->isNone() ? 30 : (int)*_timeout);
    auto endTime = std::chrono::system_clock::now() + timeout;
    ios.run_for(timeout);

    SHLOG_TRACE("Process finished");

    if (cmd.running()) {
      SHLOG_TRACE("Process still running after service wait");
      if (std::chrono::system_clock::now() > endTime) {
        cmd.terminate();
        throw ActivationError("Process timed out");
      } else {
        // give a further 1 second to terminate
        if (!cmd.wait_for(std::chrono::seconds(1))) {
          cmd.terminate();
        }
      }
    }

    // we still need to wait termination
    _outBuf = ostr.get();
    _errBuf = estr.get();

    if (cmd.exit_code() != 0) {
      SHLOG_INFO(_outBuf);
      SHLOG_ERROR(_errBuf);
      std::string err("The process exited with a non-zero exit code: ");
      err += std::to_string(cmd.exit_code());
      throw ActivationError(err);
    } else {
      if (_errBuf.size() > 0) {
        // print anyway this stream too
        SHLOG_INFO("(stderr) {}", _errBuf);
      }
      SHLOG_TRACE("Process finished successfully");
      return Var(_outBuf);
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    return awaitne(
        context,
        [&]() {
          auto moduleName = (std::string)(Var &)_executable.get();

          // add any arguments we have
          std::vector<std::string> argsArray;
          auto argsVar = _arguments.get();
          if (argsVar.valueType == SHType::Seq) {
            for (auto &arg : argsVar) {
              if (arg.payload.stringLen > 0) {
                argsArray.emplace_back(arg.payload.stringValue, arg.payload.stringLen);
              } else {
                // if really empty likely it's an error (also windows will fail
                // converting to utf16) if not maybe the string just didn't have
                // len set
                if (strlen(arg.payload.stringValue) == 0) {
                  throw ActivationError("Empty argument passed, this most likely is a mistake.");
                } else {
                  argsArray.emplace_back(arg.payload.stringValue); // ParamVar so should be same
                }
              }
            }
          }
          return run(moduleName, argsArray, input);
        },
        [&] {
          SHLOG_DEBUG("Process terminated");
          if (_cmd) {
            (*_cmd)->terminate();
          }
        });
  }
};

struct StackTrace {
  std::string _output;
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }

  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  SHVar activate(SHContext *ctx, SHVar input) {
    std::stringstream ss;
    ss << boost::stacktrace::stacktrace();
    _output = ss.str();
    return Var(_output);
  }
};

struct Exe {
  std::string _buf;

  static SHTypesInfo inputTypes() { return shards::CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::StringType; }
  static SHOptionalString help() { return SHCCSTR("Gives the current executable path."); }

  PARAM_IMPL();

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) { return outputTypes().elements[0]; }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    _buf.clear();
#ifdef _WIN32
    _buf.resize(_MAX_PATH);
    _buf.resize(GetModuleFileNameA(nullptr, _buf.data(), _MAX_PATH));
#elif defined __ANDROID__
#if __ANDROID_API__ >= 21
    _buf = getprogname();
#endif
#elif defined __linux__ && defined _GNU_SOURCE
    _buf = program_invocation_short_name;
#elif defined __APPLE__ || defined BSD
    _buf = getprogname();
#endif
    return Var(_buf);
  }
};

struct Shell {
  // boost process sucks.. we use our own pipe implementation
  
  static constexpr size_t bufferSize = 4096;
  std::vector<char> _readBuffer;

#ifdef _WIN32
  HANDLE _childStdin_Rd = NULL;
  HANDLE _childStdin_Wr = NULL;
  HANDLE _childStdout_Rd = NULL;
  HANDLE _childStdout_Wr = NULL;
  PROCESS_INFORMATION _processInfo{};
#else
  int _stdin[2];  // Parent write, child read
  int _stdout[2]; // Parent read, child write
  pid_t _pid;
#endif

  std::string _outputBuffer;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  PARAM_VAR(_shell_path, "Shell", "The shell executable path", {CoreInfo::StringType});
  PARAM_IMPL(PARAM_IMPL_FOR(_shell_path));

  Shell() : _readBuffer(bufferSize) {
#ifdef _WIN32
    _shell_path = shards::Var("cmd.exe");
#else
    _shell_path = shards::Var("/bin/bash");
#endif
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

#ifdef _WIN32
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create pipes
    if (!CreatePipe(&_childStdout_Rd, &_childStdout_Wr, &saAttr, 0) ||
        !CreatePipe(&_childStdin_Rd, &_childStdin_Wr, &saAttr, 0)) {
      throw ActivationError("Failed to create pipes");
    }

    // Ensure the read handle to the pipe for STDOUT is not inherited
    SetHandleInformation(_childStdout_Rd, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(_childStdin_Wr, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFO startupInfo{};
    startupInfo.cb = sizeof(STARTUPINFO);
    startupInfo.hStdError = _childStdout_Wr;
    startupInfo.hStdOutput = _childStdout_Wr;
    startupInfo.hStdInput = _childStdin_Rd;
    startupInfo.dwFlags |= STARTF_USESTDHANDLES;

    std::string shellPath = SHSTRING_PREFER_SHSTRVIEW(_shell_path);

    if (!CreateProcess(NULL, const_cast<LPSTR>(shellPath.c_str()), NULL, NULL, TRUE, 0, NULL, NULL, &startupInfo,
                       &_processInfo)) {
      throw ActivationError("Failed to create process");
    }

#else
    if (pipe(_stdin) == -1 || pipe(_stdout) == -1) {
      throw ActivationError("Failed to create pipes");
    }

    _pid = fork();
    if (_pid == -1) {
      throw ActivationError("Failed to fork process");
    }

    if (_pid == 0) {     // Child process
      close(_stdin[1]);  // Close write end of stdin
      close(_stdout[0]); // Close read end of stdout

      dup2(_stdin[0], STDIN_FILENO);
      dup2(_stdout[1], STDOUT_FILENO);
      dup2(_stdout[1], STDERR_FILENO);

      close(_stdin[0]);
      close(_stdout[1]);

      std::string shellPath = SHSTRING_PREFER_SHSTRVIEW(_shell_path);
      execl(shellPath.c_str(), shellPath.c_str(), NULL);
      exit(1);           // In case exec fails
    } else {             // Parent process
      close(_stdin[0]);  // Close read end of stdin
      close(_stdout[1]); // Close write end of stdout

      // Set non-blocking
      fcntl(_stdout[0], F_SETFL, O_NONBLOCK);
    }
#endif
  }

  void cleanup(SHContext *context) {
#ifdef _WIN32
    if (_processInfo.hProcess) {
      TerminateProcess(_processInfo.hProcess, 0);
      CloseHandle(_processInfo.hProcess);
      CloseHandle(_processInfo.hThread);
    }
    if (_childStdin_Rd)
      CloseHandle(_childStdin_Rd);
    if (_childStdin_Wr)
      CloseHandle(_childStdin_Wr);
    if (_childStdout_Rd)
      CloseHandle(_childStdout_Rd);
    if (_childStdout_Wr)
      CloseHandle(_childStdout_Wr);
#else
    if (_pid > 0) {
      kill(_pid, SIGTERM);
      waitpid(_pid, NULL, 0);
    }
    close(_stdin[1]);
    close(_stdout[0]);
#endif
    PARAM_CLEANUP(context);
  }

  std::string readAvailable() {
    std::string output;
#ifdef _WIN32
    DWORD bytesAvailable = 0;
    DWORD bytesRead = 0;

    while (PeekNamedPipe(_childStdout_Rd, NULL, 0, NULL, &bytesAvailable, NULL) && bytesAvailable > 0) {
      if (ReadFile(_childStdout_Rd, _readBuffer.data(), _readBuffer.size(), &bytesRead, NULL) && bytesRead > 0) {
        output.append(_readBuffer.data(), bytesRead);
      }
    }
#else
    ssize_t bytesRead;
    while ((bytesRead = read(_stdout[0], _readBuffer.data(), _readBuffer.size())) > 0) {
      output.append(_readBuffer.data(), bytesRead);
    }
#endif
    return output;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    return awaitne(
        context,
        [&]() {
          std::string cmd = SHSTRING_PREFER_SHSTRVIEW(input);
          cmd += "\n";

#ifdef _WIN32
          DWORD bytesWritten;
          WriteFile(_childStdin_Wr, cmd.c_str(), cmd.length(), &bytesWritten, NULL);
#else
          write(_stdin[1], cmd.c_str(), cmd.length());
#endif

          // Small sleep to allow output to be ready
          std::this_thread::sleep_for(std::chrono::milliseconds(10));

          _outputBuffer = readAvailable();
          return Var(_outputBuffer);
        },
        [&] {
          SHLOG_DEBUG("Shell terminated");
#ifdef _WIN32
          if (_processInfo.hProcess) {
            TerminateProcess(_processInfo.hProcess, 0);
          }
#else
          if (_pid > 0) {
            kill(_pid, SIGTERM);
          }
#endif
        });
  }
};
} // namespace Process

SHARDS_REGISTER_FN(process) {
  REGISTER_SHARD("Process.Run", Process::Run);
  REGISTER_SHARD("Process.StackTrace", Process::StackTrace);
  REGISTER_SHARD("Process.Exe", Process::Exe);
  REGISTER_SHARD("Process.Shell", Process::Shell);
}
} // namespace shards
// namespace shards
