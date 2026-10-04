#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace e8::cli {

// Each command receives the full argument list (args[0] = program, args[1] = command name).
int info(const std::vector<std::string> & args);
int ppl(const std::vector<std::string> & args);
int selftest(const std::vector<std::string> & args);
int logits(const std::vector<std::string> & args);

// Reads whitespace/comma separated token ids.
std::vector<int32_t> read_token_ids(const std::string & path);

} // namespace e8::cli
