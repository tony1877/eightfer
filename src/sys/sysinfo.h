#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace e8::sys {

struct Info {
    std::string os;
    std::string cpu;
    int         logical_cores  = 0;
    int         physical_cores = 0;
    uint64_t    ram_total      = 0;
    uint64_t    ram_avail      = 0;
};

Info query();

// Free bytes on the volume holding `dir`, or -1 if unknown.
int64_t free_space(const std::string & dir);

// Human description of the drive behind `path`, e.g. "WD_BLACK SN850X 1000GB [NVMe]". Empty if unknown.
std::string describe_drive(const std::string & path);

// Windows: root paths of all fixed drives ("C:\", "D:\", ...). Elsewhere: empty.
std::vector<std::string> fixed_drives();

// Windows 11 parks cores and throttles "background" threads; opt the calling thread out. No-op elsewhere.
void set_thread_high_perf();

// Command-line arguments as UTF-8 (on Windows, re-read via GetCommandLineW).
std::vector<std::string> utf8_args(int argc, char ** argv);

std::string join_path(const std::string & dir, const std::string & name);
bool        is_directory(const std::string & path);
bool        remove_file(const std::string & path);
bool        make_dir(const std::string & path);
bool        remove_dir(const std::string & path);

} // namespace e8::sys
