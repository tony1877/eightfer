#pragma once

#include <atomic>
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

// Temperature of each physical drive that reports one (Windows; no admin needed). Elsewhere: empty.
struct DriveTemp {
    std::string name;  // product id and bus, e.g. "WD_BLACK SN850X 1000GB [NVMe]"
    int         temp_c = 0;
};
std::vector<DriveTemp> drive_temps();

// NVIDIA GPU 0 through NVML, loaded at run time (ok = false without an NVIDIA driver). -1 = not reported.
struct GpuSensors {
    bool   ok       = false;
    int    temp_c   = -1;
    double power_w  = -1;
    int    mem_util = -1;  // % of time the memory was busy over the driver's last sample period
};
GpuSensors gpu_sensors();

// Windows 11 parks cores and throttles "background" threads; opt the calling thread out. No-op elsewhere.
void set_thread_high_perf();

// Runs args[0] with args[1..] (no shell), stdout and stderr appended to log_path, and waits for it. Polls `cancel`
// every 200 ms and kills the process when it turns true. Returns the exit code, -1 if it could not start
// (err says why), -2 when cancelled. Windows only (elsewhere: -1).
int run_process(const std::vector<std::string> & args, const std::string & log_path, const std::atomic<bool> & cancel,
                std::string & err);

// Full path of the running executable (empty if unknown).
std::string self_exe();

// Command-line arguments as UTF-8 (on Windows, re-read via GetCommandLineW).
std::vector<std::string> utf8_args(int argc, char ** argv);

std::string join_path(const std::string & dir, const std::string & name);
bool        is_directory(const std::string & path);
bool        remove_file(const std::string & path);
bool        make_dir(const std::string & path);
bool        remove_dir(const std::string & path);

} // namespace e8::sys
