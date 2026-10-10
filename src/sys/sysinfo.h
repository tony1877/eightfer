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

// Shows the system's open dialog on this machine's desktop (Windows: Explorer's picker, on top of other windows) for a
// folder, or a file matching `pattern` (e.g. "*.gguf"; empty = any), starting in `start` when it exists. Returns the
// chosen path, or empty when cancelled. Blocks until the dialog closes. Elsewhere: empty.
std::string pick_path(bool folder, const std::string & title, const std::string & start, const std::string & pattern);

// Waits up to `ms` for process `pid` to exit (returns true when it has, or does not exist).
bool wait_pid(int pid, int ms);

// Starts args[0] with args[1..] detached from this process, sharing this process's stdout / stderr (so its output
// goes to the same log). Returns false (err says why) if it could not start. Windows only.
bool spawn_detached(const std::vector<std::string> & args, std::string & err);

// CPU time counters (100 ns units, Windows): all cores' total and busy time, and this process's busy time. Two samples
// give the share of the machine other programs used in between. Zero elsewhere.
struct CpuSample {
    uint64_t total = 0, busy = 0, self = 0;
};
CpuSample cpu_sample();

// This process's id.
int self_pid();

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
