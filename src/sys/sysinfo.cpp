#include "sys/sysinfo.h"

#include <cstring>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid(unsigned leaf, unsigned r[4]) {
    int x[4];
    __cpuid(x, (int) leaf);
    for (int i = 0; i < 4; i++) {
        r[i] = (unsigned) x[i];
    }
}
#else
#include <cpuid.h>
static void cpuid(unsigned leaf, unsigned r[4]) {
    __cpuid(leaf, r[0], r[1], r[2], r[3]);
}
#endif
static std::string cpu_brand() {
    unsigned r[4];
    cpuid(0x80000000u, r);
    if (r[0] < 0x80000004u) {
        return {};
    }
    char s[49] = {};
    for (unsigned i = 0; i < 3; i++) {
        cpuid(0x80000002u + i, r);
        std::memcpy(s + 16 * i, r, 16);
    }
    std::string b(s);
    b.erase(0, b.find_first_not_of(' '));
    while (!b.empty() && b.back() == ' ') {
        b.pop_back();
    }
    return b;
}
#else
static std::string cpu_brand() {
    return {};
}
#endif

#if defined(_WIN32)

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#include <winioctl.h>

namespace e8::sys {

static std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return {};
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

static std::string narrow(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) {
        return {};
    }
    std::string s((size_t) n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

static std::string os_name() {
    using RtlGetVersionFn = LONG(WINAPI *)(PRTL_OSVERSIONINFOW);
    RTL_OSVERSIONINFOW vi{};
    vi.dwOSVersionInfoSize = sizeof(vi);
    HMODULE ntdll          = GetModuleHandleW(L"ntdll.dll");
    auto    fn = ntdll ? reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void *>(GetProcAddress(ntdll, "RtlGetVersion"))) : nullptr;
    if (!fn || fn(&vi) != 0) {
        return "Windows";
    }
    const char * name = vi.dwMajorVersion == 10 && vi.dwBuildNumber >= 22000 ? "Windows 11" : "Windows";
    return std::string(name) + " build " + std::to_string(vi.dwBuildNumber);
}

static int physical_cores() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return 0;
    }
    std::vector<char> buf(len);
    auto * p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, p, &len)) {
        return 0;
    }
    int n = 0;
    for (DWORD off = 0; off < len;) {
        auto * e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        n += e->Relationship == RelationProcessorCore;
        off += e->Size;
    }
    return n;
}

Info query() {
    Info i;
    i.os             = os_name();
    i.cpu            = cpu_brand();
    i.logical_cores  = (int) std::thread::hardware_concurrency();
    i.physical_cores = physical_cores();
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        i.ram_total = ms.ullTotalPhys;
        i.ram_avail = ms.ullAvailPhys;
    }
    return i;
}

int64_t free_space(const std::string & dir) {
    ULARGE_INTEGER avail{};
    if (!GetDiskFreeSpaceExW(widen(dir).c_str(), &avail, nullptr, nullptr)) {
        return -1;
    }
    return (int64_t) avail.QuadPart;
}

static const char * bus_name(STORAGE_BUS_TYPE t) {
    switch (t) {
        case BusTypeNvme:
            return "NVMe";
        case BusTypeSata:
            return "SATA";
        case BusTypeUsb:
            return "USB";
        case BusTypeSas:
            return "SAS";
        case BusTypeRAID:
            return "RAID";
        case BusTypeSpaces:
            return "Storage Spaces";
        default:
            return "other bus";
    }
}

std::string describe_drive(const std::string & path) {
    wchar_t root[MAX_PATH];
    if (!GetVolumePathNameW(widen(path).c_str(), root, MAX_PATH)) {
        return {};
    }
    std::wstring vol = root;  // "D:\"
    if (vol.size() < 2 || vol[1] != L':') {
        return {};
    }
    const std::wstring dev = L"\\\\.\\" + vol.substr(0, 2);
    HANDLE h = CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return {};
    }
    STORAGE_PROPERTY_QUERY q{};
    q.PropertyId = StorageDeviceProperty;
    q.QueryType  = PropertyStandardQuery;
    alignas(8) char buf[1024] = {};
    DWORD           got       = 0;
    std::string     out;
    if (DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), buf, sizeof(buf), &got, nullptr) &&
        got >= sizeof(STORAGE_DEVICE_DESCRIPTOR)) {
        const auto * d = reinterpret_cast<const STORAGE_DEVICE_DESCRIPTOR *>(buf);
        if (d->ProductIdOffset && d->ProductIdOffset < got) {
            out = buf + d->ProductIdOffset;
            while (!out.empty() && out.back() == ' ') {
                out.pop_back();
            }
        }
        out += std::string(out.empty() ? "" : " ") + "[" + bus_name(d->BusType) + "]";
    }
    CloseHandle(h);
    return out;
}

std::vector<DriveTemp> drive_temps() {
    std::vector<DriveTemp> out;
    for (int i = 0; i < 32; i++) {
        const std::wstring dev = L"\\\\.\\PhysicalDrive" + std::to_wstring(i);
        HANDLE h = CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            continue;
        }
        STORAGE_PROPERTY_QUERY q{};
        q.QueryType = PropertyStandardQuery;
        alignas(8) char buf[1024] = {};
        DWORD           got       = 0;
        DriveTemp       d;
        q.PropertyId = StorageDeviceTemperatureProperty;
        const bool has_temp = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), buf, sizeof(buf), &got, nullptr) &&
                              got >= sizeof(STORAGE_TEMPERATURE_DATA_DESCRIPTOR) &&
                              reinterpret_cast<const STORAGE_TEMPERATURE_DATA_DESCRIPTOR *>(buf)->InfoCount > 0;
        if (has_temp) {
            d.temp_c = reinterpret_cast<const STORAGE_TEMPERATURE_DATA_DESCRIPTOR *>(buf)->TemperatureInfo[0].Temperature;
            q.PropertyId = StorageDeviceProperty;
            std::memset(buf, 0, sizeof(buf));
            if (DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), buf, sizeof(buf), &got, nullptr) &&
                got >= sizeof(STORAGE_DEVICE_DESCRIPTOR)) {
                const auto * dd = reinterpret_cast<const STORAGE_DEVICE_DESCRIPTOR *>(buf);
                if (dd->ProductIdOffset && dd->ProductIdOffset < got) {
                    d.name = buf + dd->ProductIdOffset;
                    while (!d.name.empty() && d.name.back() == ' ') {
                        d.name.pop_back();
                    }
                }
                d.name += std::string(d.name.empty() ? "" : " ") + "[" + bus_name(dd->BusType) + "]";
            }
            out.push_back(d);
        }
        CloseHandle(h);
    }
    return out;
}

GpuSensors gpu_sensors() {
    // NVML from the driver (System32\nvml.dll); the handles are resolved once
    using init_t  = int (*)();
    using dev_t   = int (*)(unsigned, void **);
    using temp_t  = int (*)(void *, int, unsigned *);
    using power_t = int (*)(void *, unsigned *);
    struct Util { unsigned gpu, memory; };
    using util_t  = int (*)(void *, Util *);
    static void *  dev   = nullptr;
    static temp_t  temp  = nullptr;
    static power_t power = nullptr;
    static util_t  util  = nullptr;
    static bool    tried = false;
    if (!tried) {
        tried = true;
        if (HMODULE lib = LoadLibraryW(L"nvml.dll")) {
            auto init = reinterpret_cast<init_t>(GetProcAddress(lib, "nvmlInit_v2"));
            auto get  = reinterpret_cast<dev_t>(GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2"));
            temp      = reinterpret_cast<temp_t>(GetProcAddress(lib, "nvmlDeviceGetTemperature"));
            power     = reinterpret_cast<power_t>(GetProcAddress(lib, "nvmlDeviceGetPowerUsage"));
            util      = reinterpret_cast<util_t>(GetProcAddress(lib, "nvmlDeviceGetUtilizationRates"));
            if (!init || !get || init() != 0 || get(0, &dev) != 0) {
                dev = nullptr;
            }
        }
    }
    GpuSensors s;
    if (!dev) {
        return s;
    }
    s.ok = true;
    unsigned v = 0;
    if (temp && temp(dev, 0 /* NVML_TEMPERATURE_GPU */, &v) == 0) {
        s.temp_c = (int) v;
    }
    if (power && power(dev, &v) == 0) {
        s.power_w = v / 1000.0;
    }
    Util u{};
    if (util && util(dev, &u) == 0) {
        s.mem_util = (int) u.memory;
    }
    return s;
}

std::vector<std::string> fixed_drives() {
    std::vector<std::string> out;
    wchar_t                  buf[512];
    const DWORD              n = GetLogicalDriveStringsW(512, buf);
    for (DWORD i = 0; i < n && buf[i];) {
        const wchar_t * root = buf + i;
        if (GetDriveTypeW(root) == DRIVE_FIXED) {
            out.push_back(narrow(root));
        }
        i += (DWORD) wcslen(root) + 1;
    }
    return out;
}

void set_thread_high_perf() {
#if defined(THREAD_POWER_THROTTLING_CURRENT_VERSION)
    THREAD_POWER_THROTTLING_STATE t{};
    t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    t.StateMask   = 0;
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t));
#endif
}

std::vector<std::string> utf8_args(int argc, char ** argv) {
    int        n  = 0;
    LPWSTR *   wa = CommandLineToArgvW(GetCommandLineW(), &n);
    std::vector<std::string> out;
    if (!wa) {
        out.assign(argv, argv + argc);
        return out;
    }
    for (int i = 0; i < n; i++) {
        out.push_back(narrow(wa[i]));
    }
    LocalFree(wa);
    return out;
}

std::string join_path(const std::string & dir, const std::string & name) {
    if (!dir.empty() && (dir.back() == '\\' || dir.back() == '/')) {
        return dir + name;
    }
    return dir + "\\" + name;
}

bool is_directory(const std::string & path) {
    const DWORD a = GetFileAttributesW(widen(path).c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

bool remove_file(const std::string & path) {
    return DeleteFileW(widen(path).c_str()) != 0;
}

bool make_dir(const std::string & path) {
    return CreateDirectoryW(widen(path).c_str(), nullptr) != 0 || GetLastError() == ERROR_ALREADY_EXISTS;
}

bool remove_dir(const std::string & path) {
    return RemoveDirectoryW(widen(path).c_str()) != 0;
}

} // namespace e8::sys

#else  // POSIX

#include <cerrno>
#include <fstream>
#include <set>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>

namespace e8::sys {

static uint64_t meminfo_kb(const char * key) {
    std::ifstream f("/proc/meminfo");
    std::string   k;
    uint64_t      v = 0;
    std::string   unit;
    while (f >> k >> v >> unit) {
        if (k == key) {
            return v;
        }
    }
    return 0;
}

static int physical_cores() {
    std::ifstream                       f("/proc/cpuinfo");
    std::string                         line;
    std::string                         phys;
    std::set<std::pair<std::string, std::string>> cores;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, line.find_last_not_of(" \t", colon - 1) + 1);
        const std::string val = colon + 2 <= line.size() ? line.substr(colon + 2) : "";
        if (key == "physical id") {
            phys = val;
        } else if (key == "core id") {
            cores.insert({ phys, val });
        }
    }
    return (int) cores.size();
}

Info query() {
    Info           i;
    struct utsname u {};
    i.os             = uname(&u) == 0 ? std::string(u.sysname) + " " + u.release : "POSIX";
    i.cpu            = cpu_brand();
    i.logical_cores  = (int) std::thread::hardware_concurrency();
    i.physical_cores = physical_cores();
    if (i.physical_cores == 0) {
        i.physical_cores = i.logical_cores;
    }
    i.ram_total = meminfo_kb("MemTotal:") * 1024;
    i.ram_avail = meminfo_kb("MemAvailable:") * 1024;
    return i;
}

int64_t free_space(const std::string & dir) {
    struct statvfs s {};
    if (statvfs(dir.c_str(), &s) != 0) {
        return -1;
    }
    return (int64_t) s.f_bavail * (int64_t) s.f_frsize;
}

std::string describe_drive(const std::string &) {
    return {};
}

std::vector<std::string> fixed_drives() {
    return {};
}

std::vector<DriveTemp> drive_temps() {
    return {};
}

GpuSensors gpu_sensors() {
    return {};
}

void set_thread_high_perf() {}

std::vector<std::string> utf8_args(int argc, char ** argv) {
    return std::vector<std::string>(argv, argv + argc);
}

std::string join_path(const std::string & dir, const std::string & name) {
    if (!dir.empty() && dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

bool is_directory(const std::string & path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool remove_file(const std::string & path) {
    return unlink(path.c_str()) == 0;
}

bool make_dir(const std::string & path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

bool remove_dir(const std::string & path) {
    return rmdir(path.c_str()) == 0;
}

} // namespace e8::sys

#endif
