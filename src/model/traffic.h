#pragma once

// Weight bytes read by passes over the model, for the dashboard (/stats). Every pass reads the base weights from
// VRAM; a pass with the residual also reads the residual (the part kept in VRAM from VRAM, the rest from RAM across
// PCIe); an MTP step reads the MTP block. The loaded model sets the sizes. KV-cache and activation traffic are not
// counted, so these are lower bounds on the real memory traffic.

#include <atomic>
#include <cstdint>

namespace e8::model::traffic {

inline std::atomic<uint64_t> vram_read{ 0 }, ram_read{ 0 };
inline std::atomic<uint64_t> base_bytes{ 0 }, mtp_bytes{ 0 }, res_vram_bytes{ 0 }, res_ram_bytes{ 0 };

inline void pass(bool residual) {
    vram_read += base_bytes.load() + (residual ? res_vram_bytes.load() : 0);
    if (residual) ram_read += res_ram_bytes.load();
}

inline void mtp() { vram_read += mtp_bytes.load(); }

// tokens of the current prefill call read so far (reset at the start of each call), for the dashboard's progress
inline std::atomic<int> prefill_done{ 0 };

} // namespace e8::model::traffic
