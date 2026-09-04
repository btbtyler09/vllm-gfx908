// gfx908 zero-copy PLE row gather: rows are read by the GPU directly from the
// host page cache (hipHostRegister'd file-backed mmap of the checkpoint shards).
// Rows whose shard this rank does not own are written as zeros; the TP
// all-reduce (sum) that follows reassembles the full row exactly (x + 0 = x).
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

template <typename V>
__global__ void __launch_bounds__(64) ple_gather_k(
    const int64_t* __restrict__ ids, const int64_t* __restrict__ shard_ptr, int n_slots,
    int64_t shard_size, int row_bytes, uint8_t* __restrict__ out) {
  const int64_t id = ids[blockIdx.x];
  const int64_t shard = id / shard_size;
  const uint8_t* src = nullptr;
  if (id >= 0 && shard < n_slots) {
    const int64_t base = shard_ptr[shard];
    if (base != 0) src = reinterpret_cast<const uint8_t*>(base) + (id - shard * shard_size) * (int64_t)row_bytes;
  }
  uint8_t* dst = out + (int64_t)blockIdx.x * row_bytes;
  const int nv = row_bytes / (int)sizeof(V);
  V* d = reinterpret_cast<V*>(dst);
  const V* s = reinterpret_cast<const V*>(src);
  for (int i = threadIdx.x; i < nv; i += 64) d[i] = src ? s[i] : V{};
  for (int i = nv * (int)sizeof(V) + threadIdx.x; i < row_bytes; i += 64) dst[i] = src ? src[i] : (uint8_t)0;
}

// Registers [ptr, ptr+bytes) (page-aligned) with the GPU; returns the device pointer or 0.
int64_t host_register(int64_t ptr, int64_t bytes) {
  hipError_t e = hipHostRegister((void*)ptr, (size_t)bytes, hipHostRegisterMapped | hipHostRegisterPortable);
  if (e != hipSuccess) { TORCH_WARN("hipHostRegister failed: ", hipGetErrorString(e)); return 0; }
  void* dp = nullptr;
  e = hipHostGetDevicePointer(&dp, (void*)ptr, 0);
  if (e != hipSuccess) { TORCH_WARN("hipHostGetDevicePointer failed: ", hipGetErrorString(e)); return 0; }
  return (int64_t)dp;
}

void host_unregister(int64_t ptr) { (void)hipHostUnregister((void*)ptr); }

// ids: int64 [R]; shard_ptr: int64 [n_slots] (device); out: uint8 [R, row_bytes] contiguous.
void gather(at::Tensor ids, at::Tensor shard_ptr, int64_t shard_size, int64_t row_bytes, at::Tensor out, int64_t align) {
  TORCH_CHECK(ids.scalar_type() == at::kLong && shard_ptr.scalar_type() == at::kLong && out.scalar_type() == at::kByte);
  TORCH_CHECK(out.is_contiguous() && out.numel() == ids.numel() * row_bytes);
  const int R = (int)ids.numel();
  if (R == 0) return;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const int64_t* I = ids.data_ptr<int64_t>(); const int64_t* P = shard_ptr.data_ptr<int64_t>();
  uint8_t* O = out.data_ptr<uint8_t>(); const int n = (int)shard_ptr.numel();
  if (align >= 16) {
    ple_gather_k<uint4><<<R, 64, 0, st>>>(I, P, n, shard_size, (int)row_bytes, O);
  } else if (align >= 4) {
    ple_gather_k<uint32_t><<<R, 64, 0, st>>>(I, P, n, shard_size, (int)row_bytes, O);
  } else {
    ple_gather_k<uint8_t><<<R, 64, 0, st>>>(I, P, n, shard_size, (int)row_bytes, O);
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("host_register", &host_register); m.def("host_unregister", &host_unregister); m.def("gather", &gather);
}
