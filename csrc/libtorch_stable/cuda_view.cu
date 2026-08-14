#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

#include <array>
#include <optional>

// torch < 2.13 stable ABI lacks Tensor::layout() and the from_blob deleter
// overload (MI100 base image ships torch 2.10).
#define VLLM_STABLE_ABI_HAS_LAYOUT_AND_DELETER \
  (TORCH_VERSION_MAJOR > 2 ||                  \
   (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 13))

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
#if VLLM_STABLE_ABI_HAS_LAYOUT_AND_DELETER
  const auto layout = cpu_tensor.layout();
#else
  // Input is checked to be a CPU tensor above; CPU tensors handed to this
  // UVA-view op are always strided.
  const auto layout = torch::headeronly::Layout::Strided;
#endif
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  std::array<StableIValue, 2> is_pinned_stack{
      torch::stable::detail::from(cpu_tensor),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
  if (torch::stable::detail::to<bool>(is_pinned_stack[0])) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    STD_TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(err));

#if VLLM_STABLE_ABI_HAS_LAYOUT_AND_DELETER
    return torch::stable::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), cuda_dev, dtype,
        [base = cpu_tensor](void*) {});  // keep cpu tensor alive
#else
    // No deleter overload on this ABI: the sole vLLM caller
    // (vllm/utils/torch_utils.py uva wrapper) keeps the source CPU tensor
    // alive alongside the returned view.
    return torch::stable::from_blob(device_ptr, cpu_tensor.sizes(),
                                    cpu_tensor.strides(), cuda_dev, dtype);
#endif
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

#if VLLM_STABLE_ABI_HAS_LAYOUT_AND_DELETER
  auto deleter = [host_ptr](void*) { cudaFreeHost(host_ptr); };

  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type(), deleter);
#else
  // No deleter overload on this ABI: the pinned staging buffer is
  // intentionally left alive for the lifetime of the returned view. This
  // path only runs for non-pinned inputs at weight-load time, so the leak
  // is bounded and one-shot.
  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type());
#endif
}
