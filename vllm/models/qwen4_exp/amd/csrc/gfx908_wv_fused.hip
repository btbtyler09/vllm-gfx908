// Fused-epilogue copy of vLLM's wvSplitK_hf_sml_ (csrc/rocm/skinny_gemms.cu) for gfx908.
// EPI 0: plain; EPI 1: silu(v/HC) on columns < R (hyper-connection mix_down lora); EPI 2: gate-mix with
// YTILE=HC permuted mix_up rows. Weight A: [Mcols, K] bf16 row-major; activations B: [N, K]; C: [N, Mcols].
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>
#include <algorithm>
#define __HIP__GFX9__
#define LDS_SIZE (64 * 1024)
template <typename T> __device__ __forceinline__ T __float2s(float v);
template <> __device__ __forceinline__ __hip_bfloat16 __float2s(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ half __float2s(float v) { return __float2half(v); }
template <typename T> __device__ __forceinline__ T loadnt(T* addr) { return __builtin_nontemporal_load(addr); }
static int mindiv(int N, int div1, int div2) {
  int nPrRnd = div1 * div2; int rnds[13];
  for (int i = 0; i < 13; i++) { rnds[i] = (N + nPrRnd - 1) / nPrRnd; nPrRnd -= div1; }
  for (int i = 12; i >= 0; i--) if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}
__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

#if defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)
  #define DOT2C(V0, V2, V3)                                          \
    if constexpr (std::is_same_v<scalar_t, half>) {                  \
      asm("v_dot2c_f32_f16 %0, %2, %3"                               \
          : "=v"(V0)                                                 \
          : "0"(V0), "v"(V2), "v"(V3));                              \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) { \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *  \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));   \
      V0 += (s.x + s.y);                                             \
    }
#elif defined(__HIP__GFX1X__)
  // gfx1x: v_dot2_f32_f16 (VOP3-P, dot10-insts, available on gfx11+gfx12)
  #define DOT2C(V0, V2, V3)                                               \
    if constexpr (std::is_same_v<scalar_t, half>) {                       \
      asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(V0) : "v"(V2), "v"(V3)); \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {      \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *       \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));        \
      V0 += (s.x + s.y);                                                  \
    }
#endif

// This version targets cases where A[] fits LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int EPI>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_sml_(const int K, const int Kbp, const int Kap, const int M,
                     const int Bx, const int By, const scalar_t* B,
                     const scalar_t* __restrict__ A,
                     const scalar_t* __restrict__ BIAS, scalar_t* C,
                     const int _WvPrGrp, const int CuCount,
                     const scalar_t* __restrict__ XN, scalar_t* __restrict__ Y,
                     const int HD, const int R, const int HCn) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  //----------------------------------------------------
  // Reserving 64/160 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not going to work!
  //----------------------------------------------------
  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  while (m < M) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          if constexpr (EPI == 2) {
            const int i = m / YTILE;
            float acc = 0.f;
            for (int y = 0; y < YTILE; y++) {
              float g = __bfloat162float(__float2bfloat16(sum[n][y]));
              acc += (1.f / (1.f + __expf(-g))) * __bfloat162float(XN[(long)n * (HCn * HD) + y * HD + i]);
            }
            Y[(long)n * HD + i] = __float2bfloat16(acc / HCn);
          } else {
            for (int y = 0; y < YTILE; y++) {
              if constexpr (std::is_same_v<scalar_t, half>) {
                sum[n][y] += __half2float(biases[n][y]);
              } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
                sum[n][y] += __bfloat162float(biases[n][y]);
              }
              float v = sum[n][y];
              if constexpr (EPI == 1) {
                if (m + y < R) { float t = __bfloat162float(__float2bfloat16(v)) / HCn; v = t / (1.f + __expf(-t)); }
              }
              C[m + y + n * M] = __float2s<scalar_t>(v);
            }
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          /*float accm1 = 0;
           for (int i=0; i<64; i++)
              accm1 += __shfl(sum4[n][y][i%4], i);
          sum4[n][y][0] = accm1;*/
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31

          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          if constexpr (EPI == 2) {
            // gate-mix: YTILE(=HC) permuted rows (i*HC + s) -> mean_s sigmoid(g_s) * xn[s*HD + i]
            const int i = m / YTILE;
            float acc = 0.f;
            for (int y = 0; y < YTILE; y++) {
              float g = __bfloat162float(__float2bfloat16(sum4[n][y][0] + __bfloat162float(biases[n][y])));
              acc += (1.f / (1.f + __expf(-g))) * __bfloat162float(XN[(long)n * (HCn * HD) + y * HD + i]);
            }
            Y[(long)n * HD + i] = __float2bfloat16(acc / HCn);
          } else {
            for (int y = 0; y < YTILE; y++) {
              sum4[n][y][0] += __bfloat162float(biases[n][y]);
              float v = sum4[n][y][0];
              if constexpr (EPI == 1) {
                if (m + y < R) { float t = __bfloat162float(__float2bfloat16(v)) / HCn; v = t / (1.f + __expf(-t)); }
              }
              C[m + y + n * M] = __float2bfloat16(v);
            }
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}

using bf = __hip_bfloat16;
template <int YTILE, int UNRL, int N, int EPI>
static void launch_cfg(int K, int Kap, int Kbp, int Mc, const bf* A, const bf* B, bf* C, const bf* XN, bf* Y, int HD, int R, int HCn, int cu, hipStream_t st) {
  dim3 grid(cu), block(64, 16);
  int wv = mindiv(Mc, cu * YTILE, 16);
  // stock order: (K, Kap_in=weight stride, Kbp_in=x stride, M, Bx, By, weight, x, bias, C, wv, cu)
  wvSplitK_hf_sml_<bf, 64, YTILE, 16, 8, UNRL, N, EPI><<<grid, block, 0, st>>>(K, Kap, Kbp, Mc, 1, 1, A, B, nullptr, C, wv, cu, XN, Y, HD, R, HCn);
}
#define DISPATCH_N(EPIv, YT, UN) \
  { if (N == 1) launch_cfg<YT, UN, 1, EPIv>(K, Kap, Kbp, Mc, A, B, C, XN, Y, HD, R, HCn, cu, st); \
    else if (N == 2) launch_cfg<YT, UN, 2, EPIv>(K, Kap, Kbp, Mc, A, B, C, XN, Y, HD, R, HCn, cu, st); \
    else launch_cfg<YT, UN, 3, EPIv>(K, Kap, Kbp, Mc, A, B, C, XN, Y, HD, R, HCn, cu, st); }

// weight [Mc, K], x [N, K] (N <= 3, N*K <= 32768), out [N, Mc] (EPI 0/1) or y [N, HD] (EPI 2, weight rows permuted i*HC+s)
void wv_fused(at::Tensor weight, at::Tensor x, at::Tensor out, at::Tensor xn, int64_t epi, int64_t hc, int64_t r, int64_t cu) {
  const int Mc = weight.size(0), K = weight.size(1), N = x.size(0);
  const int Kap = weight.stride(0), Kbp = x.stride(0);
  TORCH_CHECK(N >= 1 && N <= 3 && K % 8 == 0 && (long)Kbp * N <= 32768, "wv_fused: N<=3 and x must fit LDS");
  const bf* A = reinterpret_cast<const bf*>(weight.data_ptr()); const bf* B = reinterpret_cast<const bf*>(x.data_ptr());
  bf* C = reinterpret_cast<bf*>(out.data_ptr()); const bf* XN = reinterpret_cast<const bf*>(xn.data_ptr()); bf* Y = C;
  const int HD = (epi == 2) ? (int)out.size(1) : 0, R = (int)r, HCn = (int)hc;
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (epi == 2) { TORCH_CHECK(hc == 4 && Mc % 4 == 0); DISPATCH_N(2, 4, 1) }
  else {
    int sYT = (Mc + cu * 4 - 1) / (cu * 4);
    if (epi == 1) { if (sYT <= 1) DISPATCH_N(1, 1, 4) else DISPATCH_N(1, 2, 2) }
    else { if (sYT <= 1) DISPATCH_N(0, 1, 4) else DISPATCH_N(0, 2, 2) }
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("wv_fused", &wv_fused); }
