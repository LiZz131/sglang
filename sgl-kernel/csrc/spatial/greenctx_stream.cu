// Documentation: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
#include <torch/all.h>

#include <cstdlib>

#include "cuda_utils.h"
#include "greenctx_stream.h"

static int CUDA_DRIVER_VERSION;

using PFN_cuGreenCtxStreamCreate = CUresult(CUDAAPI*)(CUstream*, CUgreenCtx, unsigned int, int);

auto probe_cuGreenCtxStreamCreate() -> PFN_cuGreenCtxStreamCreate {
  static PFN_cuGreenCtxStreamCreate pfn = nullptr;
  CUDA_DRV(cuGetProcAddress("cuGreenCtxStreamCreate", reinterpret_cast<void**>(&pfn), CUDA_DRIVER_VERSION, 0, nullptr));
  return pfn;
}

static std::vector<int64_t> create_streams_fallback_one_green(CUgreenCtx gctx, int64_t n_streams) {
  CUcontext ctx;
  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  std::vector<int64_t> out;
  out.reserve(static_cast<size_t>(n_streams));
  for (int64_t i = 0; i < n_streams; ++i) {
    CUstream stream{};
    CUDA_DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    out.push_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream)));
  }
  CUDA_DRV(cuCtxPopCurrent(nullptr));
  return out;
}

static std::vector<int64_t> create_streams_on_green_ctx(
    CUgreenCtx gctx,
    int64_t n_streams,
    PFN_cuGreenCtxStreamCreate pfn) {
  TORCH_CHECK(n_streams >= 1, "n_streams must be >= 1");
  if (!pfn) {
    return create_streams_fallback_one_green(gctx, n_streams);
  }
  std::vector<int64_t> out;
  out.reserve(static_cast<size_t>(n_streams));
  for (int64_t i = 0; i < n_streams; ++i) {
    CUstream stream{};
    CUDA_DRV(pfn(&stream, gctx, CU_STREAM_NON_BLOCKING, 0));
    out.push_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream)));
  }
  return out;
}

inline void destroy_green_context(CUgreenCtx gctx) {
  if (!gctx) return;
  CUDA_DRV(cuGreenCtxDestroy(gctx));
}

static void create_two_partition_green_contexts(
    int64_t smA,
    int64_t smB,
    int64_t device,
    CUgreenCtx gctx_out[2],
    CUgreenCtx* gctx_scratch,
    int* sm_count_a,
    int* sm_count_b) {
  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];

  TORCH_CHECK(smA > 0 && smB > 0, "SM counts must be positive");

  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));

  const unsigned minCount = static_cast<unsigned>(smA + smB);
  const unsigned minCountA = static_cast<unsigned>(smA);
  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  unsigned nbGroups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  nbGroups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));

  *sm_count_a = resources[0].sm.smCount;
  *sm_count_b = resources[1].sm.smCount;

  gctx_out[0] = gctx[0];
  gctx_out[1] = gctx[1];
  *gctx_scratch = gctx[2];
}

std::vector<int64_t> create_greenctx_streams_by_value_enhanced(
    int64_t smA,
    int64_t smB,
    int64_t n_streams_a,
    int64_t n_streams_b,
    int64_t device) {
  CUDA_DRV(cuDriverGetVersion(&CUDA_DRIVER_VERSION));

  TORCH_CHECK(n_streams_a >= 1 && n_streams_b >= 1, "n_streams_a and n_streams_b must be >= 1");

  CUgreenCtx gctx_pair[2];
  CUgreenCtx gctx_scratch{};
  int smCountA = 0;
  int smCountB = 0;
  create_two_partition_green_contexts(smA, smB, device, gctx_pair, &gctx_scratch, &smCountA, &smCountB);

  const auto pfn = probe_cuGreenCtxStreamCreate();
  if (!pfn) {
    TORCH_WARN("cuGreenCtxStreamCreate(cuda>=12.5) is not available, using fallback");
  }

  std::vector<int64_t> streams_a = create_streams_on_green_ctx(gctx_pair[0], n_streams_a, pfn);
  std::vector<int64_t> streams_b = create_streams_on_green_ctx(gctx_pair[1], n_streams_b, pfn);

  destroy_green_context(gctx_scratch);

  std::vector<int64_t> vec;
  vec.reserve(streams_a.size() + streams_b.size() + 2);
  vec.insert(vec.end(), streams_a.begin(), streams_a.end());
  vec.insert(vec.end(), streams_b.begin(), streams_b.end());
  vec.push_back(static_cast<int64_t>(smCountA));
  vec.push_back(static_cast<int64_t>(smCountB));
  return vec;
}

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  std::vector<int64_t> full = create_greenctx_streams_by_value_enhanced(smA, smB, 1, 1, device);
  const size_t n = full.size();
  TORCH_INTERNAL_ASSERT(n >= 4, "expected at least 4 return values from create_greenctx_streams_by_value_enhanced");
  return {full[0], full[1], full[n - 2], full[n - 1]};
}
