/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * PyTorch op wrappers for adaptive 4/6 FP4 quantization + global amax.
 *
 * Registers:
 *   torch.ops.trtllm.fp4_quantize_ex(...)   — extended fp4_quantize with kernelVersion + scaleRule
 *   torch.ops.trtllm.calculate_global_amax(...)  — single-kernel runtime amax + global_scale
 *
 * Derived from tllm_linear_lite/quantize/fp4_quantize_op.cu.
 */

// Minimal CHECK macros (avoid thUtils.h which pulls in NvInferRuntime.h)
#ifndef CHECK_TH_CUDA
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, #x " dtype mismatch")
#define CHECK_INPUT(x, st) CHECK_TH_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x, st)
#endif
#include "tensorrt_llm/kernels/fp4QuantizeAdaptive.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace torch_ext
{

// FP4 E2M1 packed as uint8 (2 values per byte)
constexpr auto FLOAT4_E2M1X2 = torch::ScalarType::Byte;
// Scale factor dtype: FP8 E4M3 stored as uint8
constexpr auto SF_DTYPE = torch::ScalarType::Byte;

// ---------------------------------------------------------------------------
// fp4_quantize_ex
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor> fp4_quantize_ex(at::Tensor const& self,
    std::optional<at::Tensor> const& globalScale, int64_t sfVecSize, bool sfUseUE8M0, bool isSfSwizzledLayout,
    int64_t kernelVersion, int64_t scaleRule)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    if (sfUseUE8M0)
    {
        TORCH_CHECK(sfVecSize == 32, "sfVecSize can only be 32 when sfUseUE8M0 is true");
    }
    else
    {
        TORCH_CHECK(globalScale.has_value(), "globalScale is required when sfUseUE8M0 is false");
        CHECK_INPUT(globalScale.value(), torch::kFloat32);
        TORCH_CHECK(sfVecSize == 16, "sfVecSize can only be 16 when sfUseUE8M0 is false");
    }

    float* globalScalePtr = nullptr;
    if (globalScale.has_value())
    {
        globalScalePtr = globalScale->data_ptr<float>();
    }

    auto const& inputShape = self.sizes();
    auto const& rank = inputShape.size();
    TORCH_CHECK(rank >= 2, "Input should be >=2D tensor.");

    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % sfVecSize == 0, "Last dimension must be divisible by sfVecSize");

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;
    at::Tensor valueE2M1 = at::empty(outputShape, self.options().dtype(FLOAT4_E2M1X2));

    int64_t SFSize = isSfSwizzledLayout
        ? tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize)
        : tensorrt_llm::computeLinearLayoutSFSize(m, k / sfVecSize);
    at::Tensor scaleFP8SF = at::empty({SFSize}, self.options().dtype(SF_DTYPE));

    thread_local int const mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    auto const layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
                                           : tensorrt_llm::QuantizationSFLayout::LINEAR;

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device()).stream();

#define LAUNCH_FP4_QUANTIZE_EX(T, SF_VEC_SIZE)                                                                        \
    tensorrt_llm::kernels::invokeFP4QuantizationEx<T, SF_VEC_SIZE>(1, m, k,                                           \
        reinterpret_cast<T*>(self.data_ptr()), globalScalePtr, reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),        \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), sfUseUE8M0, layout, mMultiProcessorCount, stream,           \
        static_cast<int>(kernelVersion), static_cast<int>(scaleRule))

    if (sfUseUE8M0)
    {
        if (self.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_FP4_QUANTIZE_EX(half, 32);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
            LAUNCH_FP4_QUANTIZE_EX(__nv_bfloat16, 32);
        }
        else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
            LAUNCH_FP4_QUANTIZE_EX(__nv_fp8_e4m3, 32);
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize_ex only supports fp16/bf16/fp8_e4m3 input.");
        }
    }
    else
    {
        if (self.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_FP4_QUANTIZE_EX(half, 16);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
            LAUNCH_FP4_QUANTIZE_EX(__nv_bfloat16, 16);
        }
        else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
            LAUNCH_FP4_QUANTIZE_EX(__nv_fp8_e4m3, 16);
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize_ex only supports fp16/bf16/fp8_e4m3 input.");
        }
    }

#undef LAUNCH_FP4_QUANTIZE_EX

    return {valueE2M1, scaleFP8SF};
}

// ---------------------------------------------------------------------------
// calculate_global_amax
// ---------------------------------------------------------------------------

at::Tensor calculate_global_amax(at::Tensor const& input, double quantRange, double eps)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);

    auto const& inputShape = input.sizes();
    auto const rank = inputShape.size();
    TORCH_CHECK(rank >= 2, "Input must be >= 2D tensor.");

    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const n = inputShape[rank - 1];

    static int multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    int maxGridX = std::min(static_cast<int>(m), multiProcessorCount * 4);

    // Persistent internal buffers — allocated once, reused across calls.
    static at::Tensor blockMaxBuf;
    static at::Tensor retirementCount;
    static at::Tensor outputBuf;
    static int allocatedGridX = 0;

    if (allocatedGridX < maxGridX)
    {
        auto opts = input.options();
        blockMaxBuf = at::empty({maxGridX}, opts.dtype(torch::kFloat32));
        outputBuf = at::empty({2}, opts.dtype(torch::kFloat32));
        retirementCount = at::zeros({1}, opts.dtype(torch::kInt32));
        allocatedGridX = maxGridX;
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();

    if (input.scalar_type() == at::ScalarType::Half)
    {
        tensorrt_llm::kernels::computeGlobalAmax<half>(m, n, reinterpret_cast<half const*>(input.data_ptr()),
            blockMaxBuf.data_ptr<float>(), outputBuf.data_ptr<float>(), retirementCount.data_ptr<int>(),
            static_cast<float>(quantRange), static_cast<float>(eps), multiProcessorCount, stream);
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        tensorrt_llm::kernels::computeGlobalAmax<__nv_bfloat16>(m, n,
            reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()), blockMaxBuf.data_ptr<float>(),
            outputBuf.data_ptr<float>(), retirementCount.data_ptr<int>(), static_cast<float>(quantRange),
            static_cast<float>(eps), multiProcessorCount, stream);
    }
    else
    {
        TORCH_CHECK(false, "calculate_global_amax only supports fp16/bf16 input.");
    }

    return outputBuf;
}

// ---------------------------------------------------------------------------
// fp4_quantize_fused
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp4_quantize_fused(at::Tensor const& self, int64_t sfVecSize,
    bool sfUseUE8M0, bool isSfSwizzledLayout, int64_t scaleRule, double quantRange, double eps,
    int64_t testMaxActiveBlocks, int64_t forceV2)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    c10::cuda::CUDAGuard device_guard(self.device());

    TORCH_CHECK(scaleRule >= 0 && scaleRule <= 3,
        "Invalid scaleRule: ", scaleRule, ". Must be 0 (static_6), 1 (MSE), 2 (MAE), or 3 (ABS_MAX).");
    bool configEligible = (sfVecSize == 16) && (!sfUseUE8M0);

    auto const& inputShape = self.sizes();
    auto const rank = inputShape.size();
    TORCH_CHECK(rank >= 2, "Input must be >= 2D tensor.");

    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % sfVecSize == 0, "Last dimension must be divisible by sfVecSize");

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;
    at::Tensor valueE2M1 = at::empty(outputShape, self.options().dtype(FLOAT4_E2M1X2));

    auto const layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
                                           : tensorrt_llm::QuantizationSFLayout::LINEAR;

    int64_t SFSize = isSfSwizzledLayout
        ? tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize)
        : tensorrt_llm::computeLinearLayoutSFSize(m, k / sfVecSize);
    at::Tensor scaleFP8SF = at::empty({SFSize}, self.options().dtype(SF_DTYPE));
    at::Tensor amaxScale = at::empty({2}, self.options().dtype(torch::kFloat32));

    int mMultiProcessorCount = 0;
    cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, self.get_device());
    auto stream = at::cuda::getCurrentCUDAStream(self.get_device()).stream();

    int numBlocksForM = isSfSwizzledLayout ? PadUpFn(static_cast<int>(m), 128) : static_cast<int>(m);
    int blockX = std::min(static_cast<int>(k / 16), 512);
    int numBlocksPerSM = std::max(1, 2048 / blockX);
    int gridX = std::min(numBlocksForM, mMultiProcessorCount * numBlocksPerSM);

    struct FusedWorkspace
    {
        at::Tensor blockMaxBuf;
        at::Tensor retirementCount;
        int capacity = 0;
    };
    struct WsKey
    {
        int device;
        uintptr_t stream;
        bool operator==(WsKey const& other) const
        {
            return device == other.device && stream == other.stream;
        }
    };
    struct WsKeyHash
    {
        size_t operator()(WsKey const& key) const
        {
            return std::hash<int>()(key.device) ^ (std::hash<uintptr_t>()(key.stream) << 1);
        }
    };

    static constexpr size_t kMaxWsCacheEntries = 32;
    thread_local std::unordered_map<WsKey, FusedWorkspace, WsKeyHash> ws_cache;
    WsKey wsKey{self.get_device(), reinterpret_cast<uintptr_t>(stream)};
    if (ws_cache.size() >= kMaxWsCacheEntries && ws_cache.find(wsKey) == ws_cache.end())
    {
        ws_cache.erase(ws_cache.begin());
    }
    auto& ws = ws_cache[wsKey];
    if (ws.capacity < gridX)
    {
        auto opts = at::TensorOptions().device(self.device()).dtype(torch::kFloat32);
        ws.blockMaxBuf = at::empty({gridX}, opts);
        ws.retirementCount = at::zeros({1}, opts.dtype(torch::kInt32));
        ws.capacity = gridX;
    }

    bool fusedTaken = false;
    if (configEligible)
    {
#define TRY_FUSED_V1_LAUNCH(T)                                                                                       \
    fusedTaken = tensorrt_llm::kernels::invokeFusedPrologueQuantization<T, 16>(m, k,                                  \
        reinterpret_cast<T const*>(self.data_ptr()), static_cast<float>(quantRange), static_cast<float>(eps),          \
        reinterpret_cast<int64_t*>(valueE2M1.data_ptr()), reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()),           \
        sfUseUE8M0, layout, mMultiProcessorCount, ws.blockMaxBuf.data_ptr<float>(),                                    \
        ws.retirementCount.data_ptr<int>(), amaxScale.data_ptr<float>(), stream, static_cast<int>(testMaxActiveBlocks))

#define TRY_FUSED_V2_LAUNCH(T, RULE)                                                                                  \
    fusedTaken = tensorrt_llm::kernels::invokeFusedPrologueQuantizationV2<T, 16,                                      \
        tensorrt_llm::kernels::AdaptiveScaleRule::RULE>(m, k, reinterpret_cast<T const*>(self.data_ptr()),             \
        static_cast<float>(quantRange), static_cast<float>(eps), reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),     \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), layout, mMultiProcessorCount,                               \
        ws.blockMaxBuf.data_ptr<float>(), ws.retirementCount.data_ptr<int>(), amaxScale.data_ptr<float>(), stream,     \
        static_cast<int>(testMaxActiveBlocks))

#define FUSED_DISPATCH(T)                                                                                             \
    do                                                                                                                \
    {                                                                                                                 \
        if (scaleRule == 0)                                                                                           \
        {                                                                                                             \
            if (forceV2 == 0)                                                                                         \
            {                                                                                                         \
                TRY_FUSED_V1_LAUNCH(T);                                                                               \
            }                                                                                                         \
            if (!fusedTaken)                                                                                          \
            {                                                                                                         \
                TRY_FUSED_V2_LAUNCH(T, NONE);                                                                         \
            }                                                                                                         \
        }                                                                                                             \
        else if (scaleRule == 1)                                                                                      \
        {                                                                                                             \
            TRY_FUSED_V2_LAUNCH(T, MSE);                                                                              \
        }                                                                                                             \
        else if (scaleRule == 2)                                                                                      \
        {                                                                                                             \
            TRY_FUSED_V2_LAUNCH(T, MAE);                                                                              \
        }                                                                                                             \
        else if (scaleRule == 3)                                                                                      \
        {                                                                                                             \
            TRY_FUSED_V2_LAUNCH(T, ABS_MAX);                                                                          \
        }                                                                                                             \
    } while (0)

        if (self.scalar_type() == at::ScalarType::Half)
        {
            FUSED_DISPATCH(half);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
            FUSED_DISPATCH(__nv_bfloat16);
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize_fused only supports fp16/bf16 input.");
        }

#undef TRY_FUSED_V1_LAUNCH
#undef TRY_FUSED_V2_LAUNCH
#undef FUSED_DISPATCH
    }

    if (!fusedTaken)
    {
        int fallbackGridX = std::min(static_cast<int>(m), mMultiProcessorCount * 4);
        if (ws.capacity < fallbackGridX)
        {
            auto opts = at::TensorOptions().device(self.device()).dtype(torch::kFloat32);
            ws.blockMaxBuf = at::empty({fallbackGridX}, opts);
            ws.retirementCount = at::zeros({1}, opts.dtype(torch::kInt32));
            ws.capacity = fallbackGridX;
        }

        if (self.scalar_type() == at::ScalarType::Half)
        {
            tensorrt_llm::kernels::computeGlobalAmax<half>(m, k, reinterpret_cast<half const*>(self.data_ptr()),
                ws.blockMaxBuf.data_ptr<float>(), amaxScale.data_ptr<float>(), ws.retirementCount.data_ptr<int>(),
                static_cast<float>(quantRange), static_cast<float>(eps), mMultiProcessorCount, stream);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
            tensorrt_llm::kernels::computeGlobalAmax<__nv_bfloat16>(m, k,
                reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr()), ws.blockMaxBuf.data_ptr<float>(),
                amaxScale.data_ptr<float>(), ws.retirementCount.data_ptr<int>(), static_cast<float>(quantRange),
                static_cast<float>(eps), mMultiProcessorCount, stream);
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize_fused only supports fp16/bf16 input.");
        }

        at::Tensor globalScaleTensor = amaxScale.slice(0, 1, 2);
        auto [packedFallback, sfFallback] = fp4_quantize_ex(
            self, globalScaleTensor, sfVecSize, sfUseUE8M0, isSfSwizzledLayout, 1, scaleRule);
        return {packedFallback, sfFallback, amaxScale};
    }

    return {valueE2M1, scaleFP8SF, amaxScale};
}

/// Dequantize FP4 with swizzled SF layout to BF16.
/// Input: packed FP4 [M, interm_size/2] + swizzled SF (flat) + global_scale (scalar)
/// Output: BF16 [M, interm_size]
at::Tensor dequant_nvfp4_swizzled_sf(
    at::Tensor const& fp4_packed,   // [M, packed_cols], dtype uint8 or float4_e2m1fn_x2
    at::Tensor const& sf_swizzled,  // flat 1D uint8
    at::Tensor const& global_scale, // scalar float32
    int64_t sfVecSize)
{
    CHECK_TH_CUDA(fp4_packed);
    CHECK_TH_CUDA(sf_swizzled);

    int64_t m = fp4_packed.size(0);
    int64_t packed_cols = fp4_packed.size(1);
    int64_t interm_size = packed_cols * 2;

    auto output = at::empty({m, interm_size},
        fp4_packed.options().dtype(torch::kBFloat16));

    static int mpc = tensorrt_llm::common::getMultiProcessorCount();
    auto stream = at::cuda::getCurrentCUDAStream(fp4_packed.get_device()).stream();

    float gs = global_scale.item<float>();

    tensorrt_llm::kernels::invokeDequantNvfp4SwizzledSF<__nv_bfloat16>(
        m, interm_size,
        fp4_packed.data_ptr<uint8_t>(),
        sf_swizzled.data_ptr<uint8_t>(),
        gs, static_cast<int>(sfVecSize),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        tensorrt_llm::QuantizationSFLayout::SWIZZLED,
        mpc, stream);

    return output;
}

} // namespace torch_ext

// ---------------------------------------------------------------------------
// Op registration under trtllm namespace
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp4_quantize_ex(Tensor input, Tensor? globalScale, int sfVecSize, "
          "bool sfUseUE8M0=False, bool isSfSwizzledLayout=True, "
          "int kernelVersion=1, int scaleRule=0) -> (Tensor, Tensor)");
    m.def("calculate_global_amax(Tensor input, float quantRange=0.0, float eps=1e-12) -> Tensor");
    m.def("fp4_quantize_fused(Tensor input, int sfVecSize, "
          "bool sfUseUE8M0=False, bool isSfSwizzledLayout=True, "
          "int scaleRule=0, float quantRange=2688.0, float eps=1e-12, "
          "int testMaxActiveBlocks=0, int forceV2=0) -> (Tensor, Tensor, Tensor)");
    m.def("dequant_nvfp4_swizzled_sf(Tensor fp4_packed, Tensor sf_swizzled, "
          "Tensor global_scale, int sfVecSize=16) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_quantize_ex", TORCH_FN(torch_ext::fp4_quantize_ex));
    m.impl("calculate_global_amax", TORCH_FN(torch_ext::calculate_global_amax));
    m.impl("fp4_quantize_fused", TORCH_FN(torch_ext::fp4_quantize_fused));
    m.impl("dequant_nvfp4_swizzled_sf", TORCH_FN(torch_ext::dequant_nvfp4_swizzled_sf));
}
