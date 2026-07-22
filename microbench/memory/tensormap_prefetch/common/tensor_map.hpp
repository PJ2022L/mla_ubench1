#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda.h>

namespace microbench {

inline void check_driver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    throw std::runtime_error(
        std::string(operation) + " failed: " +
        (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name) + " (" +
        (message == nullptr ? "no detail" : message) + ")");
}

inline CUtensorMap make_tma_load_64x576_b16_rank4_map(
        void* input,
        int working_pages) {
    if (working_pages <= 0) {
        throw std::invalid_argument("working_pages must be positive");
    }
    CUtensorMap map{};
    constexpr uint32_t rank = 4;
    constexpr uint64_t head_dimension = 576;
    constexpr uint64_t rows = 64;
    constexpr uint64_t page_bytes = head_dimension * rows * 2;
    // CUDA modes are [head_dim, row, head, page]. This is generic rank-4
    // shape [64, 576, 1, working_pages] with head_dim contiguous.
    const std::array<uint64_t, rank> dimensions = {
        head_dimension,
        rows,
        1,
        static_cast<uint64_t>(working_pages),
    };
    const std::array<uint64_t, rank - 1> strides = {
        head_dimension * 2,
        page_bytes,
        page_bytes,
    };
    const std::array<uint32_t, rank> box = {64, 64, 1, 1};
    const std::array<uint32_t, rank> element_strides = {1, 1, 1, 1};
    check_driver(
        cuTensorMapEncodeTiled(
            &map,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            rank,
            input,
            dimensions.data(),
            strides.data(),
            box.data(),
            element_strides.data(),
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled(load rank4 64x64 of 64x576 b16)");
    return map;
}

inline CUtensorMap make_tma_store_64x512_b16_rank4_map(
        void* output,
        int working_tiles) {
    if (working_tiles <= 0) {
        throw std::invalid_argument("working_tiles must be positive");
    }
    CUtensorMap map{};
    constexpr uint32_t rank = 4;
    const std::array<uint64_t, rank> dimensions = {
        512,
        64,
        static_cast<uint64_t>(working_tiles),
        1,
    };
    const std::array<uint64_t, rank - 1> strides = {
        512ull * 2,
        512ull * 64 * 2,
        static_cast<uint64_t>(working_tiles) * 512 * 64 * 2,
    };
    const std::array<uint32_t, rank> box = {64, 64, 1, 1};
    const std::array<uint32_t, rank> element_strides = {1, 1, 1, 1};
    check_driver(
        cuTensorMapEncodeTiled(
            &map,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            rank,
            output,
            dimensions.data(),
            strides.data(),
            box.data(),
            element_strides.data(),
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled(store rank4 64x512 b16)");
    return map;
}

}  // namespace microbench
