#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda.h>

namespace microbench {

inline std::string cuda_driver_error(CUresult status) {
    const char* name = nullptr;
    const char* description = nullptr;
    (void)cuGetErrorName(status, &name);
    (void)cuGetErrorString(status, &description);
    std::ostringstream message;
    message << (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name);
    if (description != nullptr) {
        message << ": " << description;
    }
    return message.str();
}

template <std::size_t Rank>
struct TensorMapSpec {
    static_assert(Rank >= 2 && Rank <= 5,
                  "TMA tiled tensor rank must be between 2 and 5");

    TensorMapSpec() { element_strides.fill(1); }

    CUtensorMapDataType data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    void* global_address = nullptr;

    // Dimension zero is the contiguous dimension. Global strides are bytes
    // between successive elements of dimensions one through Rank-1.
    std::array<uint64_t, Rank> global_dims{};
    std::array<uint64_t, Rank - 1> global_strides{};
    std::array<uint32_t, Rank> box_dims{};
    std::array<uint32_t, Rank> element_strides{};

    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
};

template <std::size_t Rank>
inline void validate_tensor_map_spec(const TensorMapSpec<Rank>& spec) {
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(spec.global_address);
    if (spec.global_address == nullptr || address % 16 != 0) {
        throw std::invalid_argument(
            "TMA global_address must be non-null and 16-byte aligned");
    }
    for (std::size_t i = 0; i < Rank; ++i) {
        if (spec.global_dims[i] == 0 || spec.global_dims[i] > (uint64_t{1} << 32)) {
            throw std::invalid_argument("TMA global_dims must be in [1, 2^32]");
        }
        if (spec.box_dims[i] == 0 || spec.box_dims[i] > 256) {
            throw std::invalid_argument("TMA box_dims must be in [1, 256]");
        }
        if (spec.element_strides[i] == 0 || spec.element_strides[i] > 8) {
            throw std::invalid_argument(
                "TMA element_strides must be in [1, 8]");
        }
    }
    for (uint64_t stride : spec.global_strides) {
        if (stride == 0 || stride >= (uint64_t{1} << 40) || stride % 16 != 0) {
            throw std::invalid_argument(
                "TMA global_strides must be non-zero, < 2^40, and 16-byte aligned");
        }
    }
}

template <std::size_t Rank>
inline void encode_tensor_map(CUtensorMap& tensor_map,
                              const TensorMapSpec<Rank>& spec) {
    validate_tensor_map_spec(spec);
    const CUresult status = cuTensorMapEncodeTiled(
        &tensor_map,
        spec.data_type,
        static_cast<uint32_t>(Rank),
        spec.global_address,
        spec.global_dims.data(),
        spec.global_strides.data(),
        spec.box_dims.data(),
        spec.element_strides.data(),
        spec.interleave,
        spec.swizzle,
        spec.l2_promotion,
        spec.oob_fill);
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cuTensorMapEncodeTiled failed: " +
                                 cuda_driver_error(status));
    }
}

template <std::size_t Rank>
inline CUtensorMap encode_tensor_map(const TensorMapSpec<Rank>& spec) {
    CUtensorMap tensor_map{};
    encode_tensor_map(tensor_map, spec);
    return tensor_map;
}

inline CUtensorMap encode_tensor_map_2d(const TensorMapSpec<2>& spec) {
    return encode_tensor_map(spec);
}

inline CUtensorMap encode_tensor_map_3d(const TensorMapSpec<3>& spec) {
    return encode_tensor_map(spec);
}

inline CUtensorMap encode_tensor_map_4d(const TensorMapSpec<4>& spec) {
    return encode_tensor_map(spec);
}

inline CUtensorMap encode_tensor_map_5d(const TensorMapSpec<5>& spec) {
    return encode_tensor_map(spec);
}

}  // namespace microbench
