#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace microbench {

inline void check_cuda(cudaError_t status,
                       const char* expression,
                       const char* file,
                       int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": "
            << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression)                                                \
    ::microbench::check_cuda((expression), #expression, __FILE__, __LINE__)

class Args {
public:
    Args(int argc, char** argv) {
        for (int index = 1; index < argc; ++index) {
            std::string token(argv[index]);
            if (token.rfind("--", 0) != 0 || token.size() == 2) {
                throw std::invalid_argument(
                    "expected an option beginning with --, got: " + token);
            }

            token.erase(0, 2);
            std::string name;
            std::string value;
            const std::size_t equal = token.find('=');
            if (equal != std::string::npos) {
                name = token.substr(0, equal);
                value = token.substr(equal + 1);
            } else {
                name = token;
                if (index + 1 == argc ||
                    std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
                    throw std::invalid_argument("--" + name +
                                                " requires a value");
                }
                value = argv[++index];
            }

            if (name.empty() || value.empty()) {
                throw std::invalid_argument("empty option name or value");
            }
            if (!values_.emplace(name, value).second) {
                throw std::invalid_argument("duplicate option: --" + name);
            }
        }
    }

    bool has(std::string_view name) const {
        return values_.find(std::string(name)) != values_.end();
    }

    std::string get_string(std::string_view name,
                           std::string default_value) const {
        const auto found = values_.find(std::string(name));
        return found == values_.end() ? std::move(default_value)
                                     : found->second;
    }

    int get_int(std::string_view name,
                int default_value,
                int minimum = std::numeric_limits<int>::min(),
                int maximum = std::numeric_limits<int>::max()) const {
        const auto found = values_.find(std::string(name));
        if (found == values_.end()) {
            if (default_value < minimum || default_value > maximum) {
                throw std::logic_error("integer default is outside its range");
            }
            return default_value;
        }

        errno = 0;
        char* end = nullptr;
        const long long parsed =
            std::strtoll(found->second.c_str(), &end, 10);
        if (errno == ERANGE || end == found->second.c_str() || *end != '\0' ||
            parsed < minimum || parsed > maximum) {
            std::ostringstream message;
            message << "--" << name << " must be an integer in [" << minimum
                    << ", " << maximum << ']';
            throw std::invalid_argument(message.str());
        }
        return static_cast<int>(parsed);
    }

    double get_double(std::string_view name,
                      double default_value,
                      double minimum =
                          -std::numeric_limits<double>::infinity(),
                      double maximum =
                          std::numeric_limits<double>::infinity()) const {
        const auto found = values_.find(std::string(name));
        if (found == values_.end()) {
            if (!std::isfinite(default_value) || default_value < minimum ||
                default_value > maximum) {
                throw std::logic_error("floating-point default is invalid");
            }
            return default_value;
        }

        errno = 0;
        char* end = nullptr;
        const double parsed = std::strtod(found->second.c_str(), &end);
        if (errno == ERANGE || end == found->second.c_str() || *end != '\0' ||
            !std::isfinite(parsed) || parsed < minimum || parsed > maximum) {
            std::ostringstream message;
            message << "--" << name << " must be finite and in [" << minimum
                    << ", " << maximum << ']';
            throw std::invalid_argument(message.str());
        }
        return parsed;
    }

    void require_only(std::initializer_list<std::string_view> allowed) const {
        for (const auto& item : values_) {
            bool recognized = false;
            for (const std::string_view candidate : allowed) {
                recognized |= item.first == candidate;
            }
            if (!recognized) {
                throw std::invalid_argument("unknown option: --" + item.first);
            }
        }
    }

private:
    std::unordered_map<std::string, std::string> values_;
};

struct CommonOptions {
    int iters;
    int warmup;
    int samples;
    int blocks;
    int device;
    double peak;
};

inline CommonOptions parse_common_options(const Args& args,
                                          int default_iters = 4096) {
    return CommonOptions{
        args.get_int("iters", default_iters, 1, 1 << 27),
        args.get_int("warmup", 3, 0, 1000),
        args.get_int("samples", 10, 1, 10000),
        args.get_int("blocks", 0, 0, 1 << 24),
        args.get_int("device", 0, 0, 1024),
        args.get_double("peak", 0.0, 0.0,
                        std::numeric_limits<double>::max()),
    };
}

__device__ __forceinline__ uint32_t read_smid() {
    uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

inline cudaDeviceProp require_sm90(int device = 0) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (device < 0 || device >= count) {
        throw std::out_of_range("CUDA device index is out of range");
    }
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0) {
        std::ostringstream message;
        message << "SM90 is required, but device " << device << " is SM"
                << properties.major << properties.minor;
        throw std::runtime_error(message.str());
    }
    return properties;
}

inline int device_clock_khz(int device) {
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &clock_khz, cudaDevAttrClockRate, device));
    if (clock_khz <= 0) {
        throw std::runtime_error("device SM clock rate is unavailable");
    }
    return clock_khz;
}

inline int resolve_blocks(int requested,
                          const cudaDeviceProp& properties,
                          int blocks_per_sm = 4) {
    if (requested != 0) {
        return requested;
    }
    if (blocks_per_sm <= 0 || properties.multiProcessorCount <= 0 ||
        properties.multiProcessorCount >
            std::numeric_limits<int>::max() / blocks_per_sm) {
        throw std::runtime_error("cannot derive a valid saturation grid");
    }
    return properties.multiProcessorCount * blocks_per_sm;
}

template <typename T>
class DeviceBuffer {
public:
    static_assert(std::is_trivially_copyable_v<T>,
                  "DeviceBuffer elements must be trivially copyable");

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { resize(count); }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    void resize(std::size_t count) {
        if (count == size_) {
            return;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("device allocation size overflow");
        }
        T* replacement = nullptr;
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replacement),
                                  count * sizeof(T)));
        }
        if (data_ != nullptr) {
            const cudaError_t status = cudaFree(data_);
            if (status != cudaSuccess) {
                if (replacement != nullptr) {
                    cudaFree(replacement);
                }
                CUDA_CHECK(status);
            }
        }
        data_ = replacement;
        size_ = count;
    }

    void zero() {
        if (size_ != 0) {
            CUDA_CHECK(cudaMemset(data_, 0, size_ * sizeof(T)));
        }
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> host(size_);
        if (size_ != 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), data_, size_ * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
        return host;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

inline double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("median requires at least one sample");
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if ((values.size() & 1U) != 0) {
        return upper;
    }
    const double lower =
        *std::max_element(values.begin(), values.begin() + middle);
    return 0.5 * (lower + upper);
}

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    operator cudaEvent_t() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename Launch>
std::vector<double> measure_event_ms(int warmup,
                                     int samples,
                                     Launch&& launch) {
    for (int index = 0; index < warmup; ++index) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start;
    CudaEvent stop;
    std::vector<double> elapsed;
    elapsed.reserve(samples);
    for (int index = 0; index < samples; ++index) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        if (!(milliseconds > 0.0f)) {
            throw std::runtime_error("CUDA event measured a non-positive time");
        }
        elapsed.push_back(static_cast<double>(milliseconds));
    }
    return elapsed;
}

template <typename Prepare, typename Launch>
std::vector<double> measure_event_ms_prepared(int warmup,
                                              int samples,
                                              Prepare&& prepare,
                                              Launch&& launch) {
    for (int index = 0; index < warmup; ++index) {
        prepare();
        CUDA_CHECK(cudaDeviceSynchronize());
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CudaEvent start;
    CudaEvent stop;
    std::vector<double> elapsed;
    elapsed.reserve(samples);
    for (int index = 0; index < samples; ++index) {
        prepare();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        if (!(milliseconds > 0.0f)) {
            throw std::runtime_error("CUDA event measured a non-positive time");
        }
        elapsed.push_back(static_cast<double>(milliseconds));
    }
    return elapsed;
}

template <typename Launch>
std::vector<double> measure_clock_cycles(int warmup,
                                         int samples,
                                         uint64_t* device_cycles,
                                         Launch&& launch,
                                         std::size_t cycle_count = 1) {
    if (cycle_count == 0) {
        throw std::invalid_argument("clock64 cycle_count must be positive");
    }
    for (int index = 0; index < warmup; ++index) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> cycles;
    cycles.reserve(samples);
    std::vector<uint64_t> host_cycles(cycle_count);
    for (int index = 0; index < samples; ++index) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_cycles.data(), device_cycles,
                              cycle_count * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        const uint64_t max_cycles =
            *std::max_element(host_cycles.begin(), host_cycles.end());
        if (max_cycles == 0) {
            throw std::runtime_error("clock64 measured zero cycles");
        }
        cycles.push_back(static_cast<double>(max_cycles));
    }
    return cycles;
}

struct PairedClockSamples {
    std::vector<double> target;
    std::vector<double> baseline;
};

template <typename Launch>
PairedClockSamples measure_paired_clock_cycles(
        int warmup,
        int samples,
        uint64_t* device_target_cycles,
        uint64_t* device_baseline_cycles,
        std::size_t cycle_count,
        Launch&& launch) {
    if (cycle_count == 0) {
        throw std::invalid_argument("clock64 cycle_count must be positive");
    }
    for (int index = 0; index < warmup; ++index) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    PairedClockSamples result;
    result.target.reserve(samples);
    result.baseline.reserve(samples);
    std::vector<uint64_t> target(cycle_count);
    std::vector<uint64_t> baseline(cycle_count);
    for (int index = 0; index < samples; ++index) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            target.data(), device_target_cycles,
            cycle_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            baseline.data(), device_baseline_cycles,
            cycle_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        const uint64_t target_max =
            *std::max_element(target.begin(), target.end());
        const uint64_t baseline_max =
            *std::max_element(baseline.begin(), baseline.end());
        if (target_max <= baseline_max || baseline_max == 0) {
            throw std::runtime_error(
                "target clock must exceed its matched loop baseline");
        }
        result.target.push_back(static_cast<double>(target_max));
        result.baseline.push_back(static_cast<double>(baseline_max));
    }
    return result;
}

template <typename Prepare, typename Launch>
PairedClockSamples measure_paired_clock_cycles_prepared(
        int warmup,
        int samples,
        uint64_t* device_target_cycles,
        uint64_t* device_baseline_cycles,
        std::size_t cycle_count,
        Prepare&& prepare,
        Launch&& launch) {
    if (cycle_count == 0) {
        throw std::invalid_argument("clock64 cycle_count must be positive");
    }
    for (int index = 0; index < warmup; ++index) {
        prepare();
        CUDA_CHECK(cudaDeviceSynchronize());
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    PairedClockSamples result;
    result.target.reserve(samples);
    result.baseline.reserve(samples);
    std::vector<uint64_t> target(cycle_count);
    std::vector<uint64_t> baseline(cycle_count);
    for (int index = 0; index < samples; ++index) {
        prepare();
        CUDA_CHECK(cudaDeviceSynchronize());
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            target.data(), device_target_cycles,
            cycle_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            baseline.data(), device_baseline_cycles,
            cycle_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        const uint64_t target_max =
            *std::max_element(target.begin(), target.end());
        const uint64_t baseline_max =
            *std::max_element(baseline.begin(), baseline.end());
        if (target_max <= baseline_max || baseline_max == 0) {
            throw std::runtime_error(
                "target clock must exceed its matched loop baseline");
        }
        result.target.push_back(static_cast<double>(target_max));
        result.baseline.push_back(static_cast<double>(baseline_max));
    }
    return result;
}

template <typename Prepare, typename Launch>
std::vector<double> measure_clock_cycles_prepared(
        int warmup,
        int samples,
        uint64_t* device_cycles,
        Prepare&& prepare,
        Launch&& launch,
        std::size_t cycle_count = 1) {
    if (cycle_count == 0) {
        throw std::invalid_argument("clock64 cycle_count must be positive");
    }
    for (int index = 0; index < warmup; ++index) {
        prepare();
        CUDA_CHECK(cudaDeviceSynchronize());
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<double> cycles;
    std::vector<uint64_t> host_cycles(cycle_count);
    cycles.reserve(samples);
    for (int index = 0; index < samples; ++index) {
        prepare();
        CUDA_CHECK(cudaDeviceSynchronize());
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_cycles.data(), device_cycles,
                              cycle_count * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        const uint64_t max_cycles =
            *std::max_element(host_cycles.begin(), host_cycles.end());
        if (max_cycles == 0) {
            throw std::runtime_error("clock64 measured zero cycles");
        }
        cycles.push_back(static_cast<double>(max_cycles));
    }
    return cycles;
}

__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
    return value;
}

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

inline std::string json_escape(std::string_view input) {
    std::ostringstream output;
    for (const unsigned char character : input) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

class JsonObject {
public:
    JsonObject& add(std::string key, std::string_view value) {
        return add_encoded(std::move(key),
                           '"' + json_escape(value) + '"');
    }

    JsonObject& add(std::string key, const std::string& value) {
        return add(std::move(key), std::string_view(value));
    }

    JsonObject& add(std::string key, const char* value) {
        if (value == nullptr) {
            return add_null(std::move(key));
        }
        return add(std::move(key), std::string_view(value));
    }

    JsonObject& add(std::string key, bool value) {
        return add_encoded(std::move(key), value ? "true" : "false");
    }

    template <typename T,
              std::enable_if_t<std::is_integral_v<T> &&
                                   !std::is_same_v<std::remove_cv_t<T>, bool>,
                               int> = 0>
    JsonObject& add(std::string key, T value) {
        return add_encoded(std::move(key), std::to_string(value));
    }

    template <typename T,
              std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    JsonObject& add(std::string key, T value) {
        if (!std::isfinite(value)) {
            return add_null(std::move(key));
        }
        std::ostringstream encoded;
        encoded << std::setprecision(17) << value;
        return add_encoded(std::move(key), encoded.str());
    }

    JsonObject& add(std::string key, const JsonObject& value) {
        return add_encoded(std::move(key), value.dump());
    }

    JsonObject& add_null(std::string key) {
        return add_encoded(std::move(key), "null");
    }

    JsonObject& add_raw(std::string key, std::string encoded_json) {
        if (encoded_json.empty()) {
            throw std::invalid_argument("raw JSON value cannot be empty");
        }
        return add_encoded(std::move(key), std::move(encoded_json));
    }

    std::string dump() const {
        std::ostringstream output;
        output << '{';
        for (std::size_t index = 0; index < fields_.size(); ++index) {
            if (index != 0) {
                output << ',';
            }
            output << '"' << json_escape(fields_[index].first) << "\":"
                   << fields_[index].second;
        }
        output << '}';
        return output.str();
    }

private:
    JsonObject& add_encoded(std::string key, std::string encoded_value) {
        for (const auto& field : fields_) {
            if (field.first == key) {
                throw std::invalid_argument("duplicate JSON key: " + key);
            }
        }
        fields_.emplace_back(std::move(key), std::move(encoded_value));
        return *this;
    }

    std::vector<std::pair<std::string, std::string>> fields_;
};

template <typename T>
inline std::string json_number_array(const std::vector<T>& values) {
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>,
                  "JSON number arrays require arithmetic non-bool elements");
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        const long double value = static_cast<long double>(values[index]);
        if (!std::isfinite(value)) {
            throw std::invalid_argument("JSON number array contains a non-finite value");
        }
        output << std::setprecision(17) << values[index];
    }
    output << ']';
    return output.str();
}

inline JsonObject metric(double value, std::string_view unit) {
    JsonObject result;
    result.add("value", value).add("unit", unit);
    return result;
}

inline JsonObject utilization(double measured,
                              double peak,
                              std::string_view peak_unit) {
    JsonObject result;
    if (peak > 0.0) {
        result.add("value", measured / peak)
            .add("percent", 100.0 * measured / peak)
            .add("peak", peak);
    } else {
        result.add_null("value").add_null("percent").add_null("peak");
    }
    result.add("unit", "ratio").add("peak_unit", peak_unit);
    return result;
}

inline void print_result(std::string_view name,
                         const JsonObject& params,
                         const JsonObject& latency,
                         const JsonObject& throughput,
                         const JsonObject& memory_bandwidth,
                         const JsonObject& hardware_utilization) {
    JsonObject root;
    root.add("name", name)
        .add("params", params)
        .add("latency", latency)
        .add("throughput", throughput)
        .add("memory_bandwidth", memory_bandwidth)
        .add("hardware_utilization", hardware_utilization);
    std::cout << root.dump() << '\n';
}

}  // namespace microbench
