#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
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

#include "measure.hpp"

namespace microbench {

inline void throw_if_cuda_error(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

class CliArgs {
public:
    CliArgs(int argc, char** argv) {
        bool positional_only = false;
        for (int i = 1; i < argc; ++i) {
            std::string token(argv[i]);
            if (positional_only) {
                positional_.push_back(std::move(token));
                continue;
            }
            if (token == "--") {
                positional_only = true;
                continue;
            }
            if (token.rfind("--", 0) != 0) {
                positional_.push_back(std::move(token));
                continue;
            }

            std::string option = token.substr(2);
            const std::size_t equal = option.find('=');
            if (equal != std::string::npos) {
                if (equal == 0) {
                    throw std::invalid_argument("empty command-line option name");
                }
                values_[option.substr(0, equal)] = option.substr(equal + 1);
                continue;
            }

            if (option.rfind("no-", 0) == 0) {
                if (option.size() == 3) {
                    throw std::invalid_argument("empty command-line option name");
                }
                values_[option.substr(3)] = "false";
                continue;
            }

            if (option.empty()) {
                throw std::invalid_argument("empty command-line option name");
            }

            if (i + 1 < argc && std::string_view(argv[i + 1]).rfind("--", 0) != 0) {
                values_[option] = argv[++i];
            } else {
                values_[option] = "true";
            }
        }
    }

    bool has(std::string_view name) const {
        return values_.find(normalize(name)) != values_.end();
    }

    std::string get_string(std::string_view name,
                           std::string default_value = {}) const {
        const auto found = values_.find(normalize(name));
        return found == values_.end() ? std::move(default_value) : found->second;
    }

    int get_int(std::string_view name, int default_value) const {
        const auto found = values_.find(normalize(name));
        return found == values_.end() ? default_value
                                     : parse_int(found->second, normalize(name));
    }

    int get_int(std::string_view name,
                int default_value,
                int minimum,
                int maximum) const {
        const int value = get_int(name, default_value);
        if (value < minimum || value > maximum) {
            throw std::out_of_range("--" + normalize(name) + " must be in [" +
                                    std::to_string(minimum) + ", " +
                                    std::to_string(maximum) + "]");
        }
        return value;
    }

    bool get_bool(std::string_view name, bool default_value) const {
        const auto found = values_.find(normalize(name));
        if (found == values_.end()) {
            return default_value;
        }
        std::string value = found->second;
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (value == "1" || value == "true" || value == "yes" ||
            value == "on") {
            return true;
        }
        if (value == "0" || value == "false" || value == "no" ||
            value == "off") {
            return false;
        }
        throw std::invalid_argument("--" + normalize(name) +
                                    " expects a boolean value");
    }

    const std::vector<std::string>& positional() const { return positional_; }

private:
    static std::string normalize(std::string_view name) {
        while (name.rfind("--", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.empty()) {
            throw std::invalid_argument("empty command-line option name");
        }
        return std::string(name);
    }

    static int parse_int(const std::string& text, const std::string& name) {
        if (text.empty()) {
            throw std::invalid_argument("--" + name + " expects an integer");
        }
        errno = 0;
        char* end = nullptr;
        const long long value = std::strtoll(text.c_str(), &end, 10);
        if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
            value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("invalid integer for --" + name + ": " +
                                        text);
        }
        return static_cast<int>(value);
    }

    std::unordered_map<std::string, std::string> values_;
    std::vector<std::string> positional_;
};

template <typename T>
class DeviceBuffer {
public:
    static_assert(std::is_trivially_copyable_v<T>,
                  "DeviceBuffer requires a trivially copyable element type");

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { resize(count); }

    ~DeviceBuffer() { release_noexcept(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release_noexcept();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void resize(std::size_t count) {
        if (count == count_) {
            return;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("DeviceBuffer allocation size overflow");
        }
        reset();
        if (count == 0) {
            return;
        }
        throw_if_cuda_error(cudaMalloc(reinterpret_cast<void**>(&data_),
                                       count * sizeof(T)),
                            "cudaMalloc");
        count_ = count;
    }

    void reset() {
        if (data_ != nullptr) {
            throw_if_cuda_error(cudaFree(data_), "cudaFree");
        }
        data_ = nullptr;
        count_ = 0;
    }

    T* release() {
        count_ = 0;
        return std::exchange(data_, nullptr);
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }
    bool empty() const { return count_ == 0; }

    explicit operator bool() const { return data_ != nullptr; }

    void copy_from_host(const T* source,
                        std::size_t count,
                        std::size_t offset = 0) {
        check_range(count, offset);
        if (count != 0 && source == nullptr) {
            throw std::invalid_argument("copy_from_host received a null source");
        }
        if (count == 0) {
            return;
        }
        throw_if_cuda_error(cudaMemcpy(data_ + offset, source, count * sizeof(T),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpyHostToDevice");
    }

    void copy_to_host(T* destination,
                      std::size_t count,
                      std::size_t offset = 0) const {
        check_range(count, offset);
        if (count != 0 && destination == nullptr) {
            throw std::invalid_argument("copy_to_host received a null destination");
        }
        if (count == 0) {
            return;
        }
        throw_if_cuda_error(cudaMemcpy(destination, data_ + offset,
                                       count * sizeof(T), cudaMemcpyDeviceToHost),
                            "cudaMemcpyDeviceToHost");
    }

    void copy_from_host(const std::vector<T>& source) {
        if (source.size() != count_) {
            throw std::invalid_argument("host vector size does not match DeviceBuffer");
        }
        copy_from_host(source.data(), source.size());
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> result(count_);
        copy_to_host(result.data(), result.size());
        return result;
    }

    void zero() {
        if (data_ != nullptr) {
            throw_if_cuda_error(cudaMemset(data_, 0, bytes()), "cudaMemset");
        }
    }

private:
    void check_range(std::size_t count, std::size_t offset) const {
        if (offset > count_ || count > count_ - offset) {
            throw std::out_of_range("DeviceBuffer copy exceeds allocation");
        }
    }

    void release_noexcept() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
        data_ = nullptr;
        count_ = 0;
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct Sm90Device {
    int ordinal = 0;
    cudaDeviceProp properties{};
};

inline Sm90Device require_sm90(int device = -1) {
    if (device < 0) {
        throw_if_cuda_error(cudaGetDevice(&device), "cudaGetDevice");
    } else {
        throw_if_cuda_error(cudaSetDevice(device), "cudaSetDevice");
    }

    Sm90Device result;
    result.ordinal = device;
    throw_if_cuda_error(cudaGetDeviceProperties(&result.properties, device),
                        "cudaGetDeviceProperties");
    if (result.properties.major != 9 || result.properties.minor != 0) {
        throw std::runtime_error(
            "SM90 benchmark requires compute capability 9.0; selected device is " +
            std::to_string(result.properties.major) + "." +
            std::to_string(result.properties.minor));
    }
    return result;
}

template <typename Fn>
void run_warmups(int warmup_count, Fn&& operation) {
    if (warmup_count < 0) {
        throw std::invalid_argument("warmup_count must be non-negative");
    }
    for (int i = 0; i < warmup_count; ++i) {
        (void)operation();
    }
}

template <typename Fn>
MeasurementSeries collect_samples(int sample_count, Fn&& measure_once) {
    if (sample_count <= 0) {
        throw std::invalid_argument("sample_count must be positive");
    }
    MeasurementSeries result;
    result.reserve(static_cast<std::size_t>(sample_count));
    for (int i = 0; i < sample_count; ++i) {
        result.add(static_cast<double>(measure_once()));
    }
    return result;
}

template <typename Fn>
MeasurementSeries run_samples(int warmup_count,
                              int sample_count,
                              Fn&& measure_once) {
    run_warmups(warmup_count, measure_once);
    return collect_samples(sample_count, measure_once);
}

class JsonLine {
public:
    JsonLine& add(std::string_view key, std::string_view value) {
        add_field(key, quote(value));
        return *this;
    }

    JsonLine& add(std::string_view key, const std::string& value) {
        return add(key, std::string_view(value));
    }

    JsonLine& add(std::string_view key, const char* value) {
        if (value == nullptr) {
            return add_null(key);
        }
        return add(key, std::string_view(value));
    }

    JsonLine& add(std::string_view key, bool value) {
        add_field(key, value ? "true" : "false");
        return *this;
    }

    template <typename T,
              typename std::enable_if_t<std::is_integral_v<T> &&
                                             !std::is_same_v<T, bool>,
                                         int> = 0>
    JsonLine& add(std::string_view key, T value) {
        add_field(key, std::to_string(value));
        return *this;
    }

    template <typename T,
              typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    JsonLine& add(std::string_view key, T value) {
        if (!std::isfinite(value)) {
            return add_null(key);
        }
        std::ostringstream stream;
        stream << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
        add_field(key, stream.str());
        return *this;
    }

    JsonLine& add_null(std::string_view key) {
        add_field(key, "null");
        return *this;
    }

    std::string str() const {
        std::string result = "{";
        for (std::size_t i = 0; i < fields_.size(); ++i) {
            if (i != 0) {
                result += ',';
            }
            result += fields_[i];
        }
        result += '}';
        return result;
    }

    void print(std::ostream& output = std::cout) const { output << str() << '\n'; }

private:
    static std::string quote(std::string_view value) {
        std::ostringstream stream;
        stream << '"';
        for (unsigned char c : value) {
            switch (c) {
                case '"': stream << "\\\""; break;
                case '\\': stream << "\\\\"; break;
                case '\b': stream << "\\b"; break;
                case '\f': stream << "\\f"; break;
                case '\n': stream << "\\n"; break;
                case '\r': stream << "\\r"; break;
                case '\t': stream << "\\t"; break;
                default:
                    if (c < 0x20) {
                        stream << "\\u00" << std::hex << std::setw(2)
                               << std::setfill('0') << static_cast<int>(c)
                               << std::dec << std::setfill(' ');
                    } else {
                        stream << static_cast<char>(c);
                    }
            }
        }
        stream << '"';
        return stream.str();
    }

    void add_field(std::string_view key, std::string value) {
        fields_.push_back(quote(key) + ':' + std::move(value));
    }

    std::vector<std::string> fields_;
};

inline JsonLine& add_measurement_summary(JsonLine& json,
                                         const MeasurementSummary& summary) {
    return json.add("samples", summary.count)
        .add("min", summary.min)
        .add("mean", summary.mean)
        .add("stddev", summary.stddev)
        .add("p05", summary.p05)
        .add("p10", summary.p10)
        .add("median", summary.median)
        .add("p90", summary.p90)
        .add("p95", summary.p95);
}

}  // namespace microbench
