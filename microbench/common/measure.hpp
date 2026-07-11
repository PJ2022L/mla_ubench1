#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace microbench {

struct MeasurementSummary {
    std::size_t count = 0;
    double min = std::numeric_limits<double>::quiet_NaN();
    double mean = std::numeric_limits<double>::quiet_NaN();
    double stddev = std::numeric_limits<double>::quiet_NaN();
    double p05 = std::numeric_limits<double>::quiet_NaN();
    double p10 = std::numeric_limits<double>::quiet_NaN();
    double median = std::numeric_limits<double>::quiet_NaN();
    double p90 = std::numeric_limits<double>::quiet_NaN();
    double p95 = std::numeric_limits<double>::quiet_NaN();
};

class MeasurementSeries {
public:
    MeasurementSeries() = default;
    explicit MeasurementSeries(std::vector<double> values)
        : data_(std::move(values)) {}

    void add(double value) { data_.push_back(value); }
    void reserve(std::size_t count) { data_.reserve(count); }
    void clear() { data_.clear(); }

    bool empty() const { return data_.empty(); }
    std::size_t size() const { return data_.size(); }
    const std::vector<double>& values() const { return data_; }

    double min() const {
        return data_.empty()
                   ? std::numeric_limits<double>::quiet_NaN()
                   : *std::min_element(data_.begin(), data_.end());
    }

    double minValue() const { return min(); }

    double mean() const {
        if (data_.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const long double sum =
            std::accumulate(data_.begin(), data_.end(), 0.0L);
        return static_cast<double>(sum / static_cast<long double>(data_.size()));
    }

    // Population standard deviation. The sample count is reported separately.
    double stddev() const {
        if (data_.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        long double running_mean = 0.0;
        long double m2 = 0.0;
        std::size_t count = 0;
        for (double value : data_) {
            ++count;
            const long double delta = value - running_mean;
            running_mean += delta / static_cast<long double>(count);
            const long double delta2 = value - running_mean;
            m2 += delta * delta2;
        }
        return std::sqrt(static_cast<double>(m2 / count));
    }

    // Linear interpolation between adjacent sorted samples. Sorting is done on
    // a copy, so querying statistics never changes insertion order.
    double quantile(double q) const {
        if (data_.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (!(q >= 0.0 && q <= 1.0)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::vector<double> sorted(data_);
        std::sort(sorted.begin(), sorted.end());
        if (sorted.size() == 1) {
            return sorted.front();
        }
        const double position = q * static_cast<double>(sorted.size() - 1);
        const std::size_t lower = static_cast<std::size_t>(position);
        const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }

    double p05() const { return quantile(0.05); }
    double p10() const { return quantile(0.10); }
    double median() const { return quantile(0.5); }
    double p90() const { return quantile(0.90); }
    double p95() const { return quantile(0.95); }

    MeasurementSummary summary() const {
        MeasurementSummary result;
        result.count = size();
        result.min = min();
        result.mean = mean();
        result.stddev = stddev();
        result.p05 = p05();
        result.p10 = p10();
        result.median = median();
        result.p90 = p90();
        result.p95 = p95();
        return result;
    }

private:
    std::vector<double> data_;
};

}  // namespace microbench
