#pragma once
// 多次测量取稳态。SCAFFOLD。源自 ref/ubench/NVIDIA-Hopper-Benchmark/.../MeasurementSeries.hpp。
// 核心思想：每个原子跑多次，丢弃 warmup，取中位数/最小值，抑制抖动。

#include <algorithm>
#include <vector>

namespace microbench {

class MeasurementSeries {
public:
    void add(double v) { data_.push_back(v); }

    double median() {
        if (data_.empty()) return 0.0;
        std::sort(data_.begin(), data_.end());
        return data_[data_.size() / 2];
    }
    double minValue() {
        return data_.empty() ? 0.0 : *std::min_element(data_.begin(), data_.end());
    }
    // TODO(impl): spread()/mean()，以及丢弃前 N 个 warmup 的逻辑。

private:
    std::vector<double> data_;
};

}  // namespace microbench
