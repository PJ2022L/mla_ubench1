#define MB1_WGMMA_USE_F16 1
#include "kq_calibration_bench.cuh"

int main(int argc, char** argv) {
    return microbench::kq_calibration_bench::run<
        microbench::kq_calibration_bench::Protocol::kSteadyPage>(argc, argv);
}
