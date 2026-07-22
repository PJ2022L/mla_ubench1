#define MB1_WGMMA_USE_F16 1
#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::wgmma_bench::run<
        microbench::wgmma_bench::Operation::kM64N64Rs>(argc, argv);
}
