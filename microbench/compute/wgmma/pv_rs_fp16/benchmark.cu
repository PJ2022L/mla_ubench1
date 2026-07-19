#define MB1_WGMMA_USE_F16 1
#include "wgmma_bench.cuh"

int main(int argc, char** argv) {
    return microbench::wgmma_bench::run<
        microbench::wgmma_bench::Operation::kPvRs>(argc, argv);
}
