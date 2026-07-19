#include "wgmma_bench.cuh"

int main(int argc, char** argv) {
    return microbench::wgmma_bench::run<
        microbench::wgmma_bench::Operation::kPvRs>(argc, argv);
}
