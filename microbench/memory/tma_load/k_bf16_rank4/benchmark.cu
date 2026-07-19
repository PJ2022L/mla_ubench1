#include "tma_load_bench.cuh"

int main(int argc, char** argv) {
    return microbench::tma_load_bench::run<
        microbench::tma_load_bench::Mode::kKTransaction>(argc, argv);
}
