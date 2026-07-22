#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::tma_load_bench::run<
        microbench::tma_load_bench::Mode::kTile64x576>(argc, argv);
}
