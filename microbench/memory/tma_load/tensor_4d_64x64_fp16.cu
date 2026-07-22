#define MB_TMA_USE_F16 1
#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::tma_load_bench::run<
        microbench::tma_load_bench::Mode::kTile64x64>(argc, argv);
}
