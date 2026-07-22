#define MB_TMA_RESULT_NAME "tma_load_4d_service"
#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::tma_load_bench::run<
        microbench::tma_load_bench::Mode::kTile64x64>(argc, argv);
}
