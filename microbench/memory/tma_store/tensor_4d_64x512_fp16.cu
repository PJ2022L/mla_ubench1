#define MB_TMA_USE_F16 1
#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::tma_store_bench::run(argc, argv);
}
