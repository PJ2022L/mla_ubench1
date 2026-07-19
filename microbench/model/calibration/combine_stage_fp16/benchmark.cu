#define MB1_COMBINE_USE_F16 1
#include "combine_stage_bench.cuh"

int main(int argc, char** argv) {
    return microbench::combine_stage_bench::run(argc, argv);
}
