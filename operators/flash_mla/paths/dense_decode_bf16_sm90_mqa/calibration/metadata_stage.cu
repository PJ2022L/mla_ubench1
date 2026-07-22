#include "common/metadata_stage_bench.cuh"

int main(int argc, char** argv) {
    return microbench::metadata_stage_bench::run(argc, argv);
}
