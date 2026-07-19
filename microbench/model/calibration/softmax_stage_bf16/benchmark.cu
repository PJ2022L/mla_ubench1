#include "common/softmax_stage_bench.cuh"

int main(int argc, char** argv) {
    return microbench::softmax_stage_bench::run(argc, argv);
}
