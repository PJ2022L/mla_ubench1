#include "common/page_pair_transition_bench.cuh"

int main(int argc, char** argv) {
    return microbench::page_pair_transition_bench::run(argc, argv);
}
