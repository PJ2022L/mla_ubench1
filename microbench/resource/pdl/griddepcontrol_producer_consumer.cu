#include "common/pair_harness.cuh"

int main(int argc, char** argv) {
    return microbench::griddepcontrol_producer_consumer_bench::run(argc, argv);
}
