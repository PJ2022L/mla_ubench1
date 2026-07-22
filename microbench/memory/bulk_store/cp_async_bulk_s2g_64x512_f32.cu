#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::bulk_store_bench::run(argc, argv);
}
