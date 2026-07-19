#include "common/sync_atomic_bench.cuh"

int main(int argc, char** argv) {
    return microbench::sync_atomic::run<microbench::sync_atomic::MbarrierExpectTx>(argc, argv);
}
