#include "common/scalar_atomic_bench.cuh"

int main(int argc, char** argv) {
    return microbench::scalar_atomic::run<microbench::scalar_atomic::Fmul>(argc, argv);
}
