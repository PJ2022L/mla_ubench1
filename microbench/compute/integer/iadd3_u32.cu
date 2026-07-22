#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::scalar_atomic::run<microbench::scalar_atomic::Iadd3>(argc, argv);
}
