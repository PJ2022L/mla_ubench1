#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::scalar_atomic::run<microbench::scalar_atomic::DivU32>(
        argc, argv);
}
