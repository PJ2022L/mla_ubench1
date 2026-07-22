#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::scalar_atomic::run<microbench::scalar_atomic::CvtBf16>(argc, argv);
}
