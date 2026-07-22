#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::sync_atomic::run<microbench::sync_atomic::MbarrierWait<256>>(argc, argv);
}
