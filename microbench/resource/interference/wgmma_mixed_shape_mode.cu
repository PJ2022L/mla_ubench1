#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::interference::run<
        microbench::interference::Probe::kMixedWgmma>(argc, argv);
}
