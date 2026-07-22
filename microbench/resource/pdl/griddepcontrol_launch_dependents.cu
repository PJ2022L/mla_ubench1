#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::pdl_atomic::run<
        microbench::pdl_atomic::Operation::kLaunchDependents>(argc, argv);
}
