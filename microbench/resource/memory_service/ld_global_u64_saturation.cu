#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_service::run<
        microbench::memory_service::Operation::kLoadU64>(argc, argv);
}
