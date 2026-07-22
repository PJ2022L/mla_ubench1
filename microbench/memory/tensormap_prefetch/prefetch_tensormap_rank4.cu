#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_tensormap_prefetch<
        microbench::memory_atomic::Variant::kTensorMapPrefetchRank4>(
        argc, argv);
}
