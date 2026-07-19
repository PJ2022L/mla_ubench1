#include "memory_atomic_bench.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run<
        microbench::memory_atomic::Variant::kSharedStoreU64Dense>(argc, argv);
}
