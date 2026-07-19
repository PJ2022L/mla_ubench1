#include "memory_atomic_bench.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run<
        microbench::memory_atomic::Variant::kGlobalLoadI32Cached>(argc, argv);
}
