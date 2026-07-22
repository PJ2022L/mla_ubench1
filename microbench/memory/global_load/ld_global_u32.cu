#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_global_load_i32_cached<
        microbench::memory_atomic::Variant::kGlobalLoadI32Ordinary>(argc, argv);
}
