#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_shared_load_u32<
        microbench::memory_atomic::Variant::kSharedLoadU32Patterns>(argc, argv);
}
