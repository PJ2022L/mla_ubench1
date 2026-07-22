#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_global_store_u32<
        microbench::memory_atomic::Variant::kGlobalStoreU32>(
        argc, argv);
}
