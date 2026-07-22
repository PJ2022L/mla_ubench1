#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_global_store_f32<
        microbench::memory_atomic::Variant::kGlobalStoreF32>(argc, argv);
}
