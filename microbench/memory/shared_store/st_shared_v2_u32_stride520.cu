#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_shared_store_8b<
        microbench::memory_atomic::Variant::kSharedStoreV2U32Stride520>(
            argc, argv);
}
