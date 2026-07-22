#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_shared_store_u32<
        microbench::memory_atomic::Variant::kSharedStoreU32Scalar>(argc, argv);
}
