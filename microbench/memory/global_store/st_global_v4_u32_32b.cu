#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_global_store_record_32b<
        microbench::memory_atomic::Variant::kGlobalStoreRecord32B>(
        argc, argv);
}
