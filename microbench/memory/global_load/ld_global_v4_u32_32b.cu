#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_global_load_record_32b<
        microbench::memory_atomic::Variant::kGlobalLoadRecord32B>(
        argc, argv);
}
