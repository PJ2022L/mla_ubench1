#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_global_load_v4_f32<
        microbench::memory_atomic::Variant::kGlobalLoadV4F32>(argc, argv);
}
