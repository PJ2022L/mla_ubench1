#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::memory_atomic::run_global_load_f32_strided<
        microbench::memory_atomic::Variant::kGlobalLoadF32Strided>(
        argc, argv);
}
