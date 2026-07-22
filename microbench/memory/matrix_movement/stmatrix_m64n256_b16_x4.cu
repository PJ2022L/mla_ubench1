#include "common/harness.cuh"

int main(int argc, char** argv) {
    return microbench::matrix_movement_bench::run<
        microbench::matrix_movement_bench::Variant::kStmatrixM64N256>(argc, argv);
}
