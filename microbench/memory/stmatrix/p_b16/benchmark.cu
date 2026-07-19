#include "matrix_movement_bench.cuh"

int main(int argc, char** argv) {
    return microbench::matrix_movement_bench::run<
        microbench::matrix_movement_bench::Variant::kStmatrixP>(argc, argv);
}
