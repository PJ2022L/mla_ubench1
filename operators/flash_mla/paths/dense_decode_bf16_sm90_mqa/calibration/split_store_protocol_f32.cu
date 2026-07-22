#include "common/epilogue_stage_bench.cuh"

int main(int argc, char** argv) {
    return microbench::epilogue_stage_bench::run<
        microbench::epilogue_stage_bench::Protocol::kSplitF32>(argc, argv);
}
