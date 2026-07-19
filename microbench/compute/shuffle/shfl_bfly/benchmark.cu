#include "common/scalar_atomic_bench.cuh"

int main(int argc, char** argv) {
    try {
        microbench::Args args(argc, argv);
        const int delta = args.get_int("delta", 1, 1, 16);
        switch (delta) {
            case 1: return microbench::scalar_atomic::run<microbench::scalar_atomic::ShflBfly<1>>(argc, argv);
            case 2: return microbench::scalar_atomic::run<microbench::scalar_atomic::ShflBfly<2>>(argc, argv);
            case 4: return microbench::scalar_atomic::run<microbench::scalar_atomic::ShflBfly<4>>(argc, argv);
            case 8: return microbench::scalar_atomic::run<microbench::scalar_atomic::ShflBfly<8>>(argc, argv);
            case 16: return microbench::scalar_atomic::run<microbench::scalar_atomic::ShflBfly<16>>(argc, argv);
            default: throw std::invalid_argument("--delta must be one of 1,2,4,8,16");
        }
    } catch (const std::exception& error) {
        std::cerr << "shuffle benchmark error: " << error.what() << '\n';
        return 1;
    }
}
