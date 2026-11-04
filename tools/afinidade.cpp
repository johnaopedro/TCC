#include <omp.h>
#include <sched.h>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const int requested = std::atoi(argv[1]);
    if (requested < 1) return 2;
    omp_set_dynamic(0);
    omp_set_num_threads(requested);
    std::vector<int> cpus(requested, -1);
    int actual = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual = omp_get_num_threads();
        cpus[omp_get_thread_num()] = sched_getcpu();
    }
    std::cout << "{\"requested\":" << requested << ",\"actual\":" << actual << ",\"cpus\":[";
    for (int i = 0; i < actual; ++i) {
        if (i) std::cout << ',';
        std::cout << cpus[i];
    }
    std::cout << "]}\n";
    return actual == requested ? 0 : 1;
}
