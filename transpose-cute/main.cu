#include "cutlass/util/command_line.h"

#include "include/copy.h"
#include "include/transpose_naive.h"
#include "include/transpose_smem.h"
#include "include/transpose_tmastore_vectorized2.h"
#include "include/util.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  using Element = float;

  int M, N;
  cmd.get_cmd_line_argument("M", M, 10240);
  cmd.get_cmd_line_argument("N", N, 10240);

  std::cout << "Matrix size: " << M << " x " << N << std::endl;

  // printf("Baseline copy; No transpose\n");
  // benchmark<Element, false>(copy_baseline<Element>, M, N, 5);
  
  // printf("\nNaive (no tma, no smem, not vectorized):\n");
  // benchmark<Element>(transpose_naive<Element>, M, N, 5);

  // printf("\nSMEM transpose (no tma, smem passthrough, not vectorized, not swizzled):\n");
  // benchmark<Element>(transpose_smem<Element, false>, M, N, 5);

  // printf("\nSwizzle (no tma, smem passthrough, not vectorized, swizzled):\n");
  // benchmark<Element>(transpose_smem<Element, true>, M, N, 5);

  printf("\nTMA (tma, smem passthrough, vectorized, swizzled):\n");
  benchmark<Element>(transpose_tma<Element>, M, N, 5);

  return 0;
}
