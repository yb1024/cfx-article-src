#include <cuda.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T *data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, typename AtomMMA, typename AtomLayoutMNK>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;
  AtomMMA atom_mma;

  auto gA1 = zipped_divide(A, Shape<Int<kTileM>, Int<kTileK>>{});
  auto gA = gA1(make_coord(make_coord(_, _), make_coord(iy, _)));
  auto tile_gA_to_atom_shape = zipped_divide(gA, make_shape(size<0>(AtomMMA::Shape_MNK()), size<2>(AtomMMA::Shape_MNK())));
  auto thr_val_gA_to_atom_shape = tile_gA_to_atom_shape.compose(AtomMMA::ALayout(), _);
  auto thr_tile_gA = make_tile(_, make_shape(size<0>(AtomLayoutMNK()), size<2>(AtomLayoutMNK())));
  auto shape_per_warp_gA = zipped_divide(thr_val_gA_to_atom_shape, thr_tile_gA);

  auto gB1 = zipped_divide(B, Shape<Int<kTileN>, Int<kTileK>>{});
  auto gB = gB1(make_coord(make_coord(_, _), make_coord(ix, _)));
  auto tile_gB_to_atom_shape = zipped_divide(gB, make_shape(size<1>(AtomMMA::Shape_MNK()), size<2>(AtomMMA::Shape_MNK())));
  auto thr_val_gB_to_atom_shape = tile_gB_to_atom_shape.compose(AtomMMA::BLayout(), _);
  auto thr_tile_gB = make_tile(_, make_shape(size<1>(AtomLayoutMNK()), size<2>(AtomLayoutMNK())));
  auto shape_per_warp_gB = zipped_divide(thr_val_gB_to_atom_shape, thr_tile_gB);

  auto gC1 = zipped_divide(C, Shape<Int<kTileM>, Int<kTileN>>{});
  auto gC = gC1(make_coord(make_coord(_, _), make_coord(iy, ix)));
  auto tile_gC_to_atom_shape = zipped_divide(gC, make_shape(size<0>(AtomMMA::Shape_MNK()), size<1>(AtomMMA::Shape_MNK())));
  auto thr_val_gC_to_atom_shape = tile_gC_to_atom_shape.compose(AtomMMA::CLayout(), _);
  auto thr_tile_gC = make_tile(_, make_shape(size<0>(AtomLayoutMNK()), size<1>(AtomLayoutMNK())));
  auto shape_per_warp_gC = zipped_divide(thr_val_gC_to_atom_shape, thr_tile_gC);

  auto thr_layout_vmnk = tiled_product(AtomMMA::ThrID(), AtomLayoutMNK{});
  auto thr_vmnk = thr_layout_vmnk.get_flat_coord(threadIdx.x);

  // if(thread0() && ix==0 && iy==0){
  //   print("gA: ");print(gA);print("\n");
  //   print("gB: ");print(gB);print("\n");
  //   print("shape_per_warp_gA: ");print(shape_per_warp_gA);print("\n");
  //   print("shape_per_warp_gB: ");print(shape_per_warp_gB);print("\n");
  //   print("shape_per_warp_gC: ");print(shape_per_warp_gC);print("\n");
  //   print("thr_layout_vmnk: ");print(thr_layout_vmnk);print("\n");
  // }

  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto thr_vmn = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<2>(thr_vmnk)));
  auto tAgA = shape_per_warp_gA(thr_vmk, make_coord(_, make_coord(_, _, _)));
  auto tBgB = shape_per_warp_gB(thr_vnk, make_coord(_, make_coord(_, _, _)));
  auto tCgC = shape_per_warp_gC(thr_vmn, make_coord(_, make_coord(_, _)));

  auto tArA = make_fragment_like(tAgA(_, _, _, 0));
  auto tBrB = make_fragment_like(tBgB(_, _, _, 0));
  auto tCrC = make_fragment_like(tCgC(_, _, _));
 
  clear(tCrC);

  // if(thread0() && ix==0 && iy==0){
  //   print("tAgA: ");print(tAgA);print("\n");
  //   print("tBgB: ");print(tBgB);print("\n");
  //   print("tCgC: ");print(tCgC);print("\n");
  //   print("tArA: ");print(tArA);print("\n");
  //   print("tBrB: ");print(tBrB);print("\n");
  //   print("tCrC: ");print(tCrC);print("\n");
  // }
  
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(atom_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC);
}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  int n = 256;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  constexpr int kTileM = 128; 
  constexpr int kTileN = 128; 
  constexpr int kTileK = 32; 

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  for (int i = 0; i < 100; ++i) {
    gemm_simple<T, kTileM, kTileN, kTileK, mma_atom, Layout<Shape<_1, _4, _1>>><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 100; ++i) {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  n, m, k,
          	  &alpha,
          	  (half *)Bptr, k,
          	  (half *)Aptr, k,
          	  &beta,
          	  (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  T *Cptr_host;
  T *Cptr_cublas_host;

  Cptr_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // compare
  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.1;
  int count = 0;
  for (int i = 0; i < m * n; ++i) {
    float v1 = Cptr_host[i];
    float v2 = Cptr_cublas_host[i];
    if (fabs(v2 - v1) > threshold && count < 32) {
      printf("v1 = %f, v2 = %f\n", v1, v2);
      count++;
    }
  }

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(8, 8);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  print_tensor(tc1);
  print_tensor(tc1_cublas);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}
