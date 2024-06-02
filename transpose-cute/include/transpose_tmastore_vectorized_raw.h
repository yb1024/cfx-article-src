#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "shared_storage.h"
#include "smem_helper.hpp"

using namespace cute;

template <class Tiled_tensor_S, class Tile2thr_val_Sg, class Thr_val2tile_Sg, class SmemLayoutD,
          class Tile2thr_val_Ds, class Thr_val2tile_Ds, class TmaAtom, class GmemLayoutD>
__global__ static void __launch_bounds__(256)
    transposeKernelTMARaw(Tiled_tensor_S tiled_tensor_S, Tile2thr_val_Sg tile2thr_val_Sg,
                       Thr_val2tile_Sg thr_val2tile_Sg, SmemLayoutD smemLayoutD,
                       Tile2thr_val_Ds tile2thr_val_Ds, Thr_val2tile_Ds thr_val2tile_Ds,
                       TmaAtom tmaD, GmemLayoutD gmemLayoutD) {
  using namespace cute;
  using Element = typename Tiled_tensor_S::value_type;

  int lane_predicate = cute::elect_one_sync();
  int warp_idx = cutlass::canonical_warp_idx_sync();
  bool leaderWarp = warp_idx == 0;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sD =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayoutD);

  Tensor gS = tiled_tensor_S(make_coord(_, _), blockIdx.x, blockIdx.y);
  auto gS_tile = zipped_divide(gS, product_each(shape(tile2thr_val_Sg)));
  auto gS_thr_val = gS_tile.compose(thr_val2tile_Sg, _);
  auto tSgS = gS_thr_val(make_coord(threadIdx.x, _), _);

  auto sD_tile = zipped_divide(sD, product_each(shape(tile2thr_val_Ds)));
  auto sD_thr_val = sD_tile.compose(thr_val2tile_Ds, _);
  auto tDsD = sD_thr_val(make_coord(threadIdx.x, _), _);

  Tensor tSrS = make_fragment_like(tSgS(_, 0));

  // Copy from GMEM to RMEM to SMEM
  for(int i=0; i<size<1>(tSgS); i++) {
    copy(tSgS(_, i), tSrS);
    copy(tSrS, tDsD(_, i));
  }

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // Issue the TMA store.
  Tensor gD = tmaD.get_tma_tensor(shape(gmemLayoutD));
  auto gD_tile_to_smem_shape = zipped_divide(gD, shape(smemLayoutD));
  auto gD_tile_to_smem_shape_per_warp = gD_tile_to_smem_shape(_, make_coord(blockIdx.y, blockIdx.x));
  auto sD2 = group_modes<0, 2>(sD);

  if(thread0()){
    print("gD: ");  print(gD); print("\n");
    print("gD_tile_to_smem_shape: ");  print(gD_tile_to_smem_shape); print("\n");
    print("gD_tile_to_smem_shape_per_warp: "); print(gD_tile_to_smem_shape_per_warp); print("\n");
    print("sD2: ");  print(sD2); print("\n");
  }

  if (threadIdx.x==0 && blockIdx.y==0 && blockIdx.x==0) {
    copy(tmaD, sD2, gD_tile_to_smem_shape_per_warp);
  }
  // Wait for TMA store to complete.
  tma_store_wait<0>();
}

template <typename Element> void transpose_tma_raw(TransposeParams<Element> params) {
//  printf("Vectorized load into registers, write out via TMA Store\n");
//  printf("Profiler reports uncoalesced smem accesses\n");

  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

  //
  // Tile tensors
  //

  using bM = Int<32>;
  using bN = Int<32>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape_trans); // ((bN, bM), n', m')

  // info of g2s
  auto threadLayoutSg =
      make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto vecLayoutSg = make_layout(make_shape(Int<1>{}, Int<4>{}));
  auto tile2thr_val_Sg = raked_product(threadLayoutSg, vecLayoutSg);
  auto thr_val2tile_Sg = right_inverse(tile2thr_val_Sg).with_shape(make_shape(size(threadLayoutSg), size(vecLayoutSg)));

  auto tileShapeD = block_shape_trans;
  auto smemLayoutD = make_layout(tileShapeD, LayoutRight());

  auto threadLayoutDs =
      make_layout(make_shape(Int<8>{}, Int<32>{}));
  auto vecLayoutDs = make_layout(make_shape(Int<4>{}, Int<1>{}));
  auto tile2thr_val_Ds = raked_product(threadLayoutDs, vecLayoutDs);
  auto thr_val2tile_Ds = right_inverse(tile2thr_val_Ds).with_shape(make_shape(size(threadLayoutDs), size(vecLayoutDs)));

  // TMA only supports certain swizzles
  // https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm90_tma_swizzle.hpp
  auto cta_v_map = make_identity_layout(shape(tensor_D)).compose(tileShapeD);
  auto tmaD = detail::make_tma_copy_atom<Element>(SM90_TMA_STORE{}, tensor_D, smemLayoutD,
                                                       1, cta_v_map);

  auto tileShapeM = make_shape(Int<4>{}, Int<8>{}, Int<32>{});
  auto smemLayoutM = composition(smemLayoutD, make_layout(tileShapeM));
  auto threadLayoutM = make_layout(make_shape(Int<1>{}, Int<8>{}, Int<32>{}),
                                   make_stride(Int<1>{}, Int<1>{}, Int<8>{}));

  size_t smem_size =
      int(sizeof(SharedStorageTranspose<Element, decltype(smemLayoutD)>));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutSg));

  transposeKernelTMARaw<<<gridDim, blockDim, smem_size>>>(
      tiled_tensor_S, tile2thr_val_Sg, thr_val2tile_Sg, smemLayoutD, tile2thr_val_Ds, thr_val2tile_Ds, tmaD, gmemLayoutD);
}
