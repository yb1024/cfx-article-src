#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/arch/cluster_sm90.hpp>
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

#include <cute/container/tuple.hpp>

using namespace cute;

template <class TensorS, class TensorD, class Layout_tile2thr_val, class Layout_thr_val2tile, class SmemLayout,
          class SmemLayoutM_thr_val2tile>
__global__ static void __launch_bounds__(256)
    transposeKernelTMA(TensorS const tiled_tensor_S, TensorD const tiled_tensor_D, Layout_tile2thr_val const tile2thr_val,
                       Layout_thr_val2tile const thr_val2tile, SmemLayout const smemLayout,
                       SmemLayoutM_thr_val2tile const smemLayoutM_thr_val2tile)
{
  using namespace cute;
  using Element = typename TensorS::value_type;

  auto gs = tiled_tensor_S(make_coord(_, _), blockIdx.x, blockIdx.y);
  auto partitionS2SmemShape = tiled_divide(gs, product_each(shape(tile2thr_val)));
  auto partitionS_thr_val = partitionS2SmemShape.compose(thr_val2tile, _, _);
  auto gD = tiled_tensor_D(make_coord(_, _), blockIdx.y, blockIdx.x);
  auto partitionD2SmemShape = tiled_divide(gD, product_each(shape(tile2thr_val)));
  auto partitionD_thr_val = partitionD2SmemShape.compose(thr_val2tile, _, _);

  auto tSgS = partitionS_thr_val(make_coord(threadIdx.x, _), _, _);
  auto tSgS2 = group_modes<1, rank(tSgS)>(tSgS);
  Tensor tSrS = make_fragment_like(tSgS);
  auto tDgD = partitionD_thr_val(make_coord(threadIdx.x, _), _, _);
  auto tDgD2 = group_modes<1, rank(tDgD)>(tDgD);

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTranspose<Element, SmemLayout>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor smem =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayout);
  auto partitionM2SmemShape = tiled_divide(smem, product_each(shape(tile2thr_val)));
  auto partitionM_thr_val = partitionM2SmemShape.compose(thr_val2tile, _, _);
  auto partitionM_thr_val2 = group_modes<1, rank(partitionM_thr_val)>(partitionM_thr_val);
  auto tDsM = partitionM_thr_val2(make_coord(threadIdx.x, _), _);

  Tensor smem_M =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayoutM_thr_val2tile);
  auto smem_M2 = group_modes<1, rank(smem_M)>(smem_M);
  auto tMsM = smem_M2(make_coord(threadIdx.x, _), _);

  for (int i = 0; i < size<1>(tSgS2); i++)
  {
    // copy(tSgS2(_, i), tSrS);
    copy(tSgS2(_, i), tMsM(_, i));
  }
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  for (int i = 0; i < size<1>(tSgS2); i++)
  {
    copy(tDsM(_, i), tDgD2(_, i));
  }
}

template <typename Element>
void transpose_tma(TransposeParams<Element> params)
{
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

  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);       // ((bM, bN), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape_trans); // ((bN, bM), n', m')

  auto threadLayoutS =
      make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto vecLayoutS = make_layout(make_shape(Int<1>{}, Int<4>{}));
  auto tile2thr_val = raked_product(threadLayoutS, vecLayoutS);
  auto thr_val2tile = right_inverse(tile2thr_val).with_shape(make_shape(size(threadLayoutS), size(vecLayoutS)));

  auto smemLayout = tile_to_shape(cfx::getSmemLayoutK<Element, bM{}>(),
                    make_shape(shape<0>(block_shape_trans), shape<1>(block_shape_trans)));

  auto threadLayoutM =
      make_layout(make_shape(Int<8>{}, Int<32>{}));
  auto vecLayoutM = make_layout(make_shape(Int<4>{}, Int<1>{}));
  auto tile2thr_val_M = raked_product(threadLayoutM, vecLayoutM);
  auto thr_val2tile_M = right_inverse(tile2thr_val_M).with_shape(make_shape(size(threadLayoutM), size(vecLayoutM)));

  auto partitionM2Tile = tiled_divide(smemLayout, product_each(shape(tile2thr_val_M)));
  auto smemLayoutM_thr_val2tile = partitionM2Tile.compose(thr_val2tile_M, _, _);

  // print("smemLayout:\n"); print_layout(smemLayout); print("\n");
  // print("partitionM2Tile:\n"); print_layout(partitionM2Tile(make_coord(_,_), 0, 0)); print("\n");
  // print("\n"); print(smemLayout); print("\n");

  size_t smem_size =
      int(sizeof(SharedStorageTranspose<Element, decltype(smemLayout)>));

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS));

  transposeKernelTMA<<<gridDim, blockDim, smem_size>>>(tiled_tensor_S,
                                                       tiled_tensor_D, tile2thr_val,
                                                       thr_val2tile, smemLayout,
                                                       smemLayoutM_thr_val2tile);

  // print("partitionM2Tile: ");
  // print(partitionM2Tile);
  // print("\n");
  // print("thr_val2tile_M: ");
  // print(thr_val2tile_M);
  // print("\n");
  // print("smemLayoutM_thr_val2tile: ");
  // print(smemLayoutM_thr_val2tile);
  // print("\n");
}
