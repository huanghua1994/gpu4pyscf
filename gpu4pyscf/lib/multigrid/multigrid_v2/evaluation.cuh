/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cartesian.cuh"
#include "constant_objects.cuh"
#include "utils.cuh"
#include <assert.h>
#include <cub/cub.cuh>
#include <gint/cuda_alloc.cuh>
#include <gint/gint.h>
#include <stdio.h>

#define BLOCK_DIM_XYZ 4

namespace gpu4pyscf::gpbc::multi_grid {

template <typename KernelType, int n_channels,
          int i_angular, int j_angular, bool is_non_orthogonal>
__global__ static void evaluate_density_kernel(
    KernelType *density, const KernelType *density_matrices,
    const int *non_trivial_pairs, const int *i_shells, const int *j_shells,
    const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int *image_pair_difference_index, const int n_difference_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  const int density_matrix_stride = n_i_functions * n_j_functions;
  const int density_matrix_channel_stride =
      density_matrix_stride * n_difference_images;

  const int block_index = sorted_block_index[blockIdx.x];
  const int n_blocks_b = (mesh_b + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;
  const int n_blocks_c = (mesh_c + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;

  const int block_a_stride = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_blocks_c;
  const int block_c_index = block_ab_index % n_blocks_c;

  const int a_start = block_a_index * BLOCK_DIM_XYZ;
  const int b_start = block_b_index * BLOCK_DIM_XYZ;
  const int c_start = block_c_index * BLOCK_DIM_XYZ;

  const KernelType start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const KernelType start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const KernelType start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const int a_upper = min(a_start + BLOCK_DIM_XYZ, mesh_a) - a_start;
  const int b_upper = min(b_start + BLOCK_DIM_XYZ, mesh_b) - b_start;
  const int c_upper = min(c_start + BLOCK_DIM_XYZ, mesh_c) - c_start;

  const int thread_id = threadIdx.x + threadIdx.y * BLOCK_DIM_XYZ +
                        threadIdx.z * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  KernelType i_cartesian[n_i_cartesian_functions];
  KernelType j_cartesian[n_j_cartesian_functions];

  KernelType
      prefactor[n_channels * n_i_cartesian_functions * n_j_cartesian_functions];

  __shared__ KernelType reduced_density_values[n_channels * n_threads];

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    reduced_density_values[i_channel * n_threads + thread_id] = 0;
  }

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  int a_index, b_index, c_index;
  KernelType x, y, z;

  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {
    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;

    const int image_index = image_indices[i_pair];
    const int image_index_i = image_index / n_images;
    const int image_index_j = image_index % n_images;
    const int image_difference_index = image_pair_difference_index[image_index];

    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
    const int i_shell = i_shells[i_shell_index];
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_shell = j_shells[j_shell_index];
    const int j_function = shell_to_ao_indices[j_shell];

    const KernelType i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
    const KernelType i_x =
        env[i_coord_offset] + vectors_to_neighboring_images[image_index_i * 3];
    const KernelType i_y = env[i_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_i * 3 + 1];
    const KernelType i_z = env[i_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_i * 3 + 2];
    const KernelType i_coeff = env[bas(PTR_COEFF, i_shell)];

    const KernelType j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
    const KernelType j_x =
        env[j_coord_offset] + vectors_to_neighboring_images[image_index_j * 3];
    const KernelType j_y = env[j_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_j * 3 + 1];
    const KernelType j_z = env[j_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_j * 3 + 2];
    const KernelType j_coeff = env[bas(PTR_COEFF, j_shell)];

    const KernelType ij_exponent = i_exponent + j_exponent;
    const KernelType ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const KernelType pair_x =
        (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const KernelType pair_y =
        (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const KernelType pair_z =
        (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const KernelType x0 = start_position_x - pair_x;
    const KernelType y0 = start_position_y - pair_y;
    const KernelType z0 = start_position_z - pair_z;

    const KernelType gaussian_exponent_at_reference =
        ij_exponent * distance_squared(x0, y0, z0);

    const KernelType pair_prefactor =
        is_valid_pair
            ? exp(-ij_exponent_in_prefactor - gaussian_exponent_at_reference) *
                  i_coeff * j_coeff * common_fac_sp<KernelType, i_angular>() *
                  common_fac_sp<KernelType, j_angular>()
            : 0;
#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
      const KernelType *density_matrix_pointer =
          density_matrices + density_matrix_channel_stride * i_channel +
          image_difference_index * density_matrix_stride +
          i_function * n_j_functions + j_function;
#pragma unroll
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
#pragma unroll
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          const KernelType density_matrix_value =
              prefactor[i_channel * n_i_cartesian_functions *
                            n_j_cartesian_functions +
                        i_function_index * n_j_cartesian_functions +
                        j_function_index] =
                  pair_prefactor * density_matrix_pointer[j_function_index];
        }
        density_matrix_pointer += n_j_functions;
      }
    }

    // From now on we assume that the lattice is orthogonal.
    // Shouldn't be too hard to extend to non-orthogonal lattices.

    const KernelType da_squared =
        distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    const KernelType db_squared =
        distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    const KernelType dc_squared =
        distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

    const KernelType exp_da_squared = exp(-2 * ij_exponent * da_squared);
    const KernelType exp_db_squared = exp(-2 * ij_exponent * db_squared);
    const KernelType exp_dc_squared = exp(-2 * ij_exponent * dc_squared);

    const KernelType cross_term_a =
        dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0;
    const KernelType cross_term_b =
        dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0;
    const KernelType cross_term_c =
        dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0;

    // BUG: when ij_exponents get too large and x0 is negative and large, the
    // exponential can overflow and return inf.
    // ideally recursion should start from the nearest grid point to the pair
    // center, instead of the fixed recursion path
    // (min a, min b, min c) -> (max a, max b, max c)
    // The inf ususally occurs when pseudo-potential is not used,
    // and core electrons appear with large exponents.
    // Potentially another fix is to have a better designed multi-grid
    // structure, where the gaussians with large exponents are evaluated
    // on a more dense grid. Around the boundary the numbers should be
    // within the range of double precision.
    // The same applies to the calculation of xc.
    const KernelType recursion_factor_a_start =
        exp(-ij_exponent * (2 * cross_term_a + da_squared));
    const KernelType recursion_factor_b_start =
        exp(-ij_exponent * (2 * cross_term_b + db_squared));
    const KernelType recursion_factor_c_start =
        exp(-ij_exponent * (2 * cross_term_c + dc_squared));

    KernelType gaussian_x, gaussian_y, gaussian_z, recursion_factor_a,
        recursion_factor_b, recursion_factor_c;
    for (a_index = 0, gaussian_x = 1,
        recursion_factor_a = recursion_factor_a_start, x = start_position_x;
         a_index < a_upper; a_index++, gaussian_x *= recursion_factor_a,
        recursion_factor_a *= exp_da_squared, x += dxyz_dabc[0]) {
      for (b_index = 0, gaussian_y = 1,
          recursion_factor_b = recursion_factor_b_start, y = start_position_y;
           b_index < b_upper; b_index++, gaussian_y *= recursion_factor_b,
          recursion_factor_b *= exp_db_squared, y += dxyz_dabc[4]) {
        for (c_index = 0, gaussian_z = 1,
            recursion_factor_c = recursion_factor_c_start, z = start_position_z;
             c_index < c_upper; c_index++, gaussian_z *= recursion_factor_c,
            recursion_factor_c *= exp_dc_squared, z += dxyz_dabc[8]) {
          gto_cartesian<KernelType, i_angular>(i_cartesian, x - i_x, y - i_y,
                                               z - i_z);
          gto_cartesian<KernelType, j_angular>(j_cartesian, x - j_x, y - j_y,
                                               z - j_z);

          const KernelType gaussian = gaussian_x * gaussian_y * gaussian_z;
#pragma unroll
          for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            KernelType density_value_to_be_shared = 0;
#pragma unroll
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
#pragma unroll
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
                density_value_to_be_shared +=
                    prefactor[i_channel * n_i_cartesian_functions *
                                  n_j_cartesian_functions +
                              i_function_index * n_j_cartesian_functions +
                              j_function_index] *
                    i_cartesian[i_function_index] *
                    j_cartesian[j_function_index];
              }
            }

            density_value_to_be_shared *= gaussian;

            __syncthreads();

            KernelType reduced =
                cub::BlockReduce<KernelType, BLOCK_DIM_XYZ,
                                 cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                                 BLOCK_DIM_XYZ, BLOCK_DIM_XYZ>()
                    .Sum(density_value_to_be_shared);
            if (thread_id == 0) {
              reduced_density_values[i_channel * n_threads +
                                     a_index * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ +
                                     b_index * BLOCK_DIM_XYZ + c_index] +=
                  reduced;
            }
          }
          if constexpr (is_non_orthogonal) {
            x += dxyz_dabc[6];
            y += dxyz_dabc[7];
          }
        }
        if constexpr (is_non_orthogonal) {
          x += dxyz_dabc[3];
          z += dxyz_dabc[5];
        }
      }
      if constexpr (is_non_orthogonal) {
        y += dxyz_dabc[1];
        z += dxyz_dabc[2];
      }
    }
  }
  a_index = a_start + threadIdx.z;
  b_index = b_start + threadIdx.y;
  c_index = c_start + threadIdx.x;

  __syncthreads();

  if (a_index < mesh_a && b_index < mesh_b && c_index < mesh_c) {
#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
      atomicAdd(density + i_channel * mesh_a * mesh_b * mesh_c +
                    a_index * mesh_b * mesh_c + b_index * mesh_c + c_index,
                reduced_density_values[i_channel * n_threads + thread_id]);
    }
  }
}

#define density_kernel_macro(li, lj)                                           \
  evaluate_density_kernel<KernelType, n_channels, li, lj,           \
                          is_non_orthogonal><<<block_grid, block_size>>>(      \
      density, density_matrices, non_trivial_pairs, i_shells, j_shells,        \
      n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,           \
      sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,         \
      sorted_block_index, image_indices, vectors_to_neighboring_images,        \
      n_images, image_pair_difference_index, n_difference_images, mesh_a,      \
      mesh_b, mesh_c, atm, bas, env)

#define density_kernel_case_macro(li, lj)                                      \
  case (li * 10 + lj):                                                         \
    density_kernel_macro(li, lj);                                              \
    break

template <typename KernelType, int n_channels, bool is_non_orthogonal>
int evaluate_density_driver(
    KernelType *density, const KernelType *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env) {
  dim3 block_size(BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);
  switch (i_angular * 10 + j_angular) {
    density_kernel_case_macro(0, 0);
    density_kernel_case_macro(0, 1);
    density_kernel_case_macro(0, 2);
    density_kernel_case_macro(0, 3);
    density_kernel_case_macro(0, 4);
    density_kernel_case_macro(1, 0);
    density_kernel_case_macro(1, 1);
    density_kernel_case_macro(1, 2);
    density_kernel_case_macro(1, 3);
    density_kernel_case_macro(1, 4);
    density_kernel_case_macro(2, 0);
    density_kernel_case_macro(2, 1);
    density_kernel_case_macro(2, 2);
    density_kernel_case_macro(2, 3);
    density_kernel_case_macro(2, 4);
    density_kernel_case_macro(3, 0);
    density_kernel_case_macro(3, 1);
    density_kernel_case_macro(3, 2);
    density_kernel_case_macro(3, 3);
    density_kernel_case_macro(3, 4);
    density_kernel_case_macro(4, 0);
    density_kernel_case_macro(4, 1);
    density_kernel_case_macro(4, 2);
    density_kernel_case_macro(4, 3);
    density_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_density_driver\n",
            i_angular, j_angular);
    return 1;
  }

  return checkCudaErrors(cudaPeekAtLastError());
}

template <typename KernelType, int n_channels, int i_angular, int j_angular,
          bool is_non_orthogonal>
__global__ static void evaluate_xc_kernel(
    KernelType *fock, const KernelType *xc_weights, const int *non_trivial_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int *image_pair_difference_index, const int n_difference_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;
  constexpr int n_xy_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  const int xc_weights_stride = mesh_a * mesh_b * mesh_c;
  const int fock_stride = n_i_functions * n_j_functions;
  const int fock_channel_stride = fock_stride * n_difference_images;

  const int block_index = sorted_block_index[blockIdx.x];

  const int n_blocks_b = (mesh_b + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;
  const int n_blocks_c = (mesh_c + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;

  const int block_a_stride = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_blocks_c;
  const int block_c_index = block_ab_index % n_blocks_c;

  const int a_start = block_a_index * BLOCK_DIM_XYZ;
  const int b_start = block_b_index * BLOCK_DIM_XYZ;
  const int c_start = block_c_index * BLOCK_DIM_XYZ;

  const KernelType start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const KernelType start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const KernelType start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const int a_upper = min(a_start + BLOCK_DIM_XYZ, mesh_a) - a_start;
  const int b_upper = min(b_start + BLOCK_DIM_XYZ, mesh_b) - b_start;
  const int c_upper = min(c_start + BLOCK_DIM_XYZ, mesh_c) - c_start;

  KernelType neighboring_gaussian_sum[n_channels * n_i_cartesian_functions *
                                      n_j_cartesian_functions];

  KernelType i_cartesian[n_i_cartesian_functions];
  KernelType j_cartesian[n_j_cartesian_functions];

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  __shared__ KernelType xc_values[n_channels * n_threads];

  int a_index = a_start + threadIdx.z;
  int b_index = b_start + threadIdx.y;
  int c_index = c_start + threadIdx.x;

  const bool out_of_boundary =
      a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c;
  KernelType xc_value = 0;

  const int thread_id =
      threadIdx.x + threadIdx.y * BLOCK_DIM_XYZ + threadIdx.z * n_xy_threads;

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    if (!out_of_boundary) {
      xc_value =
          xc_weights[i_channel * xc_weights_stride + a_index * mesh_b * mesh_c +
                     b_index * mesh_c + c_index];
    }

    xc_values[i_channel * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ +
              thread_id] = xc_value;
  }
  __syncthreads();
  KernelType x, y, z;
  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {
    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;

    const int image_index = image_indices[i_pair];
    const int image_index_i = image_index / n_images;
    const int image_index_j = image_index % n_images;
    const int image_difference_index = image_pair_difference_index[image_index];

    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
    const int i_shell = i_shells[i_shell_index];
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_shell = j_shells[j_shell_index];
    const int j_function = shell_to_ao_indices[j_shell];

    const KernelType i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
    const KernelType i_x =
        env[i_coord_offset] + vectors_to_neighboring_images[image_index_i * 3];
    const KernelType i_y = env[i_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_i * 3 + 1];
    const KernelType i_z = env[i_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_i * 3 + 2];
    const KernelType i_coeff = env[bas(PTR_COEFF, i_shell)];

    const KernelType j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
    const KernelType j_x =
        env[j_coord_offset] + vectors_to_neighboring_images[image_index_j * 3];
    const KernelType j_y = env[j_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_j * 3 + 1];
    const KernelType j_z = env[j_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_j * 3 + 2];
    const KernelType j_coeff = env[bas(PTR_COEFF, j_shell)];

    const KernelType ij_exponent = i_exponent + j_exponent;
    const KernelType ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const KernelType pair_x =
        (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const KernelType pair_y =
        (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const KernelType pair_z =
        (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const KernelType x0 = start_position_x - pair_x;
    const KernelType y0 = start_position_y - pair_y;
    const KernelType z0 = start_position_z - pair_z;

    const KernelType gaussian_exponent_at_reference =
        ij_exponent * distance_squared(x0, y0, z0);

    const KernelType pair_prefactor =
        is_valid_pair
            ? exp(-ij_exponent_in_prefactor - gaussian_exponent_at_reference) *
                  i_coeff * j_coeff * common_fac_sp<KernelType, i_angular>() *
                  common_fac_sp<KernelType, j_angular>()
            : 0;

    const KernelType da_squared =
        distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    const KernelType db_squared =
        distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    const KernelType dc_squared =
        distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

    const KernelType exp_da_squared = exp(-2 * ij_exponent * da_squared);
    const KernelType exp_db_squared = exp(-2 * ij_exponent * db_squared);
    const KernelType exp_dc_squared = exp(-2 * ij_exponent * dc_squared);

    const KernelType cross_term_a =
        dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0;
    const KernelType cross_term_b =
        dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0;
    const KernelType cross_term_c =
        dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0;

    const KernelType recursion_factor_a_start =
        exp(-ij_exponent * (2 * cross_term_a + da_squared));
    const KernelType recursion_factor_b_start =
        exp(-ij_exponent * (2 * cross_term_b + db_squared));
    const KernelType recursion_factor_c_start =
        exp(-ij_exponent * (2 * cross_term_c + dc_squared));

#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
#pragma unroll
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
#pragma unroll
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          neighboring_gaussian_sum[i_channel * n_i_cartesian_functions *
                                       n_j_cartesian_functions +
                                   i_function_index * n_j_cartesian_functions +
                                   j_function_index] = 0;
        }
      }
    }

    KernelType gaussian_x, gaussian_y, gaussian_z, recursion_factor_a,
        recursion_factor_b, recursion_factor_c;
    for (a_index = 0, gaussian_x = 1,
        recursion_factor_a = recursion_factor_a_start, x = start_position_x;
         a_index < a_upper; a_index++, gaussian_x *= recursion_factor_a,
        recursion_factor_a *= exp_da_squared, x += dxyz_dabc[0]) {
      for (b_index = 0, gaussian_y = 1,
          recursion_factor_b = recursion_factor_b_start, y = start_position_y;
           b_index < b_upper; b_index++, gaussian_y *= recursion_factor_b,
          recursion_factor_b *= exp_db_squared, y += dxyz_dabc[4]) {
        for (c_index = 0, gaussian_z = 1,
            recursion_factor_c = recursion_factor_c_start, z = start_position_z;
             c_index < c_upper; c_index++, gaussian_z *= recursion_factor_c,
            recursion_factor_c *= exp_dc_squared, z += dxyz_dabc[8]) {
          gto_cartesian<KernelType, i_angular>(i_cartesian, x - i_x, y - i_y,
                                               z - i_z);
          gto_cartesian<KernelType, j_angular>(j_cartesian, x - j_x, y - j_y,
                                               z - j_z);

          const KernelType gaussian = gaussian_x * gaussian_y * gaussian_z;
#pragma unroll
          for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            xc_value =
                gaussian *
                xc_values[i_channel * n_threads + a_index * n_xy_threads +
                          b_index * BLOCK_DIM_XYZ + c_index];
#pragma unroll
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
#pragma unroll
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
                neighboring_gaussian_sum[i_channel * n_i_cartesian_functions *
                                             n_j_cartesian_functions +
                                         i_function_index *
                                             n_j_cartesian_functions +
                                         j_function_index] +=
                    xc_value * i_cartesian[i_function_index] *
                    j_cartesian[j_function_index];
              }
            }
          }
          if constexpr (is_non_orthogonal) {
            x += dxyz_dabc[6];
            y += dxyz_dabc[7];
          }
        }
        if constexpr (!is_non_orthogonal) {
          x += dxyz_dabc[3];
          z += dxyz_dabc[5];
        }
      }
      if constexpr (is_non_orthogonal) {
        y += dxyz_dabc[1];
        z += dxyz_dabc[2];
      }
    }

    if (is_valid_pair) {
      KernelType *fock_pointer = fock + image_difference_index * fock_stride +
                                i_function * n_j_functions + j_function;

#pragma unroll
      for (int i_channel = 0; i_channel < n_channels; i_channel++) {
#pragma unroll
        for (int i_function_index = 0;
             i_function_index < n_i_cartesian_functions; i_function_index++) {
#pragma unroll
          for (int j_function_index = 0;
               j_function_index < n_j_cartesian_functions; j_function_index++) {
            atomicAdd(
                fock_pointer,
                neighboring_gaussian_sum[i_channel * n_i_cartesian_functions *
                                             n_j_cartesian_functions +
                                         i_function_index *
                                             n_j_cartesian_functions +
                                         j_function_index] *
                    pair_prefactor);
            fock_pointer++;
          }
          fock_pointer += n_j_functions - n_j_cartesian_functions;
        }
        fock_pointer +=
            fock_channel_stride - n_i_cartesian_functions * n_j_functions;
      }
    }
  }
}

#define xc_kernel_macro(li, lj)                                                \
  evaluate_xc_kernel<KernelType, n_channels, li, lj,                \
                     is_non_orthogonal><<<block_grid, block_size>>>(           \
      fock, xc_weights, non_trivial_pairs, i_shells, j_shells, n_j_shells,     \
      shell_to_ao_indices, n_i_functions, n_j_functions,                       \
      sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,         \
      sorted_block_index, image_indices, vectors_to_neighboring_images,        \
      n_images, image_pair_difference_index, n_difference_images, mesh_a,      \
      mesh_b, mesh_c, atm, bas, env)

#define xc_kernel_case_macro(li, lj)                                           \
  case (li * 10 + lj):                                                         \
    xc_kernel_macro(li, lj);                                                   \
    break

template <typename KernelType, int n_channels, bool is_non_orthogonal>
int evaluate_xc_driver(
    KernelType *fock, const KernelType *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env) {
  dim3 block_size(BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);

  switch (i_angular * 10 + j_angular) {
    xc_kernel_case_macro(0, 0);
    xc_kernel_case_macro(0, 1);
    xc_kernel_case_macro(0, 2);
    xc_kernel_case_macro(0, 3);
    xc_kernel_case_macro(0, 4);
    xc_kernel_case_macro(1, 0);
    xc_kernel_case_macro(1, 1);
    xc_kernel_case_macro(1, 2);
    xc_kernel_case_macro(1, 3);
    xc_kernel_case_macro(1, 4);
    xc_kernel_case_macro(2, 0);
    xc_kernel_case_macro(2, 1);
    xc_kernel_case_macro(2, 2);
    xc_kernel_case_macro(2, 3);
    xc_kernel_case_macro(2, 4);
    xc_kernel_case_macro(3, 0);
    xc_kernel_case_macro(3, 1);
    xc_kernel_case_macro(3, 2);
    xc_kernel_case_macro(3, 3);
    xc_kernel_case_macro(3, 4);
    xc_kernel_case_macro(4, 0);
    xc_kernel_case_macro(4, 1);
    xc_kernel_case_macro(4, 2);
    xc_kernel_case_macro(4, 3);
    xc_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_xc_driver\n",
            i_angular, j_angular);
    return 1;
  }

  return checkCudaErrors(cudaPeekAtLastError());
}

} // namespace gpu4pyscf::gpbc::multi_grid

namespace gpu4pyscf::gpbc::multi_grid::runtime_channel {

template <typename KernelType, int i_angular, int j_angular,
          bool is_non_orthogonal>
__global__ void evaluate_density_kernel(
    KernelType *density, const KernelType *density_matrices,
    const int *non_trivial_pairs, const int *i_shells, const int *j_shells,
    const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int *image_pair_difference_index, const int n_difference_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env, const int n_channels) {

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  const int density_matrix_stride = n_i_functions * n_j_functions;
  const int density_matrix_channel_stride =
      density_matrix_stride * n_difference_images;

  const int block_index = sorted_block_index[blockIdx.x];
  const int n_blocks_b = (mesh_b + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;
  const int n_blocks_c = (mesh_c + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;

  const int block_a_stride = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_blocks_c;
  const int block_c_index = block_ab_index % n_blocks_c;

  const int a_start = block_a_index * BLOCK_DIM_XYZ;
  const int b_start = block_b_index * BLOCK_DIM_XYZ;
  const int c_start = block_c_index * BLOCK_DIM_XYZ;

  const KernelType start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const KernelType start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const KernelType start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const int a_upper = min(a_start + BLOCK_DIM_XYZ, mesh_a) - a_start;
  const int b_upper = min(b_start + BLOCK_DIM_XYZ, mesh_b) - b_start;
  const int c_upper = min(c_start + BLOCK_DIM_XYZ, mesh_c) - c_start;

  const int thread_id = threadIdx.x + threadIdx.y * BLOCK_DIM_XYZ +
                        threadIdx.z * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  KernelType i_cartesian[n_i_cartesian_functions];
  KernelType j_cartesian[n_j_cartesian_functions];

  KernelType prefactor[n_i_cartesian_functions * n_j_cartesian_functions];

  __shared__ KernelType reduced_density_values[n_threads];

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  int a_index, b_index, c_index;
  KernelType x, y, z;

  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {

    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;

    const int image_index = image_indices[i_pair];
    const int image_index_i = image_index / n_images;
    const int image_index_j = image_index % n_images;
    const int image_difference_index = image_pair_difference_index[image_index];

    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
    const int i_shell = i_shells[i_shell_index];
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_shell = j_shells[j_shell_index];
    const int j_function = shell_to_ao_indices[j_shell];

    const KernelType i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
    const KernelType i_x =
        env[i_coord_offset] + vectors_to_neighboring_images[image_index_i * 3];
    const KernelType i_y = env[i_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_i * 3 + 1];
    const KernelType i_z = env[i_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_i * 3 + 2];
    const KernelType i_coeff = env[bas(PTR_COEFF, i_shell)];

    const KernelType j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
    const KernelType j_x =
        env[j_coord_offset] + vectors_to_neighboring_images[image_index_j * 3];
    const KernelType j_y = env[j_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_j * 3 + 1];
    const KernelType j_z = env[j_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_j * 3 + 2];
    const KernelType j_coeff = env[bas(PTR_COEFF, j_shell)];

    const KernelType ij_exponent = i_exponent + j_exponent;
    const KernelType ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const KernelType pair_x =
        (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const KernelType pair_y =
        (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const KernelType pair_z =
        (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const KernelType x0 = start_position_x - pair_x;
    const KernelType y0 = start_position_y - pair_y;
    const KernelType z0 = start_position_z - pair_z;

    const KernelType gaussian_exponent_at_reference =
        ij_exponent * distance_squared(x0, y0, z0);

    const KernelType pair_prefactor =
        is_valid_pair
            ? exp(-ij_exponent_in_prefactor - gaussian_exponent_at_reference) *
                  i_coeff * j_coeff * common_fac_sp<KernelType, i_angular>() *
                  common_fac_sp<KernelType, j_angular>()
            : 0;

    // From now on we assume that the lattice is orthogonal.
    // Shouldn't be too hard to extend to non-orthogonal lattices.

    const KernelType da_squared =
        distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    const KernelType db_squared =
        distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    const KernelType dc_squared =
        distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

    const KernelType exp_da_squared = exp(-2 * ij_exponent * da_squared);
    const KernelType exp_db_squared = exp(-2 * ij_exponent * db_squared);
    const KernelType exp_dc_squared = exp(-2 * ij_exponent * dc_squared);

    const KernelType cross_term_a =
        dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0;
    const KernelType cross_term_b =
        dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0;
    const KernelType cross_term_c =
        dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0;

    // BUG: when ij_exponents get too large and x0 is negative and large, the
    // exponential can overflow and return inf.
    // ideally recursion should start from the nearest grid point to the pair
    // center, instead of the fixed recursion path
    // (min a, min b, min c) -> (max a, max b, max c)
    // The inf ususally occurs when pseudo-potential is not used,
    // and core electrons appear with large exponents.
    // Potentially another fix is to have a better designed multi-grid
    // structure, where the gaussians with large exponents are evaluated
    // on a more dense grid. Around the boundary the numbers should be
    // within the range of double precision.
    // The same applies to the calculation of xc.
    const KernelType recursion_factor_a_start =
        exp(-ij_exponent * (2 * cross_term_a + da_squared));
    const KernelType recursion_factor_b_start =
        exp(-ij_exponent * (2 * cross_term_b + db_squared));
    const KernelType recursion_factor_c_start =
        exp(-ij_exponent * (2 * cross_term_c + dc_squared));

    KernelType gaussian_x, gaussian_y, gaussian_z, recursion_factor_a,
        recursion_factor_b, recursion_factor_c;

    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
      __syncthreads();

      reduced_density_values[thread_id] = 0;

      const KernelType *density_matrix_pointer =
          density_matrices + density_matrix_channel_stride * i_channel +
          image_difference_index * density_matrix_stride +
          i_function * n_j_functions + j_function;
#pragma unroll
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
#pragma unroll
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          prefactor[i_function_index * n_j_cartesian_functions +
                    j_function_index] =
              pair_prefactor * density_matrix_pointer[j_function_index];
        }
        density_matrix_pointer += n_j_functions;
      }

      for (a_index = 0, gaussian_x = 1,
          recursion_factor_a = recursion_factor_a_start, x = start_position_x;
           a_index < a_upper; a_index++, gaussian_x *= recursion_factor_a,
          recursion_factor_a *= exp_da_squared, x += dxyz_dabc[0]) {
        for (b_index = 0, gaussian_y = 1,
            recursion_factor_b = recursion_factor_b_start, y = start_position_y;
             b_index < b_upper; b_index++, gaussian_y *= recursion_factor_b,
            recursion_factor_b *= exp_db_squared, y += dxyz_dabc[4]) {
          for (c_index = 0, gaussian_z = 1,
              recursion_factor_c = recursion_factor_c_start,
              z = start_position_z;
               c_index < c_upper; c_index++, gaussian_z *= recursion_factor_c,
              recursion_factor_c *= exp_dc_squared, z += dxyz_dabc[8]) {
            gto_cartesian<KernelType, i_angular>(i_cartesian, x - i_x, y - i_y,
                                                 z - i_z);
            gto_cartesian<KernelType, j_angular>(j_cartesian, x - j_x, y - j_y,
                                                 z - j_z);

            const KernelType gaussian = gaussian_x * gaussian_y * gaussian_z;

            KernelType density_value_to_be_shared = 0;
#pragma unroll
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
#pragma unroll
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
                density_value_to_be_shared +=
                    prefactor[i_function_index * n_j_cartesian_functions +
                              j_function_index] *
                    i_cartesian[i_function_index] *
                    j_cartesian[j_function_index];
              }
            }

            density_value_to_be_shared *= gaussian;

            __syncthreads();

            const KernelType reduced =
                cub::BlockReduce<KernelType, BLOCK_DIM_XYZ,
                                 cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                                 BLOCK_DIM_XYZ, BLOCK_DIM_XYZ>()
                    .Sum(density_value_to_be_shared);

            if (thread_id == 0) {
              reduced_density_values[a_index * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ +
                                     b_index * BLOCK_DIM_XYZ + c_index] +=
                  reduced;
            }
            if constexpr (is_non_orthogonal) {
              x += dxyz_dabc[6];
              y += dxyz_dabc[7];
            }
          }
          if constexpr (is_non_orthogonal) {
            x += dxyz_dabc[3];
            z += dxyz_dabc[5];
          }
        }
        if constexpr (is_non_orthogonal) {
          y += dxyz_dabc[1];
          z += dxyz_dabc[2];
        }
      }
      a_index = a_start + threadIdx.z;
      b_index = b_start + threadIdx.y;
      c_index = c_start + threadIdx.x;

      __syncthreads();

      if (a_index < mesh_a && b_index < mesh_b && c_index < mesh_c) {
        atomicAdd(density + i_channel * mesh_a * mesh_b * mesh_c +
                      a_index * mesh_b * mesh_c + b_index * mesh_c + c_index,
                  reduced_density_values[thread_id]);
      }
    }
  }
}

#define density_kernel_runtime_channel_macro(li, lj)                           \
  evaluate_density_kernel<KernelType, li, lj, is_non_orthogonal>    \
      <<<block_grid, block_size>>>(                                            \
          density, density_matrices, non_trivial_pairs, i_shells, j_shells,    \
          n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,       \
          sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,     \
          sorted_block_index, image_indices, vectors_to_neighboring_images,    \
          n_images, image_pair_difference_index, n_difference_images, mesh_a,  \
          mesh_b, mesh_c, atm, bas, env, n_channels)

#define density_kernel_runtime_channel_case_macro(li, lj)                      \
  case (li * 10 + lj):                                                         \
    density_kernel_runtime_channel_macro(li, lj);                              \
    break

template <typename KernelType, bool is_non_orthogonal>
int evaluate_density_driver(
    KernelType *density, const KernelType *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env, const int n_channels) {
  dim3 block_size(BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);
  switch (i_angular * 10 + j_angular) {
    density_kernel_runtime_channel_case_macro(0, 0);
    density_kernel_runtime_channel_case_macro(0, 1);
    density_kernel_runtime_channel_case_macro(0, 2);
    density_kernel_runtime_channel_case_macro(0, 3);
    density_kernel_runtime_channel_case_macro(0, 4);
    density_kernel_runtime_channel_case_macro(1, 0);
    density_kernel_runtime_channel_case_macro(1, 1);
    density_kernel_runtime_channel_case_macro(1, 2);
    density_kernel_runtime_channel_case_macro(1, 3);
    density_kernel_runtime_channel_case_macro(1, 4);
    density_kernel_runtime_channel_case_macro(2, 0);
    density_kernel_runtime_channel_case_macro(2, 1);
    density_kernel_runtime_channel_case_macro(2, 2);
    density_kernel_runtime_channel_case_macro(2, 3);
    density_kernel_runtime_channel_case_macro(2, 4);
    density_kernel_runtime_channel_case_macro(3, 0);
    density_kernel_runtime_channel_case_macro(3, 1);
    density_kernel_runtime_channel_case_macro(3, 2);
    density_kernel_runtime_channel_case_macro(3, 3);
    density_kernel_runtime_channel_case_macro(3, 4);
    density_kernel_runtime_channel_case_macro(4, 0);
    density_kernel_runtime_channel_case_macro(4, 1);
    density_kernel_runtime_channel_case_macro(4, 2);
    density_kernel_runtime_channel_case_macro(4, 3);
    density_kernel_runtime_channel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_density_driver\n",
            i_angular, j_angular);
    return 1;
  }

  return checkCudaErrors(cudaPeekAtLastError());
}

template <typename KernelType, int i_angular, int j_angular,
          bool is_non_orthogonal>
__global__ void evaluate_xc_kernel(
    KernelType *fock, const KernelType *xc_weights, const int *non_trivial_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int *image_pair_difference_index, const int n_difference_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env, const int n_channels) {

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;
  constexpr int n_xy_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  const int xc_weights_stride = mesh_a * mesh_b * mesh_c;
  const int fock_stride = n_i_functions * n_j_functions;
  const int fock_channel_stride = fock_stride * n_difference_images;

  const int block_index = sorted_block_index[blockIdx.x];

  const int n_blocks_b = (mesh_b + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;
  const int n_blocks_c = (mesh_c + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;

  const int block_a_stride = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_blocks_c;
  const int block_c_index = block_ab_index % n_blocks_c;

  const int a_start = block_a_index * BLOCK_DIM_XYZ;
  const int b_start = block_b_index * BLOCK_DIM_XYZ;
  const int c_start = block_c_index * BLOCK_DIM_XYZ;

  const KernelType start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const KernelType start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const KernelType start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const int a_upper = min(a_start + BLOCK_DIM_XYZ, mesh_a) - a_start;
  const int b_upper = min(b_start + BLOCK_DIM_XYZ, mesh_b) - b_start;
  const int c_upper = min(c_start + BLOCK_DIM_XYZ, mesh_c) - c_start;

  KernelType neighboring_gaussian_sum[n_i_cartesian_functions *
                                  n_j_cartesian_functions];

  KernelType i_cartesian[n_i_cartesian_functions];
  KernelType j_cartesian[n_j_cartesian_functions];

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  __shared__ KernelType xc_values[n_threads];

  const int xc_a_index = a_start + threadIdx.z;
  const int xc_b_index = b_start + threadIdx.y;
  const int xc_c_index = c_start + threadIdx.x;

  const bool out_of_boundary =
      xc_a_index >= mesh_a || xc_b_index >= mesh_b || xc_c_index >= mesh_c;
  const int xc_read_index =
      xc_a_index * mesh_b * mesh_c + xc_b_index * mesh_c + xc_c_index;

  const int thread_id =
      threadIdx.x + threadIdx.y * BLOCK_DIM_XYZ + threadIdx.z * n_xy_threads;

  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {

    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;

    const int image_index = image_indices[i_pair];
    const int image_index_i = image_index / n_images;
    const int image_index_j = image_index % n_images;
    const int image_difference_index = image_pair_difference_index[image_index];

    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
    const int i_shell = i_shells[i_shell_index];
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_shell = j_shells[j_shell_index];
    const int j_function = shell_to_ao_indices[j_shell];

    const KernelType i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
    const KernelType i_x =
        env[i_coord_offset] + vectors_to_neighboring_images[image_index_i * 3];
    const KernelType i_y = env[i_coord_offset + 1] +
                       vectors_to_neighboring_images[image_index_i * 3 + 1];
    const KernelType i_z = env[i_coord_offset + 2] +
                       vectors_to_neighboring_images[image_index_i * 3 + 2];
    const KernelType i_coeff = env[bas(PTR_COEFF, i_shell)];

    const KernelType j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
    const KernelType j_x =
        env[j_coord_offset] + vectors_to_neighboring_images[image_index_j * 3];
    const KernelType j_y = env[j_coord_offset + 1] +
                       vectors_to_neighboring_images[image_index_j * 3 + 1];
    const KernelType j_z = env[j_coord_offset + 2] +
                       vectors_to_neighboring_images[image_index_j * 3 + 2];
    const KernelType j_coeff = env[bas(PTR_COEFF, j_shell)];

    const KernelType ij_exponent = i_exponent + j_exponent;
    const KernelType ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const KernelType pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const KernelType pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const KernelType pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const KernelType x0 = start_position_x - pair_x;
    const KernelType y0 = start_position_y - pair_y;
    const KernelType z0 = start_position_z - pair_z;

    const KernelType gaussian_exponent_at_reference =
        ij_exponent * distance_squared(x0, y0, z0);

    const KernelType pair_prefactor =
        is_valid_pair
            ? exp(-ij_exponent_in_prefactor - gaussian_exponent_at_reference) *
                  i_coeff * j_coeff * common_fac_sp<KernelType, i_angular>() *
                  common_fac_sp<KernelType, j_angular>()
            : 0;

    const KernelType da_squared =
        distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    const KernelType db_squared =
        distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    const KernelType dc_squared =
        distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

    const KernelType exp_da_squared = exp(-2 * ij_exponent * da_squared);
    const KernelType exp_db_squared = exp(-2 * ij_exponent * db_squared);
    const KernelType exp_dc_squared = exp(-2 * ij_exponent * dc_squared);

    const KernelType cross_term_a =
        dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0;
    const KernelType cross_term_b =
        dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0;
    const KernelType cross_term_c =
        dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0;

    const KernelType recursion_factor_a_start =
        exp(-ij_exponent * (2 * cross_term_a + da_squared));
    const KernelType recursion_factor_b_start =
        exp(-ij_exponent * (2 * cross_term_b + db_squared));
    const KernelType recursion_factor_c_start =
        exp(-ij_exponent * (2 * cross_term_c + dc_squared));

    KernelType gaussian_x, gaussian_y, gaussian_z, recursion_factor_a,
        recursion_factor_b, recursion_factor_c, x, y, z;

    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
      __syncthreads();

      xc_values[thread_id] =
          out_of_boundary
              ? 0
              : xc_weights[i_channel * xc_weights_stride + xc_read_index];

#pragma unroll
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
#pragma unroll
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          neighboring_gaussian_sum[i_function_index * n_j_cartesian_functions +
                                   j_function_index] = 0;
        }
      }

      __syncthreads();

      int a_index, b_index, c_index;

      for (a_index = 0, gaussian_x = 1,
          recursion_factor_a = recursion_factor_a_start, x = start_position_x;
           a_index < a_upper; a_index++, gaussian_x *= recursion_factor_a,
          recursion_factor_a *= exp_da_squared, x += dxyz_dabc[0]) {
        for (b_index = 0, gaussian_y = 1,
            recursion_factor_b = recursion_factor_b_start, y = start_position_y;
             b_index < b_upper; b_index++, gaussian_y *= recursion_factor_b,
            recursion_factor_b *= exp_db_squared, y += dxyz_dabc[4]) {
          for (c_index = 0, gaussian_z = 1,
              recursion_factor_c = recursion_factor_c_start,
              z = start_position_z;
               c_index < c_upper; c_index++, gaussian_z *= recursion_factor_c,
              recursion_factor_c *= exp_dc_squared, z += dxyz_dabc[8]) {
            gto_cartesian<KernelType, i_angular>(i_cartesian, x - i_x, y - i_y,
                                             z - i_z);
            gto_cartesian<KernelType, j_angular>(j_cartesian, x - j_x, y - j_y,
                                             z - j_z);

            const KernelType gaussian = gaussian_x * gaussian_y * gaussian_z;
            const KernelType xc_value =
                gaussian * xc_values[a_index * n_xy_threads +
                                     b_index * BLOCK_DIM_XYZ + c_index];
#pragma unroll
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
#pragma unroll
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
                neighboring_gaussian_sum[i_function_index *
                                             n_j_cartesian_functions +
                                         j_function_index] +=
                    xc_value * i_cartesian[i_function_index] *
                    j_cartesian[j_function_index];
              }
            }
            if constexpr (is_non_orthogonal) {
              x += dxyz_dabc[6];
              y += dxyz_dabc[7];
            }
          }
          if constexpr (!is_non_orthogonal) {
            x += dxyz_dabc[3];
            z += dxyz_dabc[5];
          }
        }
        if constexpr (is_non_orthogonal) {
          y += dxyz_dabc[1];
          z += dxyz_dabc[2];
        }
      }
      if (is_valid_pair) {
        KernelType *fock_pointer = fock + i_channel * fock_channel_stride +
                               image_difference_index * fock_stride +
                               i_function * n_j_functions + j_function;

#pragma unroll
        for (int i_function_index = 0;
             i_function_index < n_i_cartesian_functions; i_function_index++) {
#pragma unroll
          for (int j_function_index = 0;
               j_function_index < n_j_cartesian_functions; j_function_index++) {
            atomicAdd(fock_pointer,
                      neighboring_gaussian_sum[i_function_index *
                                                   n_j_cartesian_functions +
                                               j_function_index] *
                          pair_prefactor);
            fock_pointer++;
          }
          fock_pointer += n_j_functions - n_j_cartesian_functions;
        }
      }
    }
  }
}

#define xc_kernel_runtime_channel_macro(li, lj)                                \
  evaluate_xc_kernel<KernelType, li, lj, is_non_orthogonal>        \
      <<<block_grid, block_size>>>(                                           \
      fock, xc_weights, non_trivial_pairs, i_shells, j_shells, n_j_shells,     \
      shell_to_ao_indices, n_i_functions, n_j_functions,                       \
      sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,         \
      sorted_block_index, image_indices, vectors_to_neighboring_images,        \
      n_images, image_pair_difference_index, n_difference_images, mesh_a,      \
      mesh_b, mesh_c, atm, bas, env, n_channels)

#define xc_kernel_runtime_channel_case_macro(li, lj)                           \
  case (li * 10 + lj):                                                         \
    xc_kernel_runtime_channel_macro(li, lj);                                   \
    break

template <typename KernelType, bool is_non_orthogonal>
int evaluate_xc_driver(
    KernelType *fock, const KernelType *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env, const int n_channels) {
  dim3 block_size(BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);

  switch (i_angular * 10 + j_angular) {
    xc_kernel_runtime_channel_case_macro(0, 0);
    xc_kernel_runtime_channel_case_macro(0, 1);
    xc_kernel_runtime_channel_case_macro(0, 2);
    xc_kernel_runtime_channel_case_macro(0, 3);
    xc_kernel_runtime_channel_case_macro(0, 4);
    xc_kernel_runtime_channel_case_macro(1, 0);
    xc_kernel_runtime_channel_case_macro(1, 1);
    xc_kernel_runtime_channel_case_macro(1, 2);
    xc_kernel_runtime_channel_case_macro(1, 3);
    xc_kernel_runtime_channel_case_macro(1, 4);
    xc_kernel_runtime_channel_case_macro(2, 0);
    xc_kernel_runtime_channel_case_macro(2, 1);
    xc_kernel_runtime_channel_case_macro(2, 2);
    xc_kernel_runtime_channel_case_macro(2, 3);
    xc_kernel_runtime_channel_case_macro(2, 4);
    xc_kernel_runtime_channel_case_macro(3, 0);
    xc_kernel_runtime_channel_case_macro(3, 1);
    xc_kernel_runtime_channel_case_macro(3, 2);
    xc_kernel_runtime_channel_case_macro(3, 3);
    xc_kernel_runtime_channel_case_macro(3, 4);
    xc_kernel_runtime_channel_case_macro(4, 0);
    xc_kernel_runtime_channel_case_macro(4, 1);
    xc_kernel_runtime_channel_case_macro(4, 2);
    xc_kernel_runtime_channel_case_macro(4, 3);
    xc_kernel_runtime_channel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_xc_driver\n",
            i_angular, j_angular);
    return 1;
  }

  return checkCudaErrors(cudaPeekAtLastError());
}

} // namespace gpu4pyscf::gpbc::multi_grid::runtime_channel
