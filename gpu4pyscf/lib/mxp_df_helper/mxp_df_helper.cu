#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define CUDA_CHECK(statement)                                                       \
    do                                                                              \
    {                                                                               \
        cudaError_t result = (statement);                                           \
        if (cudaSuccess != result)                                                  \
        {                                                                           \
            fprintf(stderr, "[%s:%d] CUDA failed, ", __FUNCTION__, __LINE__);       \
            fprintf(stderr, "reason: %s\n", cudaGetErrorString(result));            \
        }                                                                           \
        assert(cudaSuccess == result);                                              \
    } while (0)


#define CUBLAS_CHECK(statement)                                                     \
    do                                                                              \
    {                                                                               \
        cublasStatus_t result = (statement);                                        \
        if (CUBLAS_STATUS_SUCCESS != result)                                        \
        {                                                                           \
            fprintf(stderr, "[%s:%d] cuBLAS failed, ", __FUNCTION__, __LINE__);     \
            fprintf(stderr, "ret = %d\n", result);                                  \
        }                                                                           \
        assert(CUBLAS_STATUS_SUCCESS == result);                                    \
    } while (0)


#define ERROR_CHECK(statement, failed_reason)                                       \
    do                                                                              \
    {                                                                               \
        int result = (statement);                                                   \
        if (!result)                                                                \
        {                                                                           \
            fprintf(stderr, "[%s:%d] Check failed, ", __FUNCTION__, __LINE__);      \
            fprintf(stderr, "reason: %s\n", failed_reason);                         \
        }                                                                           \
        assert(result);                                                             \
    } while (0)


static double f64_one = 1, f64_zero = 0;
static float  f32_one = 1, f32_zero = 0;

#define GEMM_ALIGNMENT 32  // Tensor Core kernels prefer this alignment

// =============== Kernel functions ===============

template<typename OutType>
__global__ void copy_C_or_D_mat_kernel(
    const int nrow, const int padded_nrow, const int ncol, const int padded_ncol,
    const double *input, OutType* __restrict__ output
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int input_idx = row * ncol + col;
    const int output_idx = row * padded_ncol + col;
    double in_value = (row < nrow && col < ncol) ? input[input_idx] : 0.0;
    if (row >= padded_nrow || col >= padded_ncol) return;
    output[output_idx] = static_cast<OutType>(in_value);
}


template<typename OutType>
__global__ void unpack_sym_cderi_kernel(
    const int nnz, const int* __restrict__ rows, const int* __restrict__ cols,
    const int* __restrict__ blk_nnz_displs, const int* __restrict__ blk_nnz_orig_idx,
    const int padded_nao, const double* __restrict__ cderi_sparse, OutType* __restrict__ cderi
)
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int blk_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int blk_nnz_start = blk_nnz_displs[blk_id];
    const int blk_nnz_end = blk_nnz_displs[blk_id + 1];
    const int blk_nnz = blk_nnz_end - blk_nnz_start;

    if (tid < blk_nnz)
    {
        int nnz_idx = blk_nnz_orig_idx[blk_nnz_start + tid];
        int row = rows[nnz_idx];
        int col = cols[nnz_idx];
        int input_idx = blockIdx.z * nnz + nnz_idx;
        double in_value = cderi_sparse[input_idx];
        size_t padded_nao2 = padded_nao * padded_nao;
        size_t output_idx0 = blockIdx.z * padded_nao2 + row * padded_nao + col;
        size_t output_idx1 = blockIdx.z * padded_nao2 + col * padded_nao + row;
        cderi[output_idx0] = static_cast<OutType>(in_value);
        cderi[output_idx1] = static_cast<OutType>(in_value);
    }
}


template <typename T, int TILE_DIM = 32, int BLOCK_ROWS = 8>
__global__ void stride_batched_transpose_kernel(
    const int M, const int N,
    const T * __restrict__ in, size_t input_mat_stride, size_t input_mat_ld,
    T * __restrict__ out, size_t output_mat_stride, size_t output_mat_ld
)
{
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];
    const int batch = blockIdx.z;

    // Origin (top-left) of the tile in input coordinates
    const int tile_row = blockIdx.y * TILE_DIM;  // along M
    const int tile_col = blockIdx.x * TILE_DIM;  // along N

    // Thread's local indices inside tile:
    const int local_x = threadIdx.x;    // 0..TILE_DIM-1 (columns)
    const int local_y = threadIdx.y;    // 0..BLOCK_ROWS-1 (rows within load loop)

    // Load: read TILE_DIM x TILE_DIM tile from input into shared memory
    // Iterate over rows with stride BLOCK_ROWS so blockDim.y == BLOCK_ROWS
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        int in_r = tile_row + local_y + i;   // row index in [0..M)
        int in_c = tile_col + local_x;       // col index in [0..N)
        if (in_r < M && in_c < N)
        {
            const size_t idx = (size_t) batch * input_mat_stride +
                               (size_t) in_r * input_mat_ld + (size_t) in_c;
            tile[local_y + i][local_x] = in[idx];
        } else {
            tile[local_y + i][local_x] = 0;
        }
    }

    __syncthreads();

    // Store: write transposed tile from shared memory to output, swap rows and columns
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        int out_r = tile_col + local_y + i;     // new row in [0..N)
        int out_c = tile_row + local_x;         // new col in [0..M)
        if (out_r < N && out_c < M)
        {
            const size_t out_idx = (size_t) batch * output_mat_stride +
                                   (size_t) out_r * output_mat_ld + (size_t) out_c;
            out[out_idx] = tile[local_x][local_y + i];
        }
    }
}


__global__ void copy_fp32_padded_K_kernel(
    const int nao, const int padded_nao, double *mat_K,
    const double K_mat_acc, float* __restrict__ padded_K
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nao || col >= nao) return;
    int src_idx = row * padded_nao + col;
    int dst_idx = row * nao + col;
    double K_val = static_cast<double>(padded_K[src_idx]);
    mat_K[dst_idx] = K_mat_acc * mat_K[dst_idx] + K_val;
}


// =============== Helper functions ===============


template<typename T>
static T pad_size_for_gemm(const T n, const T alignment = GEMM_ALIGNMENT)
{
    assert(std::is_integral<T>::value);
    return ((n + alignment - 1) / alignment) * alignment;
}


template<typename DType>
void get_cublas_gemm_dtypes(
    cublasComputeType_t &compute_type, cudaDataType_t &A_dtype,
    cudaDataType_t &B_dtype, cudaDataType_t &C_dtype, cudaDataType_t &scale_dtype
)
{
    if constexpr (std::is_same_v<DType, double>)
    {
        compute_type = CUBLAS_COMPUTE_64F;
        A_dtype = CUDA_R_64F;
        B_dtype = CUDA_R_64F;
        C_dtype = CUDA_R_64F;
        scale_dtype = CUDA_R_64F;
    }
    if constexpr (std::is_same_v<DType, float>)
    {
        compute_type = CUBLAS_COMPUTE_32F;
        A_dtype = CUDA_R_32F;
        B_dtype = CUDA_R_32F;
        C_dtype = CUDA_R_32F;
        scale_dtype = CUDA_R_32F;
    }
}


static uint8_t *move_ptr_to_next_256B_aligned(uint8_t *ptr)
{
  return reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(ptr) + 255) &
                                     ~static_cast<uintptr_t>(255));
}

// =============== Work functions with type template parameters ===============

void DF_J_build_work(
    cudaStream_t stream, cublasHandle_t cublas_handle, const int naux_blk,
    const int npair, const int init_output, const double *cderi_sparse,
    const double *D_mat_sparse, double *J_mat_sparse, double *rho_j
)
{
    // Step 1: GEMV: cderi_sparse [naux_blk, npair] * D_mat_sparse [npair] -> rho_j [naux_blk]
    cublasOperation_t trans = CUBLAS_OP_T;
    int m = npair, n = naux_blk;  // Swap parameters to match cuBLAS's column-major order
    int ldA = npair, incx = 1, incy = 1;
    const double *A = cderi_sparse;
    const double *x = D_mat_sparse;
    double *y = rho_j;
    double *alpha = &f64_one;
    double *beta = &f64_zero;
    CUBLAS_CHECK( cublasDgemv(
        cublas_handle, trans, m, n, alpha,
        A, ldA, x, incx, beta, y, incy
    ) );

    // Step 2: GEMV: cderi_sparse^T [npair, naux_blk] * rho_j [naux_blk] -> J_mat_sparse [npair]
    trans = CUBLAS_OP_N;
    x = rho_j;
    y = J_mat_sparse;
    beta = init_output ? &f64_zero : &f64_one;
    CUBLAS_CHECK( cublasDgemv(
        cublas_handle, trans, m, n, alpha,
        A, ldA, x, incx, beta, y, incy
    ) );
}


void sort_nnz_coord_by_blocks(
    const int nnz, const int nrow, const int ncol, const int *rows, const int *cols,
    const int blk_size, int *blk_nnz_displs, int *blk_nnz_orig_idx
)
{
    int n_blk_row = (nrow + blk_size - 1) / blk_size;
    int n_blk_col = (ncol + blk_size - 1) / blk_size;
    int n_blk = n_blk_row * n_blk_col;
    int *blk_idx = (int *) malloc(sizeof(int) * nnz);
    int *blk_nnz_cnt = blk_nnz_displs + 1;
    memset(blk_nnz_cnt, 0, sizeof(int) * n_blk);

    for (int i = 0; i < nnz; i++)
    {
        int row_blk_i = rows[i] / blk_size;
        int col_blk_i = cols[i] / blk_size;
        int blk_idx_i = row_blk_i * n_blk_col + col_blk_i;
        blk_idx[i] = blk_idx_i;
        blk_nnz_cnt[blk_idx_i]++;
    }

    blk_nnz_displs[0] = 0;
    for (int i = 1; i < n_blk; i++)
        blk_nnz_cnt[i] += blk_nnz_cnt[i - 1];

    for (int i = 0; i < nnz; i++)
    {
        int blk_idx_i = blk_idx[i];
        int dst_idx = blk_nnz_displs[blk_idx_i];
        blk_nnz_orig_idx[dst_idx] = i;
        blk_nnz_displs[blk_idx_i]++;
    }

    for (int i = n_blk; i > 0; i--)
        blk_nnz_displs[i] = blk_nnz_displs[i - 1];
    blk_nnz_displs[0] = 0;
}


template<typename OutType>
void unpack_sym_cderi_work(
    cudaStream_t stream, const int npair, const int *rows_d, const int *cols_d,
    const int *blk_nnz_displs_d, const int *blk_nnz_orig_idx_d, const int naux_blk,
    const int padded_nao, const double *cderi_sparse, OutType *cderi
)
{
    dim3 block(32, 32), grid;
    grid.x = (padded_nao + block.x - 1) / block.x;
    grid.y = (padded_nao + block.y - 1) / block.y;
    grid.z = naux_blk;
    unpack_sym_cderi_kernel<OutType><<<grid, block, 0, stream>>>(
        npair, rows_d, cols_d,
        blk_nnz_displs_d, blk_nnz_orig_idx_d,
        padded_nao, cderi_sparse, cderi
    );
    CUDA_CHECK(cudaGetLastError());
}


// Transpose row-major [B, M, N] to row-major [B, N, M]
template <typename T>
void launch_stride_batched_transpose(
    cudaStream_t stream, const int n_batch, const int M, const int N,
    const T * d_in, const size_t input_mat_stride, size_t input_mat_ld,
    T * d_out, const size_t output_mat_stride, size_t output_mat_ld
)
{
    constexpr int TILE = 32;
    constexpr int BLOCK_ROWS = 8;
    dim3 blockDim(TILE, BLOCK_ROWS);
    dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, n_batch);
    stride_batched_transpose_kernel<T, TILE, BLOCK_ROWS><<<gridDim, blockDim, 0, stream>>>(
        M, N, d_in, input_mat_stride, input_mat_ld, d_out, output_mat_stride, output_mat_ld
    );
    CUDA_CHECK( cudaGetLastError() );
}


template<typename T>
void DF_K_build_subtask1(
    cudaStream_t stream, cublasHandle_t cublas_handle, const int use_Dmat,
    const int init_output, const int naux_blk, const int padded_nao, const int padded_nocc,
    const T *cderi_i, const T *C_or_D_j, void *rho_K_k, const int fp64_emu_bits = 0
)
{
    cublasComputeType_t compute_type;
    cudaDataType_t A_dtype, B_dtype, C_dtype, scale_dtype;
    get_cublas_gemm_dtypes<T>(compute_type, A_dtype, B_dtype, C_dtype, scale_dtype);

    // Step 1: einsum Lij,jk -> Lki.
    // For each L, ij,jk -> ki is doing C^T = B^T * A^T. Since the input cderi and C/D are in
    // row-major and cuBLAS takes column-major, use m == the original m, lda = the original k,
    // transA = T, n == the original n, ldb = n, transB = T.
    int m = padded_nao;
    int n = use_Dmat ? padded_nao : padded_nocc;
    int k = padded_nao;
    void *alpha = nullptr, *beta = nullptr;
    if constexpr (std::is_same_v<T, double>)
    {
        alpha = static_cast<void *>(&f64_one);
        beta = init_output ? static_cast<void *>(&f64_zero) : alpha;
    } else {
        alpha = static_cast<void *>(&f32_one);
        beta = init_output ? static_cast<void *>(&f32_zero) : alpha;
    }
    const void *A = static_cast<const void*>(cderi_i);
    const void *B = static_cast<const void*>(C_or_D_j);
    void *C = rho_K_k;
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_T;
    int ldA = k;
    int ldB = n;
    int ldC = m;
    using lli = long long int;
    lli stride_A = static_cast<lli>(m) * static_cast<lli>(k);
    lli stride_B = 0;
    lli stride_C = static_cast<lli>(m) * static_cast<lli>(n);
    int batch_count = naux_blk;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK( cublasGemmStridedBatchedEx(
        cublas_handle, transA, transB, m, n, k,
        alpha,
        A, A_dtype, ldA, stride_A,
        B, B_dtype, ldB, stride_B,
        beta,
        C, C_dtype, ldC, stride_C,
        batch_count, compute_type, algo
    ) );
}


template<typename T>
void DF_K_build_subtask1_transpose(
    cudaStream_t stream, cublasHandle_t cublas_handle, const int use_Dmat,
    const int init_output, const int naux_blk, const int padded_nao, const int padded_nocc,
    const T *cderi_i, const T *C_or_D_j, void *rho_K_k_0, void *rho_K_k, const int fp64_emu_bits = 0
)
{
    cublasComputeType_t compute_type;
    cudaDataType_t A_dtype, B_dtype, C_dtype, scale_dtype;
    get_cublas_gemm_dtypes<T>(compute_type, A_dtype, B_dtype, C_dtype, scale_dtype);

    // Step 1: einsum Lij,jk -> Lik. Just a normal GEMM, swap A and B to
    // match cuBLAS's column-major order.
    int m = use_Dmat ? padded_nao : padded_nocc;
    int n = naux_blk * padded_nao;
    int k = padded_nao;
    void *alpha = nullptr, *beta = nullptr;
    if constexpr (std::is_same_v<T, double>)
    {
        alpha = static_cast<void *>(&f64_one);
        beta = init_output ? static_cast<void *>(&f64_zero) : alpha;
    } else {
        alpha = static_cast<void *>(&f32_one);
        beta = init_output ? static_cast<void *>(&f32_zero) : alpha;
    }
    const void *A = static_cast<const void*>(C_or_D_j);
    const void *B = static_cast<const void*>(cderi_i);
    void *C = rho_K_k_0;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    int ldA = m;
    int ldB = k;
    int ldC = m;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK( cublasGemmEx(
        cublas_handle, transA, transB, m, n, k,
        alpha, A, A_dtype, ldA, B, B_dtype, ldB,
        beta, C, C_dtype, ldC, compute_type, algo
    ) );

    // Step 2: batch transpose Lik --> Lki
    size_t n_batch = naux_blk, M = padded_nao, N = use_Dmat ? padded_nao : padded_nocc;
    size_t input_mat_stride = M * N, input_mat_ld = N;
    size_t output_mat_stride = M * N, output_mat_ld = M;
    launch_stride_batched_transpose<T>(
        stream, n_batch, M, N,
        static_cast<const T *>(rho_K_k_0), input_mat_stride, input_mat_ld,
        static_cast<T *>(rho_K_k), output_mat_stride, output_mat_ld
    );
}


template<typename T>
void DF_K_build_subtask2(
    cudaStream_t stream, cublasHandle_t cublas_handle, const int use_Dmat,
    const int init_output, const int naux_blk, const int padded_nao, const int padded_nocc,
    const T *cderi_or_rho_K_i, const T *rho_K_j, void *padded_K_k, const int fp64_emu_bits = 0
)
{
    cublasComputeType_t compute_type;
    cudaDataType_t A_dtype, B_dtype, C_dtype, scale_dtype;
    get_cublas_gemm_dtypes<T>(compute_type, A_dtype, B_dtype, C_dtype, scale_dtype);

    // Step 2: einsum Lki,Lkj -> ij.
    // This is actually computing K = C^T * C, and since C is in row-major, C^T == C in
    // column-major. K is symmetric, does not matter if it is in row-major or column-major.
    int m = padded_nao;
    int n = padded_nao;
    int k = naux_blk * (use_Dmat ? padded_nao : padded_nocc);
    void *alpha = nullptr, *beta = nullptr;
    if constexpr (std::is_same_v<T, double>)
    {
        alpha = static_cast<void *>(&f64_one);
        beta = init_output ? static_cast<void *>(&f64_zero) : alpha;
    } else {
        alpha = static_cast<void *>(&f32_one);
        beta = init_output ? static_cast<void *>(&f32_zero) : alpha;
    }
    const void *A = static_cast<const void*>(cderi_or_rho_K_i);
    const void *B = static_cast<const void*>(rho_K_j);
    void *C = padded_K_k;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    int ldA = m;
    int ldB = n;
    int ldC = m;

    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK( cublasGemmEx(
        cublas_handle, transA, transB, m, n, k,
        alpha, A, A_dtype, ldA, B, B_dtype, ldB,
        beta, C, C_dtype, ldC, compute_type, algo
    ) );
}


void DF_K_build_fp64_work(
    cudaStream_t stream, cublasHandle_t cublas_handle, const int use_Dmat,
    const int naux_blk, const int nao, const int padded_nao, const int padded_nocc,
    const double *padded_cderi, const double *padded_C_or_D_mat, double *rho_K_0_buf,
    double *rho_K_buf, double *padded_K_buf, double *mat_K, const int K_mat_acc,
    const int fp64_emu_bits = 0
)
{
    setenv("OZIMMU_INTERCEPT_THRESHOLD_M", "64", 1);
    setenv("OZIMMU_INTERCEPT_THRESHOLD_N", "64", 1);
    setenv("OZIMMU_INTERCEPT_THRESHOLD_K", "64", 1);
    if (fp64_emu_bits == 0)  setenv("OZIMMU_COMPUTE_MODE", "dgemm", 1);        // Disable hijacking
    if (fp64_emu_bits == 23) setenv("OZIMMU_COMPUTE_MODE", "fp64_int8_4", 1);  // 4xint8
    if (fp64_emu_bits == 31) setenv("OZIMMU_COMPUTE_MODE", "fp64_int8_5", 1);  // 5xint8
    if (fp64_emu_bits == 39) setenv("OZIMMU_COMPUTE_MODE", "fp64_int8_6", 1);  // 6xint8

    int init_output = 1;
    if (fp64_emu_bits > 0)
    {
        DF_K_build_subtask1_transpose<double>(
            stream, cublas_handle, use_Dmat,
            init_output, naux_blk, padded_nao, padded_nocc,
            padded_cderi, padded_C_or_D_mat, rho_K_0_buf, rho_K_buf, fp64_emu_bits
        );
    } else {
        DF_K_build_subtask1<double>(
            stream, cublas_handle, use_Dmat,
            init_output, naux_blk, padded_nao, padded_nocc,
            padded_cderi, padded_C_or_D_mat, rho_K_buf, fp64_emu_bits
        );
    }

    const double *padded_cderi_or_rho_K = use_Dmat ? padded_cderi : rho_K_buf;
    DF_K_build_subtask2<double>(
        stream, cublas_handle, use_Dmat,
        init_output, naux_blk, padded_nao, padded_nocc,
        padded_cderi_or_rho_K, rho_K_buf, padded_K_buf, fp64_emu_bits
    );

    setenv("OZIMMU_COMPUTE_MODE", "dgemm", 1);  // Reset to default

    cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
    int m = nao, n = nao, lda = padded_nao, ldb = nao, ldc = nao;
    double *A = padded_K_buf, *B = mat_K, *C = mat_K;
    double *alpha = &f64_one, *beta = K_mat_acc ? &f64_one : &f64_zero;
    CUBLAS_CHECK( cublasDgeam(
        cublas_handle, transa, transb, m, n,
        alpha, A, lda, beta, B, ldb, C, ldc
    ) );
}


void DF_K_build_fp32_work(
    cudaStream_t stream, cublasHandle_t cublas_handle, const int use_Dmat,
    const int naux_blk, const int nao, const int padded_nao, const int padded_nocc,
    const float *padded_cderi, const float *padded_C_or_D_mat, float *rho_K_buf,
    float *padded_K_buf, double *mat_K, const int K_mat_acc
)
{
    int init_output = 1;
    DF_K_build_subtask1<float>(
        stream, cublas_handle, use_Dmat,
        init_output, naux_blk, padded_nao, padded_nocc,
        padded_cderi, padded_C_or_D_mat, rho_K_buf
    );

    const float *padded_cderi_or_rho_K = use_Dmat ? padded_cderi : rho_K_buf;
    DF_K_build_subtask2<float>(
        stream, cublas_handle, use_Dmat,
        init_output, naux_blk, padded_nao, padded_nocc,
        padded_cderi_or_rho_K, rho_K_buf, padded_K_buf
    );

    dim3 block(32, 32);
    dim3 grid((padded_nao + block.x - 1) / block.x, (padded_nao + block.y - 1) / block.y);
    float K_mat_acc_ = K_mat_acc ? 1.0f : 0.0f;
    copy_fp32_padded_K_kernel<<<grid, block, 0, stream>>>(
        nao, padded_nao, mat_K, K_mat_acc_, padded_K_buf
    );
    CUDA_CHECK( cudaGetLastError() );
}

template<typename DType>
void DF_JK_build_work(
    cudaStream_t stream, const int naux, const int nao, const int nocc,
    const int naux_blk, const int fp64_emu_bits, const int use_Dmat,
    const int npair, const int *rows, const int *cols, const double *cderi_sparse,
    const double *C_or_D_mat, const double *D_mat_sparse, const int build_J,
    const int build_K, double *J_mat_sparse, double *K_mat, void *workbuf
)
{
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK( cublasCreate(&cublas_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, stream) );

    static const size_t cublas_workspace_bytes = 32 * 1024 * 1024;
    uint8_t *workbuf_ptr = static_cast<uint8_t *>(workbuf);
    workbuf_ptr = move_ptr_to_next_256B_aligned(workbuf_ptr);
    CUBLAS_CHECK( cublasSetWorkspace(cublas_handle, workbuf_ptr, cublas_workspace_bytes) );
    workbuf_ptr += cublas_workspace_bytes;

    if (build_J)
    {
        DF_J_build_work(
            stream, cublas_handle, naux, npair, 1, cderi_sparse, D_mat_sparse,
            J_mat_sparse, reinterpret_cast<double *>(workbuf_ptr)
        );
    }

    if (!build_K) return;
    int padded_nao = pad_size_for_gemm(nao);
    int padded_nocc = pad_size_for_gemm(nocc);
    int C_or_D_ncol = use_Dmat ? nao : nocc;
    int padded_C_or_D_ncol = use_Dmat ? padded_nao : padded_nocc;
    size_t padded_C_or_D_size = padded_nao * padded_C_or_D_ncol;
    size_t padded_cderi_size = naux_blk * padded_nao * padded_nao;
    size_t rho_K_size = naux_blk * padded_nao * padded_C_or_D_ncol;
    size_t padded_K_size = padded_nao * padded_nao;
    DType *C_or_D_buf = reinterpret_cast<DType *>(workbuf_ptr);
    DType *cderi_buf = C_or_D_buf + padded_C_or_D_size;
    DType *rho_K_buf = cderi_buf + padded_cderi_size;
    DType *rho_K_0_buf = rho_K_buf + rho_K_size;
    DType *padded_K_buf = rho_K_0_buf + rho_K_size;
    float *fp32_out_buf = reinterpret_cast<float *>(padded_K_buf + padded_K_size);

    dim3 block(32, 32);
    dim3 grid((padded_nao + block.x - 1) / block.x, (padded_C_or_D_ncol + block.y - 1) / block.y);
    copy_C_or_D_mat_kernel<DType><<<grid, block, 0, stream>>>(
        nao, padded_nao, C_or_D_ncol, padded_C_or_D_ncol,
        C_or_D_mat, C_or_D_buf
    );
    CUDA_CHECK( cudaGetLastError() );
    #undef DISPATCH_SPLIT_FP_ARRAY

    size_t nnz_idx_bytes = sizeof(int) * npair;
    int *blk_nnz_displs = (int *) malloc(nnz_idx_bytes);  // Actually sizeof(int) * (n_blk + 1)
    int *blk_nnz_orig_idx = (int *) malloc(nnz_idx_bytes);
    sort_nnz_coord_by_blocks(
        npair, padded_nao, padded_nao, rows, cols,
        block.x, blk_nnz_displs, blk_nnz_orig_idx
    );
    int *rows_d = nullptr;
    int *cols_d = nullptr;
    int *blk_nnz_displs_d = nullptr;
    int *blk_nnz_orig_idx_d = nullptr;
    CUDA_CHECK( cudaMalloc((void**) &rows_d, nnz_idx_bytes) );
    CUDA_CHECK( cudaMalloc((void**) &cols_d, nnz_idx_bytes) );
    CUDA_CHECK( cudaMalloc((void**) &blk_nnz_displs_d, nnz_idx_bytes) );
    CUDA_CHECK( cudaMalloc((void**) &blk_nnz_orig_idx_d, nnz_idx_bytes) );
    CUDA_CHECK( cudaMemcpyAsync(rows_d, rows, nnz_idx_bytes, cudaMemcpyHostToDevice, stream) );
    CUDA_CHECK( cudaMemcpyAsync(cols_d, cols, nnz_idx_bytes, cudaMemcpyHostToDevice, stream) );
    CUDA_CHECK( cudaMemcpyAsync(blk_nnz_displs_d, blk_nnz_displs, nnz_idx_bytes, cudaMemcpyHostToDevice, stream) );
    CUDA_CHECK( cudaMemcpyAsync(blk_nnz_orig_idx_d, blk_nnz_orig_idx, nnz_idx_bytes, cudaMemcpyHostToDevice, stream) );

    for (int i = 0; i < naux; i += naux_blk)
    {
        int curr_naux_blk = (i + naux_blk < naux) ? naux_blk : (naux - i);
        size_t cderi_sparse_offset = static_cast<size_t>(i) * static_cast<size_t>(npair);
        const double *cderi_sparse_blk = cderi_sparse + cderi_sparse_offset;
        unpack_sym_cderi_work<DType>(
            stream, npair, rows_d, cols_d,
            blk_nnz_displs_d, blk_nnz_orig_idx_d, curr_naux_blk,
            padded_nao, cderi_sparse_blk, cderi_buf
        );

        int K_mat_acc = (i > 0);
        if constexpr(std::is_same_v<DType, double>)
        {
            DF_K_build_fp64_work(
                stream, cublas_handle, use_Dmat,
                curr_naux_blk, nao, padded_nao, padded_nocc,
                cderi_buf, C_or_D_buf, rho_K_0_buf,
                rho_K_buf, padded_K_buf, K_mat, K_mat_acc, fp64_emu_bits
            );
        }
        if constexpr(std::is_same_v<DType, float>)
        {
            DF_K_build_fp32_work(
                stream, cublas_handle, use_Dmat,
                curr_naux_blk, nao, padded_nao, padded_nocc,
                cderi_buf, C_or_D_buf, rho_K_buf,
                padded_K_buf, K_mat, K_mat_acc
            );
        }
    }

    free(blk_nnz_displs);
    free(blk_nnz_orig_idx);
    CUDA_CHECK( cudaFree(rows_d) );
    CUDA_CHECK( cudaFree(cols_d) );
    CUDA_CHECK( cudaFree(blk_nnz_displs_d) );
    CUDA_CHECK( cudaFree(blk_nnz_orig_idx_d) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
}

// =============== C interface for Python ctypes ===============

extern "C" {

int DF_JK_build(
    cudaStream_t stream, const int naux, const int nao, const int nocc,
    const int naux_blk, const int dtype_id, const int fp64_emu_bits,
    const int use_Dmat, const int npair, const int *rows, const int *cols,
    const void *cderi_sparse, const void *C_or_D_mat, const void *D_mat_sparse,
    const int build_J, const int build_K, void *J_mat_sparse, void *K_mat, void *workbuf
)
{
    const double *cderi_sparse_d = static_cast<const double *>(cderi_sparse);
    const double *C_or_D_mat_d = static_cast<const double *>(C_or_D_mat);
    const double *D_mat_sparse_d = static_cast<const double *>(D_mat_sparse);
    double *J_mat_sparse_d = static_cast<double *>(J_mat_sparse);
    double *K_mat_d = static_cast<double *>(K_mat);
    #define DISPATCH_DF_JK_BUILD_WORK(DType) \
    do { \
        DF_JK_build_work<DType>( \
            stream, naux, nao, nocc, \
            naux_blk, fp64_emu_bits, use_Dmat, \
            npair, rows, cols, cderi_sparse_d, \
            C_or_D_mat_d, D_mat_sparse_d, build_J, \
            build_K, J_mat_sparse_d, K_mat_d, workbuf \
        ); \
    } while (0)
    if (dtype_id == 0) DISPATCH_DF_JK_BUILD_WORK(double);
    else if (dtype_id == 1) DISPATCH_DF_JK_BUILD_WORK(float);
    else
    {
        fprintf(stderr, "[ERROR][%s:%d] Unsupported dtype_id (%d).\n",
                __FUNCTION__, __LINE__, dtype_id);
        return 1;
    }
    #undef DISPATCH_DF_JK_BUILD_WORK

    CUDA_CHECK( cudaStreamSynchronize(stream) );
    CUDA_CHECK( cudaGetLastError() );
    return 0;
}

}  // End of "extern C"
