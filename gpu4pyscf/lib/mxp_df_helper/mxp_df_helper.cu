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

#include "gemmul8.hpp"

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

static cublasHandle_t cublas_handle = nullptr;
static cublasLtHandle_t lt_handle = nullptr;
static void *cublas_workspace = nullptr;
static const size_t cublas_workspace_bytes = 32 * 1024 * 1024;

#define GEMM_ALIGNMENT 32  // Tensor Core kernels prefer this alignment

// =============== Kernel functions ===============

template<typename T> struct Exp2ScaleFactor;
template<> struct Exp2ScaleFactor<float> { static constexpr int value = 0; };
template<> struct Exp2ScaleFactor<__half> { static constexpr int value = 10; };
template<> struct Exp2ScaleFactor<__nv_bfloat16> { static constexpr int value = 8; };


template <typename InType, typename SplitType, int num_split>
__device__ inline void fp_split(
    InType curr_x, SplitType &output_0,
    SplitType &output_1, SplitType &output_2
)
{
    constexpr int exp2_scale_factor = Exp2ScaleFactor<SplitType>::value;

    SplitType out0 = static_cast<SplitType>(curr_x);
    output_0 = out0;
    if constexpr(num_split == 1) return;

    curr_x -= static_cast<InType>(out0);
    curr_x = scalbn(curr_x, exp2_scale_factor);
    SplitType out1 = static_cast<SplitType>(curr_x);
    output_1 = out1;
    if constexpr(num_split == 2) return;

    curr_x -= static_cast<InType>(out1);
    curr_x = scalbn(curr_x, exp2_scale_factor);
    SplitType out2 = static_cast<SplitType>(curr_x);
    output_2 = out2;
}


template <typename OutType, typename SplitType, int num_split>
__device__ inline void fp_upcast(
    OutType &output, const SplitType &input_0, const SplitType &input_1,
    const SplitType &input_2, const int exp2_scale_factor
)
{
    output = static_cast<OutType>(input_0);
    if constexpr(num_split == 1) return;

    output += scalbn(static_cast<OutType>(input_1), -exp2_scale_factor);
    if constexpr(num_split == 2) return;

    output += scalbn(static_cast<OutType>(input_2), -2 * exp2_scale_factor);
}


template<int num_split, typename InType, typename SplitType>
__global__ void split_C_or_D_mat_kernel(
    const int nrow, const int padded_nrow, const int ncol, const int padded_ncol,
    const InType *input, SplitType* __restrict__ output_0,
    SplitType* __restrict__ output_1, SplitType* __restrict__ output_2
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int input_idx = row * ncol + col;
    const int output_idx = row * padded_ncol + col;
    InType in_value = (row < nrow && col < ncol) ? input[input_idx] : static_cast<InType>(0);
    if (row >= padded_nrow || col >= padded_ncol) return;

    if constexpr(std::is_same_v<SplitType, double>) output_0[output_idx] = in_value;
    if constexpr(std::is_same_v<SplitType, float>) output_0[output_idx] = static_cast<float>(in_value);
    if constexpr(std::is_same_v<SplitType, __half> || std::is_same_v<SplitType, __nv_bfloat16>)
    {
        fp_split<InType, SplitType, num_split>(
            in_value, output_0[output_idx],
            output_1[output_idx], output_2[output_idx]
        );
    }
}


template<int num_split, typename InType, typename SplitType>
__global__ void unpack_sym_split_cderi_kernel(
    const int nnz, const int* __restrict__ rows, const int* __restrict__ cols,
    const int* __restrict__ blk_nnz_displs, const int* __restrict__ blk_nnz_orig_idx,
    const int padded_nao, const InType* __restrict__ cderi_sparse,
    SplitType* __restrict__ cderi_0, SplitType* __restrict__ cderi_1,
    SplitType* __restrict__ cderi_2
)
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int blk_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int blk_nnz_start = blk_nnz_displs[blk_id];
    const int blk_nnz_end = blk_nnz_displs[blk_id + 1];
    const int blk_nnz = blk_nnz_end - blk_nnz_start;

    int row = 0, col = 0, input_idx = 0;
    InType in_value = 0;
    if (tid < blk_nnz)
    {
        int nnz_idx = blk_nnz_orig_idx[blk_nnz_start + tid];
        row = rows[nnz_idx];
        col = cols[nnz_idx];
        input_idx = blockIdx.z * nnz + nnz_idx;
        in_value = cderi_sparse[input_idx];
        size_t padded_nao2 = padded_nao * padded_nao;
        size_t output_idx0 = blockIdx.z * padded_nao2 + row * padded_nao + col;
        size_t output_idx1 = blockIdx.z * padded_nao2 + col * padded_nao + row;

        if constexpr(std::is_same_v<SplitType, double>)
        {
            cderi_0[output_idx0] = in_value;
            cderi_0[output_idx1] = in_value;
        }

        if constexpr(std::is_same_v<SplitType, __half> || std::is_same_v<SplitType, __nv_bfloat16>)
        {
            SplitType out0, out1, out2;
            fp_split<InType, SplitType, num_split>(in_value, out0, out1, out2);
            cderi_0[output_idx0] = out0;
            cderi_0[output_idx1] = out0;
            if constexpr(num_split >= 2)
            {
                cderi_1[output_idx0] = out1;
                cderi_1[output_idx1] = out1;
            }
            if constexpr(num_split >= 3)
            {
                cderi_2[output_idx0] = out2;
                cderi_2[output_idx1] = out2;
            }
        }
    }
}


template <typename SplitType, int num_split>
__global__ void sum_rho_K_splits_kernel(
    const int rho_K_size,float* __restrict__ fp32_out_splits_0,
    float* __restrict__ fp32_out_splits_1, float* __restrict__ fp32_out_splits_2,
    SplitType* __restrict__ rho_K_splits_0, SplitType* __restrict__ rho_K_splits_1,
    SplitType* __restrict__ rho_K_splits_2
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rho_K_size) return;
    double upcast_val = 0.0;
    constexpr int exp2_scale_factor = Exp2ScaleFactor<SplitType>::value;
    fp_upcast<double, SplitType, num_split>(
        upcast_val, fp32_out_splits_0[idx], fp32_out_splits_1[idx],
        fp32_out_splits_2[idx], exp2_scale_factor
    );
    fp_split<double, SplitType, num_split>(
        upcast_val, rho_K_splits_0[idx],
        rho_K_splits_1[idx], rho_K_splits_2[idx]
    );
}


template <typename SplitType, int num_split>
__global__ void sum_padded_K_splits_kernel(
    const int nao, const int padded_nao, double *mat_K,
    const double K_mat_acc, float* __restrict__ padded_K_0,
    float* __restrict__ padded_K_1, float* __restrict__ padded_K_2
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nao || col >= nao) return;
    int src_idx = row * padded_nao + col;
    int dst_idx = row * nao + col;
    double K_val = 0.0;
    constexpr int exp2_scale_factor = Exp2ScaleFactor<SplitType>::value;
    fp_upcast<double, float, num_split>(
        K_val, padded_K_0[src_idx], padded_K_1[src_idx],
        padded_K_2[src_idx], exp2_scale_factor
    );
    mat_K[dst_idx] = K_mat_acc * mat_K[dst_idx] + K_val;
}


// =============== Helper functions ===============

#define SPLIT_POINTERS(base_ptr, size) \
    auto *base_ptr##_0 = base_ptr; \
    auto *base_ptr##_1 = num_split > 1 ? base_ptr + size : nullptr; \
    auto *base_ptr##_2 = num_split > 2 ? base_ptr + 2 * size : nullptr;


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
    // Ozaki scheme needs FP16/BF16 GEMM FP32 output
    if constexpr (std::is_same_v<DType, __half>)
    {
        compute_type = CUBLAS_COMPUTE_32F;
        A_dtype = CUDA_R_16F;
        B_dtype = CUDA_R_16F;
        C_dtype = CUDA_R_32F;
        scale_dtype = CUDA_R_32F;
    }
    if constexpr (std::is_same_v<DType, __nv_bfloat16>)
    {
        compute_type = CUBLAS_COMPUTE_32F;
        A_dtype = CUDA_R_16BF;
        B_dtype = CUDA_R_16BF;
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
    cudaStream_t stream, const int naux_blk, const int npair, const int init_output,
    const double *cderi_sparse, const double *D_mat_sparse, double *J_mat_sparse, double *rho_j
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


template<typename SplitType>
void unpack_sym_split_cderi_work(
    cudaStream_t stream, const int npair, const int *rows_d, const int *cols_d,
    const int *blk_nnz_displs_d, const int *blk_nnz_orig_idx_d,
    const int naux_blk, const int padded_nao, const double *cderi_sparse,
    const int num_split, SplitType *cderi_splits, const size_t cderi_split_size
)
{
    SPLIT_POINTERS(cderi_splits, cderi_split_size);
    dim3 block(32, 32), grid;
    grid.x = (padded_nao + block.x - 1) / block.x;
    grid.y = (padded_nao + block.y - 1) / block.y;
    grid.z = naux_blk;
    #define DISPATCH_UNPACK_SYM_SPLIT_CDERI(InType, OutType, num_split) \
    do { \
        unpack_sym_split_cderi_kernel<num_split, InType, OutType><<<grid, block, 0, stream>>>( \
            npair, rows_d, cols_d, \
            blk_nnz_displs_d, blk_nnz_orig_idx_d, padded_nao, \
            static_cast<const InType*>(cderi_sparse), \
            static_cast<OutType*>(cderi_splits_0), \
            static_cast<OutType*>(cderi_splits_1), \
            static_cast<OutType*>(cderi_splits_2) \
        ); \
    } while (0)
    if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, SplitType, 1);
    if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, SplitType, 2);
    if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, SplitType, 3);
    CUDA_CHECK(cudaGetLastError());
    #undef DISPATCH_UNPACK_SYM_SPLIT_CDERI
}

template<typename SplitType>
void DF_K_build_subtask1(
    cudaStream_t stream, const int use_Dmat, const int init_output,
    const int naux_blk, const int padded_nao, const int padded_nocc,
    const SplitType *cderi_i, const SplitType *C_or_D_j, void *rho_K_k,
    const int fp64_emu_bits = 0, const int moduli = 0, void *gemmul8_workspace = nullptr
)
{
    cublasComputeType_t compute_type;
    cudaDataType_t A_dtype, B_dtype, C_dtype, scale_dtype;
    get_cublas_gemm_dtypes<SplitType>(compute_type, A_dtype, B_dtype, C_dtype, scale_dtype);

    // Step 1: einsum Lij,jk -> Lki.
    // For each L, ij,jk -> ki is doing C^T = B^T * A^T. Since the input cderi and C/D are in
    // row-major and cuBLAS takes column-major, use m == the original m, lda = the original k,
    // transA = T, n == the original n, ldb = n, transB = T.
    int m = padded_nao;
    int n = use_Dmat ? padded_nao : padded_nocc;
    int k = padded_nao;
    void *alpha = nullptr, *beta = nullptr;
    if constexpr (std::is_same_v<SplitType, double>)
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
    if (fp64_emu_bits > 0)
    {
        const double *alpha_ = static_cast<const double*>(alpha);
        const double *beta_ = static_cast<const double*>(beta);
        const double *A_ = static_cast<const double*>(A);
        const double *B_ = static_cast<const double*>(B);
        double *C_ = static_cast<double*>(C);
        std::vector<double> time_breakdown(4, 0);
        for (int i = 0; i < batch_count; i++)
        {
            const double *A_i = A_ + i * stride_A;
            const double *B_i = B_ + i * stride_B;
            double *C_i = C_ + i * stride_C;
            time_breakdown = gemmul8::gemm<double>(
                cublas_handle, transA, transB, m, n, k,
                alpha_, A_i, ldA, B_i, ldB, beta_, C_i, ldC,
                moduli, false, gemmul8_workspace
            );
        }
    } else {
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
}

template<typename SplitType>
void DF_K_build_subtask2(
    cudaStream_t stream, const int use_Dmat, const int init_output,
    const int naux_blk, const int padded_nao, const int padded_nocc,
    const SplitType *cderi_or_rho_K_i, const SplitType *rho_K_j,
    void *padded_K_k, const int fp64_emu_bits = 0, const int moduli = 0,
    void *gemmul8_workspace = nullptr
)
{
    cublasComputeType_t compute_type;
    cudaDataType_t A_dtype, B_dtype, C_dtype, scale_dtype;
    get_cublas_gemm_dtypes<SplitType>(compute_type, A_dtype, B_dtype, C_dtype, scale_dtype);

    // Step 2: einsum Lki,Lkj -> ij.
    // This is actually computing K = C^T * C, and since C is in row-major, C^T == C in
    // column-major. K is symmetric, does not matter if it is in row-major or column-major.
    int m = padded_nao;
    int n = padded_nao;
    int k = naux_blk * (use_Dmat ? padded_nao : padded_nocc);
    void *alpha = nullptr, *beta = nullptr;
    if constexpr (std::is_same_v<SplitType, double>)
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

    if (fp64_emu_bits > 0)
    {
        std::vector<double> time_breakdown(4, 0);
        const double *alpha_ = static_cast<const double*>(alpha);
        const double *beta_ = static_cast<const double*>(beta);
        const double *A_ = static_cast<const double *>(A);
        const double *B_ = static_cast<const double *>(B);
        double *C_ = static_cast<double *>(C);
        const int k_blk_size = std::min(k, 16384);
        for (int k_pos = 0; k_pos < k; k_pos += k_blk_size)
        {
            int curr_k = std::min(k_blk_size, k - k_pos);
            const double *curr_A = A_ + k_pos * ldA;
            const double *curr_B = B_ + k_pos * ldB;
            time_breakdown = gemmul8::gemm<double>(
                cublas_handle, transA, transB, m, n, curr_k,
                alpha_, curr_A, ldA, curr_B, ldB, beta_, C_, ldC,
                moduli, false, gemmul8_workspace
            );
            beta_ = &f64_one;
        }
        CUDA_CHECK( cudaGetLastError() );
    } else {
        cublasLtMatmulDesc_t matmul_desc = nullptr;
        cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
        cublasLtMatmulPreference_t preference = NULL;
        cublasLtMatmulHeuristicResult_t heur_result = {};
        int return_result = 0;
        CUBLAS_CHECK( cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_dtype) );
        CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)) );
        CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)) );
        CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Adesc, A_dtype, m, k, ldA) );
        CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Bdesc, B_dtype, n, k, ldB) );
        CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Cdesc, C_dtype, m, n, ldC));
        CUBLAS_CHECK( cublasLtMatmulPreferenceCreate(&preference) );
        CUBLAS_CHECK( cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas_workspace_bytes, sizeof(cublas_workspace_bytes)
        ) );
        CUBLAS_CHECK( cublasLtMatmulAlgoGetHeuristic(
            lt_handle, matmul_desc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heur_result, &return_result
        ) );
        if (return_result == 0) CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
        CUBLAS_CHECK( cublasLtMatmul(
            lt_handle, matmul_desc, alpha, A, Adesc, B, Bdesc,
            beta, C, Cdesc, C, Cdesc, &heur_result.algo,
            cublas_workspace, cublas_workspace_bytes, stream
        ) );
    }
}


int get_gemmul8_moduli(const int fp64_emu_bits)
{
    // TODO: need to tune
    if (fp64_emu_bits == 23) return 9;
    else if (fp64_emu_bits == 31) return 11;
    else if (fp64_emu_bits == 39) return 13;
    else return 18;
}


void DF_K_build_fp64_work(
    cudaStream_t stream, const int use_Dmat, const int naux_blk,
    const int nao, const int padded_nao, const int padded_nocc,
    const double *padded_cderi, const double *padded_C_or_D_mat,
    double *rho_K_buf, double *padded_K_buf, double *mat_K,
    const int K_mat_acc, const int fp64_emu_bits, void *gemmul8_workspace
)
{
    int moduli = get_gemmul8_moduli(fp64_emu_bits);

    int init_output = 1;
    DF_K_build_subtask1<double>(
        stream, use_Dmat, init_output, naux_blk, padded_nao, padded_nocc,
        padded_cderi, padded_C_or_D_mat, rho_K_buf, fp64_emu_bits,
        moduli, gemmul8_workspace
    );

    const double *padded_cderi_or_rho_K = use_Dmat ? padded_cderi : rho_K_buf;
    DF_K_build_subtask2<double>(
        stream, use_Dmat, init_output, naux_blk, padded_nao, padded_nocc,
        padded_cderi_or_rho_K, rho_K_buf, padded_K_buf, fp64_emu_bits,
        moduli, gemmul8_workspace
    );

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
    cudaStream_t stream, const int use_Dmat, const int naux_blk,
    const int nao, const int padded_nao, const int padded_nocc,
    const float *padded_cderi, const float *padded_C_or_D_mat,
    float *rho_K_buf, float *padded_K_buf, double *mat_K, const int K_mat_acc
)
{
    constexpr int num_split = 1;
    int init_output = 1;
    DF_K_build_subtask1<float>(
        stream, use_Dmat, init_output, naux_blk, padded_nao, padded_nocc,
        padded_cderi, padded_C_or_D_mat, rho_K_buf
    );

    const float *padded_cderi_or_rho_K = use_Dmat ? padded_cderi : rho_K_buf;
    DF_K_build_subtask2<float>(
        stream, use_Dmat, init_output, naux_blk, padded_nao, padded_nocc,
        padded_cderi_or_rho_K, rho_K_buf, padded_K_buf
    );

    dim3 block(32, 32);
    dim3 grid((padded_nao + block.x - 1) / block.x, (padded_nao + block.y - 1) / block.y);
    float K_mat_acc_ = K_mat_acc ? 1.0f : 0.0f;
    SPLIT_POINTERS(padded_K_buf, 0);
    sum_padded_K_splits_kernel<float, 1><<<grid, block, 0, stream>>>(
        nao, padded_nao, mat_K, K_mat_acc_,
        padded_K_buf_0, padded_K_buf_1, padded_K_buf_2
    );
    CUDA_CHECK( cudaGetLastError() );
}


template <typename SplitType, int num_split>
void DF_K_build_fp_split_work(
    cudaStream_t stream, const int use_Dmat, const int naux_blk,
    const int nao, const int padded_nao, const int padded_nocc,
    SplitType *cderi_splits, const size_t cderi_split_size,
    SplitType *C_or_D_splits, const size_t C_or_D_split_size,
    SplitType *rho_K_splits, const size_t rho_K_size,
    SplitType *padded_K_splits, const size_t padded_K_size,
    float *fp32_out_splits, double *mat_K, const int K_mat_acc
)
{
    for (size_t i = 0; i < num_split; i++)
    {
        int init_output = (i == 0);
        SplitType *cderi_i = cderi_splits + i * cderi_split_size;
        for (size_t j = 0; j < num_split - i; j++)
        {
            SplitType *C_or_D_j = C_or_D_splits + j * C_or_D_split_size;
            const size_t rho_K_k_offset = (i + j) * rho_K_size;
            void *rho_K_k = static_cast<void *>(fp32_out_splits + rho_K_k_offset);
            DF_K_build_subtask1<SplitType>(
                stream, use_Dmat, init_output,
                naux_blk, padded_nao, padded_nocc,
                cderi_i, C_or_D_j, rho_K_k
            );
        }
    }

    dim3 block(1024);
    dim3 grid((rho_K_size + block.x - 1) / block.x);
    SPLIT_POINTERS(fp32_out_splits, rho_K_size);
    SPLIT_POINTERS(rho_K_splits, rho_K_size);
    sum_rho_K_splits_kernel<SplitType, num_split><<<grid, block, 0, stream>>>(
        rho_K_size, fp32_out_splits_0, fp32_out_splits_1, fp32_out_splits_2,
        rho_K_splits_0, rho_K_splits_1, rho_K_splits_2
    );

    for (size_t i = 0; i < num_split; i++)
    {
        SplitType *cderi_i = cderi_splits + i * cderi_split_size;
        SplitType *rho_K_i = rho_K_splits + i * rho_K_size;
        SplitType *cderi_or_rho_K_i = use_Dmat ? cderi_i : rho_K_i;
        int init_output = (i == 0);
        for (size_t j = 0; j < num_split - i; j++)
        {
            SplitType *rho_K_j = rho_K_splits + j * rho_K_size;
            const size_t padded_K_k_offset = (i + j) * padded_K_size;
            void *padded_K_k = static_cast<void *>(fp32_out_splits + padded_K_k_offset);
            DF_K_build_subtask2<SplitType>(
                stream, use_Dmat, init_output,
                naux_blk, padded_nao, padded_nocc,
                cderi_or_rho_K_i, rho_K_j, padded_K_k
            );
        }
    }

    block.x = 32;
    block.y = 32;
    grid.x = (padded_nao + block.x - 1) / block.x;
    grid.y = (padded_nao + block.y - 1) / block.y;
    double K_mat_acc_ = K_mat_acc ? 1.0 : 0.0;
    sum_padded_K_splits_kernel<SplitType, num_split><<<grid, block, 0, stream>>>(
        nao, padded_nao, mat_K, K_mat_acc_,
        fp32_out_splits_0, fp32_out_splits_1, fp32_out_splits_2
    );
    CUDA_CHECK( cudaGetLastError() );
}


template<typename SplitType>
void DF_JK_build_work(
    cudaStream_t stream, const int naux, const int nao, const int nocc,
    const int num_split, const int fp64_emu_bits, const int use_Dmat,
    const int npair, const int *rows, const int *cols,
    const double *cderi_sparse, const double *C_or_D_mat, const double *D_mat_sparse,
    const int build_J, const int build_K, double *J_mat_sparse, double *K_mat,
    const int max_naux_blk, const size_t gemmul8_workspace_bytes, void *workbuf
)
{
    CUBLAS_CHECK( cublasCreate(&cublas_handle) );
    CUBLAS_CHECK( cublasLtCreate(&lt_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, stream) );

    uint8_t *workbuf_ptr = static_cast<uint8_t *>(workbuf);
    workbuf_ptr = move_ptr_to_next_256B_aligned(workbuf_ptr);
    CUBLAS_CHECK( cublasSetWorkspace(cublas_handle, workbuf_ptr, cublas_workspace_bytes) );
    cublas_workspace = workbuf_ptr;
    workbuf_ptr += cublas_workspace_bytes;

    uint8_t *gemmul8_workspace = workbuf_ptr;
    workbuf_ptr += gemmul8_workspace_bytes;

    if (build_J)
    {
        DF_J_build_work(
            stream, naux, npair, 1, cderi_sparse, D_mat_sparse,
            J_mat_sparse, reinterpret_cast<double *>(workbuf_ptr)
        );
    }

    if (!build_K) return;
    int padded_nao = pad_size_for_gemm(nao);
    int padded_nocc = pad_size_for_gemm(nocc);
    int C_or_D_ncol = use_Dmat ? nao : nocc;
    int padded_C_or_D_ncol = use_Dmat ? padded_nao : padded_nocc;
    size_t padded_C_or_D_size = padded_nao * padded_C_or_D_ncol;
    size_t padded_cderi_size = max_naux_blk * padded_nao * padded_nao;
    size_t rho_K_size = max_naux_blk * padded_nao * padded_C_or_D_ncol;
    size_t padded_K_size = padded_nao * padded_nao;
    SplitType *C_or_D_splits = reinterpret_cast<SplitType *>(workbuf_ptr);
    SplitType *cderi_splits = C_or_D_splits + num_split * padded_C_or_D_size;
    SplitType *rho_K_splits = cderi_splits + num_split * padded_cderi_size;
    SplitType *padded_K_splits = rho_K_splits + num_split * rho_K_size;
    float *fp32_out_splits = reinterpret_cast<float *>(padded_K_splits + num_split * padded_K_size);

    SPLIT_POINTERS(C_or_D_splits, padded_C_or_D_size);
    dim3 block(32, 32);
    dim3 grid((padded_nao + block.x - 1) / block.x, (padded_C_or_D_ncol + block.y - 1) / block.y);
    #define DISPATCH_SPLIT_FP_ARRAY(num_split) \
    do { \
        split_C_or_D_mat_kernel<num_split, double, SplitType><<<grid, block, 0, stream>>>( \
            nao, padded_nao, C_or_D_ncol, padded_C_or_D_ncol, \
            C_or_D_mat, C_or_D_splits_0, C_or_D_splits_1, C_or_D_splits_2 \
        ); \
    } while (0)
    if (num_split == 1) DISPATCH_SPLIT_FP_ARRAY(1);
    if (num_split == 2) DISPATCH_SPLIT_FP_ARRAY(2);
    if (num_split == 3) DISPATCH_SPLIT_FP_ARRAY(3);
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

    for (int i = 0; i < naux; i += max_naux_blk)
    {
        int naux_blk = (i + max_naux_blk < naux) ? max_naux_blk : (naux - i);
        size_t cderi_sparse_offset = static_cast<size_t>(i) * static_cast<size_t>(npair);
        const double *cderi_sparse_blk = cderi_sparse + cderi_sparse_offset;
        unpack_sym_split_cderi_work<SplitType>(
            stream, npair, rows_d, cols_d,
            blk_nnz_displs_d, blk_nnz_orig_idx_d,
            naux_blk, padded_nao, cderi_sparse_blk, num_split,
            cderi_splits, padded_cderi_size
        );

        int K_mat_acc = (i > 0);
        if constexpr(std::is_same_v<SplitType, double>)
        {
            assert(num_split == 1);
            DF_K_build_fp64_work(
                stream, use_Dmat, naux_blk, nao, padded_nao, padded_nocc,
                cderi_splits, C_or_D_splits, rho_K_splits, padded_K_splits,
                K_mat, K_mat_acc, fp64_emu_bits, gemmul8_workspace
            );
        }
        if constexpr(std::is_same_v<SplitType, float>)
        {
            assert(num_split == 1);
            DF_K_build_fp32_work(
                stream, use_Dmat, naux_blk, nao, padded_nao, padded_nocc,
                cderi_splits, C_or_D_splits, rho_K_splits, padded_K_splits,
                K_mat, K_mat_acc
            );
        }
        #define DISPATCH_DF_K_BUILD_FP_SPLIT_WORK(num_split_) \
        do {  \
            DF_K_build_fp_split_work<SplitType, num_split_>(  \
                stream, use_Dmat, naux_blk, nao, padded_nao, padded_nocc,  \
                cderi_splits, padded_cderi_size,  \
                C_or_D_splits, padded_C_or_D_size,  \
                rho_K_splits, rho_K_size,  \
                padded_K_splits, padded_K_size,  \
                fp32_out_splits, K_mat, K_mat_acc  \
            );  \
        } while (0)
        if constexpr(std::is_same_v<SplitType, __half> || std::is_same_v<SplitType, __nv_bfloat16>)
        {
            if (num_split == 2) DISPATCH_DF_K_BUILD_FP_SPLIT_WORK(2);
            if (num_split == 3) DISPATCH_DF_K_BUILD_FP_SPLIT_WORK(3);
        }
        #undef DISPATCH_DF_K_BUILD_FP_SPLIT_WORK
    }

    free(blk_nnz_displs);
    free(blk_nnz_orig_idx);
    CUDA_CHECK( cudaFree(rows_d) );
    CUDA_CHECK( cudaFree(cols_d) );
    CUDA_CHECK( cudaFree(blk_nnz_displs_d) );
    CUDA_CHECK( cudaFree(blk_nnz_orig_idx_d) );

    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    CUBLAS_CHECK( cublasLtDestroy(lt_handle) );
}

// =============== C interface for Python ctypes ===============

extern "C" {

size_t get_gemmul8_workspace_bytes(
    const int naux, const int nao, const int nocc,
    const int use_Dmat, const int fp64_emu_bits
)
{
    int padded_nao = pad_size_for_gemm(nao);
    int padded_nocc = pad_size_for_gemm(nocc);
    int padded_C_or_D_ncol = use_Dmat ? padded_nao : padded_nocc;
    int gemmul8_moduli = get_gemmul8_moduli(fp64_emu_bits);

    int m1 = padded_nao, n1 = padded_C_or_D_ncol, k1 = padded_nao;
    size_t bytes1 = gemmul8::workSize(m1, n1, k1, gemmul8_moduli);

    int m2 = padded_nao, n2 = padded_nao, k2 = naux * padded_C_or_D_ncol;
    k2 = std::min(k2, 16384);  // Limit k2 to avoid excessive memory usage
    size_t bytes2 = gemmul8::workSize(m2, n2, k2, gemmul8_moduli);

    size_t bytes = std::max(bytes1, bytes2) + 256;
    return bytes;
}

int DF_JK_build(
    cudaStream_t stream, const int naux, const int nao, const int nocc,
    const int num_split, const int split_dtype_id, const int fp64_emu_bits,
    const int use_Dmat, const int npair, const int *rows, const int *cols,
    const void *cderi_sparse, const void *C_or_D_mat, const void *D_mat_sparse,
    const int build_J, const int build_K, void *J_mat_sparse, void *K_mat,
    const int max_naux_blk, const size_t gemmul8_workspace_bytes, void *workbuf
)
{
    ERROR_CHECK(num_split >= 1 && num_split <= 3, "num_split must be between 1 and 3.");

    const double *cderi_sparse_d = static_cast<const double *>(cderi_sparse);
    const double *C_or_D_mat_d = static_cast<const double *>(C_or_D_mat);
    const double *D_mat_sparse_d = static_cast<const double *>(D_mat_sparse);
    double *J_mat_sparse_d = static_cast<double *>(J_mat_sparse);
    double *K_mat_d = static_cast<double *>(K_mat);
    #define DISPATCH_DF_JK_BUILD_WORK(SplitType) \
    do { \
        DF_JK_build_work<SplitType>( \
            stream, naux, nao, nocc, num_split, fp64_emu_bits, \
            use_Dmat, npair, rows, cols, cderi_sparse_d, \
            C_or_D_mat_d, D_mat_sparse_d, build_J, build_K, \
            J_mat_sparse_d, K_mat_d, max_naux_blk, \
            gemmul8_workspace_bytes, workbuf \
        ); \
    } while (0)
    if (split_dtype_id == 0) DISPATCH_DF_JK_BUILD_WORK(double);
    else if (split_dtype_id == 1) DISPATCH_DF_JK_BUILD_WORK(float);
    else if (split_dtype_id == 2) DISPATCH_DF_JK_BUILD_WORK(__half);
    else if (split_dtype_id == 3) DISPATCH_DF_JK_BUILD_WORK(__nv_bfloat16);
    else
    {
        fprintf(stderr, "[ERROR][%s:%d] Unsupported split_dtype_id (%d).\n",
                __FUNCTION__, __LINE__, split_dtype_id);
        return 1;
    }
    #undef DISPATCH_DF_JK_BUILD_WORK

    CUDA_CHECK( cudaStreamSynchronize(stream) );
    CUDA_CHECK( cudaGetLastError() );
    return 0;
}

}  // End of "extern C"
