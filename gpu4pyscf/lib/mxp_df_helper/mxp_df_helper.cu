#include <stdio.h>
#include <assert.h>
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


static int cderi_npair = 0;
static int *cderi_rows_h = nullptr;
static int *cderi_cols_h = nullptr;
static int *cderi_rows_d = nullptr;
static int *cderi_cols_d = nullptr;

static double f64_one = 1, f64_zero = 0;
static float  f32_one = 1, f32_zero = 0;

static cublasHandle_t cublas_handle = nullptr;
static cublasLtHandle_t lt_handle = nullptr;
static void *cublas_workspace = nullptr;
static const size_t cublas_workspace_bytes = 32 * 1024 * 1024;

#define GEMM_ALIGNMENT 32  // Tensor Core kernels prefer this alignment

// =============== Kernel functions ===============

template<int num_split, typename IType, typename SplitType>
__global__ void split_C_or_D_mat_kernel(
    const int nrow, const int padded_nrow, const int ncol, const int padded_ncol,
    const IType* __restrict__ input,
    SplitType* __restrict__ output_0, SplitType* __restrict__ output_1,
    SplitType* __restrict__ output_2, SplitType* __restrict__ output_3
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int input_idx = row * ncol + col;
    const int output_idx = row * padded_ncol + col;
    IType in_value = (row < nrow && col < ncol) ?
        input[input_idx] : static_cast<IType>(0);
    SplitType out_value0 = static_cast<SplitType>(in_value);
    output_0[output_idx] = out_value0;
    in_value -= static_cast<IType>(out_value0);
    if constexpr(num_split == 1) return;
    SplitType out_value1 = static_cast<SplitType>(in_value);
    output_1[output_idx] = out_value1;
    in_value -= static_cast<IType>(out_value1);
    if constexpr(num_split == 2) return;
    SplitType out_value2 = static_cast<SplitType>(in_value);
    output_2[output_idx] = out_value2;
    in_value -= static_cast<IType>(out_value2);
    if constexpr(num_split == 3) return;
    SplitType out_value3 = static_cast<SplitType>(in_value);
    output_3[output_idx] = out_value3;
}

template<int num_split, typename IType, typename SplitType>
__global__ void unpack_sym_split_cderi_kernel(
    const int npair, const int* __restrict__ rows, const int* __restrict__ cols,
    const int nrow, const IType* __restrict__ cderi_sparse,
    SplitType* __restrict__ cderi_0, SplitType* __restrict__ cderi_1,
    SplitType* __restrict__ cderi_2, SplitType* __restrict__ cderi_3
)
{
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int bid = blockIdx.x;
    const int thread_block_size = blockDim.x * blockDim.y;
    const int mat_size = nrow * nrow;
    for (int pair_id = tid; pair_id < npair; pair_id += thread_block_size)
    {
        if (pair_id > npair) continue;
        int row = rows[pair_id];
        int col = cols[pair_id];
        size_t input_idx = bid * npair + pair_id;
        size_t output_idx0 = bid * mat_size + row * nrow + col;
        size_t output_idx1 = bid * mat_size + col * nrow + row;
        IType in_value = cderi_sparse[input_idx];
        SplitType out_value0 = static_cast<SplitType>(in_value);
        cderi_0[output_idx0] = out_value0;
        cderi_0[output_idx1] = out_value0;
        if constexpr(num_split == 1) continue;
        in_value -= static_cast<IType>(out_value0);
        SplitType out_value1 = static_cast<SplitType>(in_value);
        cderi_1[output_idx0] = out_value1;
        cderi_1[output_idx1] = out_value1;
        if constexpr(num_split == 2) continue;
        in_value -= static_cast<IType>(out_value1);
        SplitType out_value2 = static_cast<SplitType>(in_value);
        cderi_2[output_idx0] = out_value2;
        cderi_2[output_idx1] = out_value2;
        if constexpr(num_split == 3) continue;
        in_value -= static_cast<IType>(out_value2);
        SplitType out_value3 = static_cast<SplitType>(in_value);
        cderi_3[output_idx0] = out_value3;
        cderi_3[output_idx1] = out_value3;
    }
}


template<int num_split, typename SplitType, typename OutType>
__global__ void sum_rho_K_splits_kernel(
    const int nao, const int padded_nao, OutType *mat_K,
    SplitType* __restrict__ padded_K_0, SplitType* __restrict__ padded_K_1,
    SplitType* __restrict__ padded_K_2, SplitType* __restrict__ padded_K_3
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < nao && col < nao)
    {
        int src_idx = row * padded_nao + col;
        int dst_idx = row * nao + col;
        OutType K_val = mat_K[dst_idx];
        K_val += static_cast<OutType>(padded_K_0[src_idx]);
        if constexpr(num_split >= 2) K_val += static_cast<OutType>(padded_K_1[src_idx]);
        if constexpr(num_split >= 3) K_val += static_cast<OutType>(padded_K_2[src_idx]);
        if constexpr(num_split >= 4) K_val += static_cast<OutType>(padded_K_3[src_idx]);
        mat_K[dst_idx] = K_val;
    }
}


// =============== Helper functions ===============

#define DF_K_BUILD_SUBTASK_COMMON \
    cublasComputeType_t compute_type; \
    cudaDataType_t AB_type, C_type, scale_type; \
    get_cublas_gemm_dtypes<SplitType>(compute_type, AB_type, C_type, scale_type);


#define SPLIT_POINTERS(base_ptr, size) \
    SplitType *base_ptr##_0 = base_ptr; \
    SplitType *base_ptr##_1 = num_split > 1 ? base_ptr + size : nullptr; \
    SplitType *base_ptr##_2 = num_split > 2 ? base_ptr + size * 2 : nullptr; \
    SplitType *base_ptr##_3 = num_split > 3 ? base_ptr + size * 3 : nullptr;


template<typename T>
static T pad_size_for_gemm(const T n, const T alignment = GEMM_ALIGNMENT)
{
    assert(std::is_integral<T>::value);
    return ((n + alignment - 1) / alignment) * alignment;
}


template<typename DType>
void get_cublas_gemm_dtypes(
    cublasComputeType_t &compute_type, cudaDataType_t &AB_type,
    cudaDataType_t &C_type, cudaDataType_t &scale_type
)
{
    if constexpr (std::is_same_v<DType, double>)
    {
        compute_type = CUBLAS_COMPUTE_64F;
        AB_type = CUDA_R_64F;
        C_type = CUDA_R_64F;
        scale_type = CUDA_R_64F;
    }
    if constexpr (std::is_same_v<DType, float>)
    {
        compute_type = CUBLAS_COMPUTE_32F;
        AB_type = CUDA_R_32F;
        C_type = CUDA_R_32F;
        scale_type = CUDA_R_32F;
    }
    if constexpr (std::is_same_v<DType, __half>)
    {
        // Use higher precision for intermediate results and accmulation
        compute_type = CUBLAS_COMPUTE_32F;
        AB_type = CUDA_R_16F;
        C_type = CUDA_R_16F;
        scale_type = CUDA_R_32F;
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
    const double *cderi_sparse, const double *D_mat_sparse, double *J_mat_sparse,
    double *rho_j
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


template<typename SplitType>
void unpack_sym_split_cderi_work(
    cudaStream_t stream, const int npair, const int *rows, const int *cols,
    const int naux_blk, const int nrow, const double *cderi_sparse,
    const int num_split, SplitType *cderi_splits, const size_t cderi_split_size
)
{
    ERROR_CHECK(num_split >= 1 && num_split <= 4, "num_split must be between 1 and 4.");

    if (npair != cderi_npair)
    {
        free(cderi_rows_h);
        free(cderi_cols_h);
        CUDA_CHECK( cudaFree(cderi_rows_d) );
        CUDA_CHECK( cudaFree(cderi_cols_d) );
        cderi_npair = npair;
        size_t rc_bytes = sizeof(int) * npair;
        cderi_rows_h = (int *) malloc(rc_bytes);
        cderi_cols_h = (int *) malloc(rc_bytes);
        CUDA_CHECK( cudaMalloc((void**) &cderi_rows_d, rc_bytes) );
        CUDA_CHECK( cudaMalloc((void**) &cderi_cols_d, rc_bytes) );
        memcpy(cderi_rows_h, rows, rc_bytes);
        memcpy(cderi_cols_h, cols, rc_bytes);
        CUDA_CHECK( cudaMemcpyAsync(cderi_rows_d, rows, rc_bytes, cudaMemcpyHostToDevice, stream) );
        CUDA_CHECK( cudaMemcpyAsync(cderi_cols_d, cols, rc_bytes, cudaMemcpyHostToDevice, stream) );
    }

    SPLIT_POINTERS(cderi_splits, cderi_split_size);
    dim3 block(32, 32), grid(naux_blk);
    #define DISPATCH_UNPACK_SYM_SPLIT_CDERI(itype, OutType, num_split) \
    do { \
        unpack_sym_split_cderi_kernel<num_split, itype, OutType><<<grid, block, 0, stream>>>( \
            npair, cderi_rows_d, cderi_cols_d, nrow, \
            static_cast<const itype*>(cderi_sparse), \
            static_cast<OutType*>(cderi_splits_0), \
            static_cast<OutType*>(cderi_splits_1), \
            static_cast<OutType*>(cderi_splits_2), \
            static_cast<OutType*>(cderi_splits_3) \
        ); \
    } while (0)
    if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, SplitType, 1);
    if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, SplitType, 2);
    if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, SplitType, 3);
    if (num_split == 4) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, SplitType, 4);
    CUDA_CHECK(cudaGetLastError());
    #undef DISPATCH_UNPACK_SYM_SPLIT_CDERI
}


template<int num_split, typename SplitType>
void DF_K_build_subtask1(
    cudaStream_t stream, const int use_Dmat, const int init_output,
    const int i, const int j,
    const int naux_blk, const int padded_nao, const int padded_nocc,
    SplitType *cderi_i, SplitType *C_or_D_j, SplitType *rho_K_k
)
{
    DF_K_BUILD_SUBTASK_COMMON;

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
    void *C = static_cast<void*>(rho_K_k);
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_T;
    int lda = k;
    int ldb = n;
    int ldc = m;
    long long int stride_A = static_cast<long long int>(m * k);
    long long int stride_B = 0;
    long long int stride_C = static_cast<long long int>(m * n);
    int batch_count = naux_blk;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    CUBLAS_CHECK( cublasGemmStridedBatchedEx(
        cublas_handle, transA, transB, m, n, k,
        alpha,
        A, AB_type, lda, stride_A,
        B, AB_type, ldb, stride_B,
        beta,
        C, C_type, ldc, stride_C,
        batch_count, compute_type, algo
    ) );
}


template<int num_split, typename SplitType>
void DF_K_build_subtask2(
    cudaStream_t stream, const int use_Dmat, const int init_output,
    const int i, const int j,
    const int naux_blk, const int padded_nao, const int padded_nocc,
    SplitType *cderi_or_rho_K_i, SplitType *rho_K_j, SplitType *padded_K_k
)
{
    DF_K_BUILD_SUBTASK_COMMON;

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
    void *C = static_cast<void*>(padded_K_k);
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    int lda = m;
    int ldb = n;
    int ldc = m;
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heur_result = {};
    int return_result = 0;
    CUBLAS_CHECK( cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type) );
    CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)) );
    CUBLAS_CHECK( cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)) );
    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Adesc, AB_type, m, k, lda) );
    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Bdesc, AB_type, n, k, ldb) );
    CUBLAS_CHECK( cublasLtMatrixLayoutCreate(&Cdesc, C_type, m, n, ldc));
    CUBLAS_CHECK( cublasLtMatmulPreferenceCreate(&preference) );
    CUBLAS_CHECK( cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas_workspace_bytes, sizeof(cublas_workspace_bytes)
    ) );
    CUBLAS_CHECK( cublasLtMatmulAlgoGetHeuristic(
        lt_handle, matmul_desc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heur_result, &return_result
    ) );
    if (return_result == 0) CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);

    #if 1
    CUBLAS_CHECK( cublasLtMatmul(
        lt_handle, matmul_desc, alpha, A, Adesc, B, Bdesc,
        beta, C, Cdesc, C, Cdesc, &heur_result.algo,
        cublas_workspace, cublas_workspace_bytes, stream
    ) );
    #else
    CUBLAS_CHECK( cublasGemmEx(
        cublas_handle, transA, transB, m, n, k,
        alpha, A, AB_type, lda, B, AB_type, ldb,
        beta, C, C_type, ldc,
        compute_type, CUBLAS_GEMM_DEFAULT
    ) );
    #endif
}


template<int num_split, typename OutType, typename SplitType>
void DF_K_build_work(
    cudaStream_t stream, const int use_Dmat, const int naux_blk,
    const int nao, const int padded_nao, const int padded_nocc,
    SplitType *cderi_splits, const size_t cderi_split_size,
    SplitType *C_or_D_splits, const size_t C_or_D_split_size,
    SplitType *rho_K_splits, const size_t rho_K_size,
    SplitType *padded_K_splits, const size_t padded_K_size,
    OutType *mat_K
)
{
    for (int i = 0; i < num_split; i++)
    {
        int init_output = (i == 0);
        SplitType *cderi_i = cderi_splits + i * cderi_split_size;
        for (int j = 0; j < num_split - i; j++)
        {
            const int k = i + j;
            SplitType *C_or_D_j = C_or_D_splits + j * C_or_D_split_size;
            SplitType *rho_K_k = rho_K_splits + k * rho_K_size;
            DF_K_build_subtask1<num_split, SplitType>(
                stream, use_Dmat, init_output, i, j,
                naux_blk, padded_nao, padded_nocc,
                cderi_i, C_or_D_j, rho_K_k
            );
        }
    }

    for (int i = 0; i < num_split; i++)
    {
        SplitType *cderi_i = cderi_splits + i * cderi_split_size;
        SplitType *rho_K_i = rho_K_splits + i * rho_K_size;
        SplitType *cderi_or_rko_K_i = use_Dmat ? cderi_i : rho_K_i;
        int init_output = (i == 0);
        for (int j = 0; j < num_split - i; j++)
        {
            const int k = i + j;
            SplitType *rho_K_j = rho_K_splits + j * rho_K_size;
            SplitType *padded_K_k = padded_K_splits + k * padded_K_size;
            DF_K_build_subtask2<num_split, SplitType>(
                stream, use_Dmat, init_output, i, j,
                naux_blk, padded_nao, padded_nocc,
                cderi_or_rko_K_i, rho_K_j, padded_K_k
            );
        }
    }

    SPLIT_POINTERS(padded_K_splits, padded_K_size);
    dim3 block(32, 32);
    dim3 grid((nao + block.x - 1) / block.x, (nao + block.y - 1) / block.y);
    sum_rho_K_splits_kernel<num_split, SplitType><<<grid, block, 0, stream>>>(
        nao, padded_nao, mat_K, padded_K_splits_0, padded_K_splits_1,
        padded_K_splits_2, padded_K_splits_3
    );
    CUDA_CHECK( cudaGetLastError() );
}


template<typename SplitType>
void DF_JK_build_work(
    cudaStream_t stream, const int naux, const int nao, const int nocc, const int num_split,
    const int use_Dmat, const int npair, const int *rows, const int *cols,
    const double *cderi_sparse, const double *C_or_D_mat, const double *D_mat_sparse,
    const int build_J, const int build_K, double *J_mat_sparse, double *K_mat,
    const int max_naux_blk, void *workbuf
)
{
    if (cublas_handle == nullptr)
    {
        CUBLAS_CHECK( cublasCreate(&cublas_handle) );
        CUBLAS_CHECK( cublasLtCreate(&lt_handle) );
    }
    CUBLAS_CHECK( cublasSetStream(cublas_handle, stream) );

    uint8_t *workbuf_ptr = static_cast<uint8_t *>(workbuf);
    workbuf_ptr = move_ptr_to_next_256B_aligned(workbuf_ptr);
    cublas_workspace = workbuf_ptr;
    workbuf_ptr += cublas_workspace_bytes;

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
    size_t padded_K_size = max_naux_blk * padded_nao * padded_nao;
    SplitType *C_or_D_splits = reinterpret_cast<SplitType *>(workbuf_ptr);
    SplitType *cderi_splits = C_or_D_splits + num_split * padded_C_or_D_size;
    SplitType *rho_K_splits = cderi_splits + num_split * padded_cderi_size;
    SplitType *padded_K_splits = rho_K_splits + num_split * rho_K_size;

    SPLIT_POINTERS(C_or_D_splits, padded_C_or_D_size);
    dim3 block(32, 32);
    dim3 grid((padded_nao + block.x - 1) / block.x, (padded_C_or_D_ncol + block.y - 1) / block.y);
    #define DISPATCH_SPLIT_FP_ARRAY(num_split) \
    do { \
        split_C_or_D_mat_kernel<num_split, double, SplitType><<<grid, block, 0, stream>>>( \
            nao, padded_nao, C_or_D_ncol, padded_C_or_D_ncol, C_or_D_mat, \
            C_or_D_splits_0, C_or_D_splits_1, C_or_D_splits_2, C_or_D_splits_3 \
        ); \
    } while (0)
    if (num_split == 1) DISPATCH_SPLIT_FP_ARRAY(1);
    if (num_split == 2) DISPATCH_SPLIT_FP_ARRAY(2);
    if (num_split == 3) DISPATCH_SPLIT_FP_ARRAY(3);
    if (num_split == 4) DISPATCH_SPLIT_FP_ARRAY(4);
    CUDA_CHECK( cudaGetLastError() );
    #undef DISPATCH_SPLIT_FP_ARRAY

    #define DISPATCH_UNPACK_SYM_SPLIT_CDERI_WORK(SplitType) \
    do { \
        unpack_sym_split_cderi_work<SplitType>( \
            stream, npair, rows, cols, naux_blk, padded_nao, \
            cderi_sparse_blk, num_split, \
            cderi_splits, padded_cderi_size \
        ); \
    } while (0)
    #define DISPATCH_DF_K_BUILD_WORK(num_split, OutType, SplitType) \
    do { \
        DF_K_build_work<num_split, OutType, SplitType>( \
            stream, use_Dmat, naux_blk, nao, padded_nao, padded_nocc, \
            cderi_splits, padded_cderi_size, \
            C_or_D_splits, padded_C_or_D_size, \
            rho_K_splits, rho_K_size, \
            padded_K_splits, padded_K_size, \
            K_mat \
        ); \
    } while (0)
    for (int i = 0; i < naux; i += max_naux_blk)
    {
        int naux_blk = (i + max_naux_blk < naux) ? max_naux_blk : (naux - i);
        size_t cderi_sparse_offset = static_cast<size_t>(i) * static_cast<size_t>(npair);
        const double *cderi_sparse_blk = cderi_sparse + cderi_sparse_offset;
        DISPATCH_UNPACK_SYM_SPLIT_CDERI_WORK(SplitType);
        if (num_split == 1) DISPATCH_DF_K_BUILD_WORK(1, double, SplitType);
        if (num_split == 2) DISPATCH_DF_K_BUILD_WORK(2, double, SplitType);
        if (num_split == 3) DISPATCH_DF_K_BUILD_WORK(3, double, SplitType);
        if (num_split == 4) DISPATCH_DF_K_BUILD_WORK(4, double, SplitType);
    }
    #undef DISPATCH_UNPACK_SYM_SPLIT_CDERI_WORK
    #undef DISPATCH_DF_K_BUILD_WORK
}

extern "C" {

int DF_JK_build(
    cudaStream_t stream, const int naux, const int nao, const int nocc,
    const int num_split, const int split_dtype_bytes, const int use_Dmat,
    const int npair, const int *rows, const int *cols,
    const void *cderi_sparse, const void *C_or_D_mat, const void *D_mat_sparse,
    const int build_J, const int build_K, void *J_mat_sparse, void *K_mat,
    const int max_naux_blk, void *workbuf
)
{
    ERROR_CHECK(num_split >= 1 && num_split <= 4, "num_split must be between 1 and 4.");

    const double *cderi_sparse_d = static_cast<const double *>(cderi_sparse);
    const double *C_or_D_mat_d = static_cast<const double *>(C_or_D_mat);
    const double *D_mat_sparse_d = static_cast<const double *>(D_mat_sparse);
    double *J_mat_sparse_d = static_cast<double *>(J_mat_sparse);
    double *K_mat_d = static_cast<double *>(K_mat);
    #define DISPATCH_DF_JK_BUILD_WORK(SplitType) \
    do { \
        DF_JK_build_work<SplitType>( \
            stream, naux, nao, nocc, num_split, use_Dmat, \
            npair, rows, cols, cderi_sparse_d, \
            C_or_D_mat_d, D_mat_sparse_d, build_J, build_K, \
            J_mat_sparse_d, K_mat_d, max_naux_blk, workbuf \
        ); \
    } while (0)
    if (split_dtype_bytes == 8) DISPATCH_DF_JK_BUILD_WORK(double);
    else if (split_dtype_bytes == 4) DISPATCH_DF_JK_BUILD_WORK(float);
    else if (split_dtype_bytes == 2) DISPATCH_DF_JK_BUILD_WORK(__half);
    else
    {
        fprintf(stderr, "[ERROR][%s:%d] Unsupported split_dtype_bytes (%d).\n",
                __FUNCTION__, __LINE__, split_dtype_bytes);
        return 1;
    }
    #undef DISPATCH_DF_JK_BUILD_WORK

    CUDA_CHECK( cudaStreamSynchronize(stream) );
    CUDA_CHECK( cudaGetLastError() );
    return 0;
}

}  // End of "extern C"
