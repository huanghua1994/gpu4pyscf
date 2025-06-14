#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <vector>

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
            fprintf(stderr, "reason: %s", cudaGetErrorString(result));              \
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


template<int num_split, typename IType, typename OType>
__global__ void unpack_sym_split_cderi_kernel(
    const int nnz, const int* __restrict__ rows, const int* __restrict__ cols,
    const int nrow, const IType* __restrict__ cderi_sparse,
    OType* __restrict__ cderi_0, OType* __restrict__ cderi_1,
    OType* __restrict__ cderi_2, OType* __restrict__ cderi_3
)
{
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int bid = blockIdx.x;
    const int thread_block_size = blockDim.x * blockDim.y;
    const int mat_size = nrow * nrow;
    for (int nnz_id = tid; nnz_id < nnz; nnz_id += thread_block_size)
    {
        if (nnz_id < nnz)
        {
            int row = rows[nnz_id];
            int col = cols[nnz_id];
            size_t input_idx = bid * nnz + nnz_id;
            size_t output_idx0 = bid * mat_size + row * nrow + col;
            size_t output_idx1 = bid * mat_size + col * nrow + row;
            IType in_value = cderi_sparse[input_idx];
            OType out_value0 = static_cast<OType>(in_value);
            cderi_0[output_idx0] = out_value0;
            cderi_0[output_idx1] = out_value0;
            if constexpr(num_split == 1) continue;
            in_value -= static_cast<IType>(out_value0);
            OType out_value1 = static_cast<OType>(in_value);
            cderi_1[output_idx0] = out_value1;
            cderi_1[output_idx1] = out_value1;
            if constexpr(num_split == 2) continue;
            in_value -= static_cast<IType>(out_value1);
            OType out_value2 = static_cast<OType>(in_value);
            cderi_2[output_idx0] = out_value2;
            cderi_2[output_idx1] = out_value2;
            if constexpr(num_split == 3) continue;
            in_value -= static_cast<IType>(out_value2);
            OType out_value3 = static_cast<OType>(in_value);
            cderi_3[output_idx0] = out_value3;
            cderi_3[output_idx1] = out_value3;
        }
    }
}


template<int num_split, typename SplitType, typename OType>
__global__ void sum_rho_K_splits_kernel(
    const int nao, const int padded_nao, OType *mat_K,
    SplitType* __restrict__ padded_K_0, SplitType* __restrict__ padded_K_1,
    SplitType* __restrict__ padded_K_2, SplitType* __restrict__ padded_K_3
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < padded_nao && col < padded_nao)
    {
        int src_idx = row * padded_nao + col;
        int dst_idx = row * nao + col;
        OType K_val = mat_K[dst_idx];
        K_val += static_cast<OType>(padded_K_0[src_idx]);
        if constexpr(num_split >= 2) K_val += static_cast<OType>(padded_K_1[src_idx]);
        if constexpr(num_split >= 3) K_val += static_cast<OType>(padded_K_2[src_idx]);
        if constexpr(num_split >= 4) K_val += static_cast<OType>(padded_K_3[src_idx]);
        mat_K[dst_idx] = K_val;
    }
}

static int cderi_nnz = 0;
static int *cderi_rows_d = nullptr;
static int *cderi_cols_d = nullptr;

static double f64_one = 1, f64_zero = 0;
static float  f32_one = 1, f32_zero = 0;

static cublasHandle_t cublas_handle = nullptr;
static cublasLtHandle_t lt_handle = nullptr;
static void *cublas_workspace = nullptr;
static const size_t cublas_workspace_bytes = 32 * 1024 * 1024;

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

inline static std::vector<float> get_dev_f16_to_host_fp32(const void *dptr, size_t nelem, bool is_bfloat16)
{
    const size_t n = nelem * 2;
    char *hptr = (char *) malloc(n);
    cudaMemcpy((void *) hptr, dptr, n, cudaMemcpyDeviceToHost);
    std::vector<float> floats;
    if (is_bfloat16) {
        __nv_bfloat16* h16p = reinterpret_cast<__nv_bfloat16*>(hptr);
        for (size_t i = 0; i < nelem; i++)
            floats.push_back(__bfloat162float(h16p[i]));
    } else {
        __half* h16p = reinterpret_cast<__half*>(hptr);
        for (size_t i = 0; i < nelem; i++)
            floats.push_back(__half2float(h16p[i]));
    }
    free(hptr);
    return floats;
};

inline static void print_matrix(float *mat, int ldm, int nrow, int ncol, const char *name)
{
    printf("Matrix %s:\n", name);
    for (int i = 0; i < nrow; ++i)
    {
        for (int j = 0; j < ncol; ++j)
        {
            printf("% 8.4e ", mat[i * ldm + j]);
        }
        printf("\n");
    }
    printf("\n");
}

inline static void dump_binary(const char *filename, const void *data, size_t size)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    size_t written = fwrite(data, 1, size, fp);
    if (written != size) {
        fprintf(stderr, "fwrite failed: wrote %zu of %zu bytes\n", written, size);
    }

    fclose(fp);
    printf("Dumped binary data to %s (%zu bytes)\n", filename, written);
}

#define DF_K_BUILD_SUBTASK_COMMON \
    if (cublas_handle == nullptr) \
    { \
        CUBLAS_CHECK( cublasCreate(&cublas_handle) ); \
        CUBLAS_CHECK( cublasLtCreate(&lt_handle) ); \
        CUDA_CHECK( cudaMalloc((void**) &cublas_workspace, cublas_workspace_bytes) ); \
    } \
    CUBLAS_CHECK( cublasSetStream(cublas_handle, stream) ); \
    cublasComputeType_t compute_type; \
    cudaDataType_t AB_type, C_type, scale_type; \
    get_cublas_gemm_dtypes<SplitType>(compute_type, AB_type, C_type, scale_type); \


template<int num_split, typename SplitType>
void DF_K_build_subtask1(
    cudaStream_t stream, const int use_Dmat, const int init_output,
    const int i, const int j,
    const int naux_blk, const int padded_nao, const int padded_nocc,
    SplitType *cderi_i, SplitType *C_or_D_j, SplitType *rho_K_k
)
{
    DF_K_BUILD_SUBTASK_COMMON;

    char fname_prefix[256];
    sprintf(fname_prefix, "i%d_j%d_k%d", i, j, i+j);

    // Step 1: einsum Lij,jk -> Lki. 
    // For each L, ij,jk -> ki is doing C^T = B^T * A^T. Since the input cderi and C/D are in
    // row-major and cuBLAS takes column-major, A^T, B^T, C^T in row-major == A, B, C in
    // column-major. We do a strided batched GEMM for C = B * A in column-major.
    int m = use_Dmat ? padded_nao : padded_nocc;
    int n = padded_nao;
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
    const void *A = static_cast<const void*>(C_or_D_j);
    const void *B = static_cast<const void*>(cderi_i);
    void *C = static_cast<void*>(rho_K_k);
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    int lda = m;
    int ldb = k;
    int ldc = m;
    long long int stride_A = 0;
    long long int stride_B = static_cast<long long int>(k * n);
    long long int stride_C = static_cast<long long int>(m * n);
    int batch_count = naux_blk;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    auto A_arr = get_dev_f16_to_host_fp32(A, /*padded_nao * padded_nocc*/ 5 * padded_nocc, 0);
    auto B_arr = get_dev_f16_to_host_fp32(B, /*naux_blk * padded_nao * padded_nao*/ 5 * padded_nao, 0);
    auto C_arr_before = get_dev_f16_to_host_fp32(C, /*naux_blk * padded_nocc * padded_nao */ 5 * padded_nao, 0);
    //print_matrix(A_arr.data(), padded_nocc, 5, 5, "lhs (original rhs, C or D) [0:5, 0:5]");
    //print_matrix(B_arr.data(), padded_nao, 5, 5, "rhs (cderi_i) [0:5, 0:5]");
    //print_matrix(C_arr_before.data(), padded_nao, 5, 5, "C before [0:5, 0:5]");

    CUBLAS_CHECK( cublasGemmStridedBatchedEx(
        cublas_handle, transA, transB, m, n, k,
        alpha,
        A, AB_type, lda, stride_A,
        B, AB_type, ldb, stride_B,
        beta,
        C, C_type, ldc, stride_C,
        batch_count, compute_type, algo
    ) );

    auto C_arr_after = get_dev_f16_to_host_fp32(C, naux_blk * padded_nocc * padded_nao, 0);
    print_matrix(C_arr_after.data(), padded_nao, 5, 5, "C after [0:5, 0:5]");
    /*
    dump_binary(
        (std::string(fname_prefix) + "_rhok_out.bin").c_str(),
        C_arr_after.data(), 4 * naux_blk * padded_nao * padded_nocc
    );
    */
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

    char fname_prefix[256];
    sprintf(fname_prefix, "i%d_j%d_k%d", i, j, i+j);

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

    auto A_arr = get_dev_f16_to_host_fp32(A, naux_blk * padded_nocc * padded_nao, 0);
    auto B_arr = get_dev_f16_to_host_fp32(B, naux_blk * padded_nocc * padded_nao, 0);
    auto C_arr_before = get_dev_f16_to_host_fp32(C, 5 * padded_nao, 0);
    print_matrix(A_arr.data(), padded_nao, 5, 5, "lhs (rho_k_i) [0:5, 0:5]");
    print_matrix(B_arr.data(), padded_nao, 5, 5, "rhs (rho_k_j) [0:5, 0:5]");
    print_matrix(C_arr_before.data(), padded_nao, 5, 5, "C before [0:5, 0:5]");

    printf("[DEBUG] m, n, k = %d, %d, %d\n", m, n, k);
    printf("[DEBUG] lda, ldb, ldc = %d, %d, %d\n", lda, ldb, ldc);
    
    dump_binary(
        (std::string(fname_prefix) + "_rhok_i.bin").c_str(),
        (const void *) A_arr.data(), 4 * naux_blk * padded_nocc * padded_nao
    );

    #if 0
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

    auto C_arr_after = get_dev_f16_to_host_fp32(C, 5 * padded_nao, 0);
    print_matrix(C_arr_after.data(), padded_nao, 5, 5, "C after [0:5, 0:5]");
}

template<int num_split, typename OType, typename SplitType>
void DF_K_build_work(
    cudaStream_t stream, const int use_Dmat,
    const int naux_blk, const int nao, const int padded_nao, const int padded_nocc,
    SplitType *cderi_split_0, SplitType *cderi_split_1,
    SplitType *cderi_split_2, SplitType *cderi_split_3,
    SplitType *C_or_D_split_0, SplitType *C_or_D_split_1,
    SplitType *C_or_D_split_2, SplitType *C_or_D_split_3, 
    OType *mat_K
)
{
    SplitType *rho_K_splits = nullptr, *padded_K_splits = nullptr;
    size_t rho_K_size = naux_blk * padded_nao * (use_Dmat ? padded_nao : padded_nocc);
    size_t padded_K_size = naux_blk * padded_nao * padded_nao;
    CUDA_CHECK( cudaMalloc((void **) &rho_K_splits, sizeof(SplitType) * num_split * rho_K_size) );
    CUDA_CHECK( cudaMalloc((void **) &padded_K_splits, sizeof(SplitType) * num_split * padded_K_size) );

    SplitType *cderi_splits[4] = {cderi_split_0, cderi_split_1, cderi_split_2, cderi_split_3};
    SplitType *C_or_D_splits[4] = {C_or_D_split_0, C_or_D_split_1, C_or_D_split_2, C_or_D_split_3};

    printf("[DEBUG] ********** Lij,jk -> Lki **********\n");
    for (int i = 0; i < num_split; i++)
    {
        int init_output = (i == 0);
        SplitType *cderi_i = cderi_splits[i];
        for (int j = 0; j < num_split - i; j++)
        {
            const int k = i + j;
            printf("[DEBUG] =====> subtask: i=%d, j=%d, k=%d <=====\n", i, j, k);
            SplitType *C_or_D_j = C_or_D_splits[j];
            SplitType *rho_K_k = rho_K_splits + k * rho_K_size;
            DF_K_build_subtask1<num_split, SplitType>(
                stream, use_Dmat, init_output, i, j,
                naux_blk, padded_nao, padded_nocc,
                cderi_i, C_or_D_j, rho_K_k
            );
        }
    }

    printf("[DEBUG] ********** Lki,Lkj -> ij **********\n");
    for (int i = 0; i < num_split; i++)
    {
        SplitType *cderi_i = cderi_splits[i];
        SplitType *rho_K_i = rho_K_splits + i * rho_K_size;
        SplitType *cderi_or_rko_K_i = use_Dmat ? cderi_i : rho_K_i;
        int init_output = (i == 0);
        for (int j = 0; j < num_split - i; j++)
        {
            const int k = i + j;
            printf("[DEBUG] =====> subtask: i=%d, j=%d, k=%d <=====\n", i, j, k);
            SplitType *rho_K_j = rho_K_splits + j * rho_K_size;
            SplitType *padded_K_k = padded_K_splits + k * padded_K_size;
            DF_K_build_subtask2<num_split, SplitType>(
                stream, use_Dmat, init_output, i, j,
                naux_blk, padded_nao, padded_nocc,
                cderi_or_rko_K_i, rho_K_j, padded_K_k
            );
        }
    }

    int nget = 10;
    auto K_0_arr = get_dev_f16_to_host_fp32(padded_K_splits, nget, 0);
    auto K_1_arr = get_dev_f16_to_host_fp32(padded_K_splits + padded_K_size, nget, 0);
    printf("[DEBUG] First %d elements of padded_K_splits:\n", nget);
    for (int i = 0; i < nget; ++i) printf("%e ", K_0_arr[i]);
    printf("\n");
    printf("[DEBUG] First %d elements of padded_K_splits + padded_K_size:\n", nget);
    for (int i = 0; i < nget; ++i) printf("%e ", K_1_arr[i]);
    printf("\n");

    SplitType *padded_K_0 = padded_K_splits;
    SplitType *padded_K_1 = num_split > 1 ? padded_K_splits + padded_K_size : nullptr;
    SplitType *padded_K_2 = num_split > 2 ? padded_K_splits + padded_K_size * 2 : nullptr;
    SplitType *padded_K_3 = num_split > 3 ? padded_K_splits + padded_K_size * 3 : nullptr;
    dim3 block(16, 16);
    dim3 grid((nao + block.x - 1) / block.x, (nao + block.y - 1) / block.y);
    sum_rho_K_splits_kernel<num_split, SplitType><<<grid, block, 0, stream>>>(
        nao, padded_nao, mat_K,
        padded_K_0, padded_K_1, padded_K_2, padded_K_3
    );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaStreamSynchronize(stream) );

    CUDA_CHECK( cudaFree(rho_K_splits));
    CUDA_CHECK( cudaFree(padded_K_splits));
}

extern "C" {

int unpack_sym_split_cderi(
    cudaStream_t stream, const int nnz, const int *rows, const int *cols,
    const int naux_blk, const int nrow, const int itype_bytes,
    const void *cderi_sparse, const int num_split, const int otype_bytes,
    void *cderi_0, void *cderi_1, void *cderi_2, void *cderi_3
)
{
    ERROR_CHECK(num_split >= 1 && num_split <= 4, "num_split must be between 1 and 4.");

    if (nnz != cderi_nnz)
    {
        CUDA_CHECK( cudaFree(cderi_rows_d) );
        CUDA_CHECK( cudaFree(cderi_cols_d) );
        cderi_nnz = nnz;
        CUDA_CHECK( cudaMalloc((void**) &cderi_rows_d, nnz * sizeof(int)) );
        CUDA_CHECK( cudaMalloc((void**) &cderi_cols_d, nnz * sizeof(int)) );
        size_t rows_cols_bytes = sizeof(int) * nnz;
        CUDA_CHECK( cudaMemcpyAsync(cderi_rows_d, rows, rows_cols_bytes, cudaMemcpyHostToDevice, stream) );
        CUDA_CHECK( cudaMemcpyAsync(cderi_cols_d, cols, rows_cols_bytes, cudaMemcpyHostToDevice, stream) );
    }

    dim3 block(16, 16), grid(naux_blk);
    #define DISPATCH_UNPACK_SYM_SPLIT_CDERI(itype, otype, num_split) \
    do { \
        unpack_sym_split_cderi_kernel<num_split, itype, otype><<<grid, block, 0, stream>>>( \
            nnz, cderi_rows_d, cderi_cols_d, nrow, \
            static_cast<const itype*>(cderi_sparse), \
            static_cast<otype*>(cderi_0), \
            static_cast<otype*>(cderi_1), \
            static_cast<otype*>(cderi_2), \
            static_cast<otype*>(cderi_3) \
        ); \
        CUDA_CHECK(cudaGetLastError()); \
    } while (0)
    if (itype_bytes == 8 && otype_bytes == 8)
    {
        if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, double, 1);
        if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, double, 2);
        if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, double, 3);
        if (num_split == 4) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, double, 4);
    }
    else if (itype_bytes == 8 && otype_bytes == 4)
    {
        if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, float, 1);
        if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, float, 2);
        if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, float, 3);
        if (num_split == 4) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, float, 4);
    }
    else if (itype_bytes == 8 && otype_bytes == 2)
    {
        if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, __half, 1);
        if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, __half, 2);
        if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, __half, 3);
        if (num_split == 4) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, __half, 4);
    }
    else if (itype_bytes == 4 && otype_bytes == 4)
    {
        if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(float, float, 1);
        if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(float, float, 2);
        if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(float, float, 3);
        if (num_split == 4) DISPATCH_UNPACK_SYM_SPLIT_CDERI(float, float, 4);
    }
    else
    {
        fprintf(stderr, "[ERROR][%s:%s] Unsupported itype_bytes (%d) or otype_bytes (%d).\n",
                __FILE__, __FUNCTION__, itype_bytes, otype_bytes);
        return 1;
    }
    #undef DISPATCH_UNPACK_SYM_SPLIT_CDERI

    CUDA_CHECK( cudaStreamSynchronize(stream) );
    CUDA_CHECK( cudaGetLastError() );
    return 0;
}


int DF_K_build(
    cudaStream_t stream, const int use_Dmat,
    const int naux_blk, const int nao, const int padded_nao, const int padded_nocc,
    const int num_split, const int split_dtype_bytes,
    void *cderi_split_0, void *cderi_split_1, void *cderi_split_2, void *cderi_split_3,
    void *C_or_D_split_0, void *C_or_D_split_1, void *C_or_D_split_2, void *C_or_D_split_3, 
    void *mat_K
)
{
    ERROR_CHECK(num_split >= 1 && num_split <= 4, "num_split must be between 1 and 4.");
    ERROR_CHECK(mat_K != nullptr, "mat_K must not be nullptr.");

    #define DISPATCH_DF_K_BUILD_WORK(num_split, out_dytpe, split_dtype) \
    do { \
        out_dytpe *mat_K_ = reinterpret_cast<out_dytpe*>(mat_K); \
        DF_K_build_work<num_split, out_dytpe, split_dtype>( \
            stream, use_Dmat, naux_blk, nao, padded_nao, padded_nocc, \
            static_cast<split_dtype *>(cderi_split_0), \
            static_cast<split_dtype *>(cderi_split_1), \
            static_cast<split_dtype *>(cderi_split_2), \
            static_cast<split_dtype *>(cderi_split_3), \
            static_cast<split_dtype *>(C_or_D_split_0), \
            static_cast<split_dtype *>(C_or_D_split_1), \
            static_cast<split_dtype *>(C_or_D_split_2), \
            static_cast<split_dtype *>(C_or_D_split_3), \
            mat_K_ \
        ); \
    } while (0)
    if (split_dtype_bytes == 8)
    {
        ERROR_CHECK(num_split == 1, "num_split must be 1 when split_dtype is double.");
        DISPATCH_DF_K_BUILD_WORK(1, double, double);
    }
    else if (split_dtype_bytes == 4)
    {
        if (num_split == 1) DISPATCH_DF_K_BUILD_WORK(1, double, float);
        if (num_split == 2) DISPATCH_DF_K_BUILD_WORK(2, double, float);
        if (num_split == 3) DISPATCH_DF_K_BUILD_WORK(3, double, float);
        if (num_split == 4) DISPATCH_DF_K_BUILD_WORK(4, double, float);
    }
    else if (split_dtype_bytes == 2)
    {
        if (num_split == 1) DISPATCH_DF_K_BUILD_WORK(1, double, __half);
        if (num_split == 2) DISPATCH_DF_K_BUILD_WORK(2, double, __half);
        if (num_split == 3) DISPATCH_DF_K_BUILD_WORK(3, double, __half);
        if (num_split == 4) DISPATCH_DF_K_BUILD_WORK(4, double, __half);
    }
    else
    {
        fprintf(
            stderr, "[ERROR][%s:%d] split_dtype must be double, float, or __half.\n",
            __FUNCTION__, __LINE__
        );
        return 1;
    }
    #undef DISPATCH_DF_K_BUILD_WORK

    CUDA_CHECK( cudaStreamSynchronize(stream) );
    CUDA_CHECK( cudaGetLastError() );
    return 0;
}

}  // End of "extern C"
