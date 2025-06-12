#include <stdio.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(statement)                                                       \
    do                                                                              \
    {                                                                               \
        cudaError_t result = (statement);                                           \
        if (cudaSuccess != result)                                                  \
        {                                                                           \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__,   \
                    cudaGetErrorString(result));                                    \
        }                                                                           \
        assert(cudaSuccess == result);                                              \
    } while (0)


#define CUBLAS_CHECK(statement)                                                     \
    do                                                                              \
    {                                                                               \
        cublasStatus_t result = (statement);                                        \
        if (CUBLAS_STATUS_SUCCESS != result)                                        \
        {                                                                           \
            fprintf(stderr, "[%s:%d] cuBLAS failed: ", __FILE__, __LINE__);         \
            fprintf(stderr, "%d\n", result);                                        \
        }                                                                           \
        assert(CUBLAS_STATUS_SUCCESS == result);                                    \
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

static int cderi_nnz = 0;
static int *cderi_rows_d = nullptr;
static int *cderi_cols_d = nullptr;

extern "C" {

int unpack_sym_split_cderi(
    cudaStream_t stream, const int nnz, const int *rows, const int *cols,
    const int block_size, const int nrow, const int itype_bytes,
    const void *cderi_sparse, const int num_split, const int otype_bytes,
    void *cderi_0, void *cderi_1, void *cderi_2, void *cderi_3
)
{
    if (num_split < 1 || num_split > 4)
    {
        fprintf(stderr, "num_split must be between 1 and 4.\n");
        return 1;
    }

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

    dim3 block(16, 16), grid(block_size);
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
    if (itype_bytes == 8 && otype_bytes == 4)
    {
        if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, float, 1);
        if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, float, 2);
        if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, float, 3);
        if (num_split == 4) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, float, 4);
    }
    if (itype_bytes == 8 && otype_bytes == 2)
    {
        if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, __half, 1);
        if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, __half, 2);
        if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, __half, 3);
        if (num_split == 4) DISPATCH_UNPACK_SYM_SPLIT_CDERI(double, __half, 4);
    }
    if (itype_bytes == 4 && otype_bytes == 4)
    {
        if (num_split == 1) DISPATCH_UNPACK_SYM_SPLIT_CDERI(float, float, 1);
        if (num_split == 2) DISPATCH_UNPACK_SYM_SPLIT_CDERI(float, float, 2);
        if (num_split == 3) DISPATCH_UNPACK_SYM_SPLIT_CDERI(float, float, 3);
        if (num_split == 4) DISPATCH_UNPACK_SYM_SPLIT_CDERI(float, float, 4);
    }
    #undef DISPATCH_UNPACK_SYM_SPLIT_CDERI

    CUDA_CHECK( cudaStreamSynchronize(stream) );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "CUDA error in unpack_sym_split_cderi: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

}  // End of "extern C"
