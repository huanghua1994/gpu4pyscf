import os
import ctypes
import numpy as np
import cupy
import os
from typing import Union, List, Tuple

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

libmxp_df_helper = load_library('libmxp_df_helper')

_use_cuda_jk_build = os.getenv('MXP_DF_USE_CUDA_JK_BUILD', '1') == '1'

cuda_device_cc = int(cupy.cuda.device.get_compute_capability())

def use_cuda_jk_build():
    return _use_cuda_jk_build


def is_flagship_gpu() -> bool:
    return cuda_device_cc in (70, 80, 90, 100)


def get_gemm_padding(dim: int, alignment: int = 32) -> int:
    """
    Get the padded number for enabling more efficient Tensor Core GEMM kernels.

    Args:
        dim (int): Dimension to be padded.
        alignment (int): Alignment value, default is 32.
    """
    return (dim + alignment - 1) // alignment * alignment


def mxp_df_level_to_split_dtype(mxp_df_level: int) -> Tuple[int, cupy.dtype]:
    """
    Get the split dtype and number of splits based on MxP DF level.

    Args:
        mxp_df_level (int): Mixed-precision density fitting level.

    Returns:
        Tuple[int, cupy.dtype]: Number of splits and the corresponding dtype.
    """
    # TODO: need to tune
    if mxp_df_level == 0:
        return (1, cupy.float64)
    elif mxp_df_level == 1:
        # Use 3x FP16 to simulate FP64, ~1e-9 accuracy
        return (4, cupy.float16)
    elif mxp_df_level == 2:
        # Use 2x FP16 to simulater FP32, ~1e-6 accuracy
        return (3, cupy.float16)
    else:
        raise ValueError(f"Unsupported MxP DF level: {mxp_df_level}. Supported levels are 0, 1, 2.")


def cuda_jk_build(
    naux, nao, nocc, mxp_df_level, use_Dmat,
    rows, cols, cderi_sparse, C_or_D_mat, D_mat_sparse,
    build_J, build_K, J_mat_sparse, K_mat, blk_size
):
    """
    Build J and/or K matrices using the CUDA implementation.

    Args:
        naux (int): Number of auxiliary basis functions.
        nao (int): Number of atomic orbitals.
        nocc (int): Number of occupied orbitals.
        mxp_df_level (int): Mixed-precision density fitting level.
        use_Dmat (bool): Whether to use D matrix (1st iter or later).
        rows (numpy.ndarray): Row indices for cderi_sparse.
        cols (numpy.ndarray): Column indices for cderi_sparse.
        cderi_sparse (cupy.ndarray): Sparse CDERI tensor, [naux, npair]
        C_or_D_mat (cupy.ndarray): C or D matrix.
        D_mat_sparse (cupy.ndarray): Sparse D matrix matching cderi_sparse.
        build_J (bool): Whether to build J matrix.
        build_K (bool): Whether to build K matrix.
        J_mat_sparse (cupy.ndarray): Output sparse J matrix matching cderi_sparse.
        K_mat (cupy.ndarray): Output dense K matrix.
        blk_size (int): Block size on the naux dimension.
    """
    stream = cupy.cuda.get_current_stream()
    assert len(rows) == len(cols)
    npair = len(rows)
    rows_int32 = rows.astype(np.int32)
    cols_int32 = cols.astype(np.int32)

    num_split, split_dtype = mxp_df_level_to_split_dtype(mxp_df_level)
    if split_dtype == cupy.float64:
        split_dtype_bytes = 8
    elif split_dtype == cupy.float32:
        split_dtype_bytes = 4
    elif split_dtype == cupy.float16:
        split_dtype_bytes = 2
    else:
        raise ValueError(f'Unsupported split dtype: {split_dtype}')

    use_Dmat = 1 if use_Dmat else 0
    build_J = 1 if build_J else 0
    build_K = 1 if build_K else 0

    J_mat_sparse_ptr = J_mat_sparse.data.ptr if J_mat_sparse is not None else 0
    K_mat_ptr = K_mat.data.ptr if K_mat is not None else 0

    padded_nao = get_gemm_padding(nao)
    padded_nao_2 = padded_nao * padded_nao
    # Each cderi block cannot > 4GB, some old version of cublasGemmStridedBatchedEx()
    # cannot handle > 4GB input (int32 range issue)
    blk_size2 = (4 * 1024 * 1024 * 1024) // (split_dtype_bytes * padded_nao_2)
    blk_size2 = get_gemm_padding(blk_size2-32, alignment=32)
    blk_size = min(blk_size, blk_size2)
    workbuf_bytes = 32 * 1024 * 1024 + 1024  # cuBLAS workspace and padding for aligment
    workbuf_bytes += split_dtype_bytes * num_split * padded_nao_2  # C_or_D_splits
    workbuf_bytes += split_dtype_bytes * num_split * blk_size * padded_nao_2  # cderi_splits
    workbuf_bytes += split_dtype_bytes * num_split * blk_size * padded_nao_2  # rho_K_splits
    workbuf_bytes += split_dtype_bytes * num_split * padded_nao_2  # padded_K_splits
    workbuf_bytes += 4 * num_split * blk_size * padded_nao_2  # fp32_out_splits
    workbuf = cupy.empty(workbuf_bytes, dtype=cupy.uint8)

    err = libmxp_df_helper.DF_JK_build(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.c_int(naux),
        ctypes.c_int(nao),
        ctypes.c_int(nocc),
        ctypes.c_int(num_split),
        ctypes.c_int(split_dtype_bytes),
        ctypes.c_int(use_Dmat),
        ctypes.c_int(npair),
        rows_int32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cols_int32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.cast(cderi_sparse.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_or_D_mat.data.ptr, ctypes.c_void_p),
        ctypes.cast(D_mat_sparse.data.ptr, ctypes.c_void_p),
        ctypes.c_int(build_J),
        ctypes.c_int(build_K),
        ctypes.cast(J_mat_sparse_ptr, ctypes.c_void_p),
        ctypes.cast(K_mat_ptr, ctypes.c_void_p),
        ctypes.c_int(blk_size),
        ctypes.cast(workbuf.data.ptr, ctypes.c_void_p),
    )

    if err != 0:
        raise RuntimeError('Failed in DF_JK_build()')
