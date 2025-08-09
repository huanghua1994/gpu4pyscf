import os
import ctypes
import numpy as np
import cupy
import os
from typing import Tuple

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

libmxp_df_helper = load_library('libmxp_df_helper')

_use_cuda_jk_build = os.getenv('MXP_DF_USE_CUDA_JK_BUILD', '1') == '1'

cuda_device_cc = int(cupy.cuda.device.get_compute_capability())

dtype_id_size_dict = {
    'FP64': (0, 8),
    'FP32': (1, 4),
}

non_flagship_gpu_mxp_strategies = {
    0: ('FP64', 0),   # Standard FP64, no split, no emulation
    1: ('FP64', 0),   # Standard FP64, no split, no emulation
    2: ('FP64', 39),  # Emulated FP64, 39 bits accuracy (~2e-12)
    3: ('FP64', 31),  # Emulated FP64, 31 bits accuracy (~4e-10)
    4: ('FP64', 23),  # Emulated FP64, 23 bits accuracy (~2e-7)
}

flagship_gpu_mxp_strategies = {
    0: ('FP64', 0),   # Standard FP64, no split, no emulation
    1: ('FP64', 0),   # Standard FP64, no split, no emulation
    2: ('FP64', 0),   # Standard FP64, no split, no emulation
    3: ('FP64', 0),   # Standard FP64, no split, no emulation
    4: ('FP64', 23),  # Emulated FP64, 23 bits accuracy (~2e-7)
}

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


def mxp_df_level_to_dtype_emu_bits(mxp_df_level: int) -> Tuple[str, int]:
    """
    Get the compute dtype and FP64 emu bits based on MxP DF level.

    Args:
        mxp_df_level (int): Mixed-precision density fitting level.

    Returns:
        Tuple[str, fp64_emu_bits]
    """
    if mxp_df_level < 0 or mxp_df_level > 4:
        raise ValueError(f"Unsupported MxP DF level: {mxp_df_level}. Supported levels are 0, 1, 2, 3, 4.")
    if is_flagship_gpu():
        return flagship_gpu_mxp_strategies[mxp_df_level]
    else:
        return non_flagship_gpu_mxp_strategies[mxp_df_level]


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

    split_dtype_str, fp64_emu_bits = mxp_df_level_to_dtype_emu_bits(mxp_df_level)
    dtype_id, split_dtype_bytes = dtype_id_size_dict[split_dtype_str]

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
    workbuf_bytes = 32 * 1024 * 1024 + 1024  # cuBLAS workspace and padding for alignment
    workbuf_bytes += split_dtype_bytes * padded_nao_2  # C_or_D_buf
    workbuf_bytes += split_dtype_bytes * blk_size * padded_nao_2  # cderi_buf
    workbuf_bytes += split_dtype_bytes * blk_size * padded_nao_2  # rho_K_0_buf
    workbuf_bytes += split_dtype_bytes * blk_size * padded_nao_2  # rho_K_buf
    workbuf_bytes += split_dtype_bytes * padded_nao_2  # padded_K_buf
    workbuf_bytes += 4 * blk_size * padded_nao_2  # fp32_out_buf
    workbuf = cupy.empty(workbuf_bytes, dtype=cupy.uint8)

    err = libmxp_df_helper.DF_JK_build(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.c_int(naux),
        ctypes.c_int(nao),
        ctypes.c_int(nocc),
        ctypes.c_int(blk_size),
        ctypes.c_int(dtype_id),
        ctypes.c_int(fp64_emu_bits),
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
        ctypes.cast(workbuf.data.ptr, ctypes.c_void_p),
    )

    if err != 0:
        raise RuntimeError('Failed in DF_JK_build()')
