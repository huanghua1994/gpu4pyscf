import os
import sys
import functools
import ctypes
import numpy as np
import cupy
import pdb

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

libmxp_df_helper = load_library('libmxp_df_helper')

def unpack_sym_split_cderi(
    rows, cols, itype_bytes, cderi_sparse, padded_nao, num_split,
    otype_bytes, cderi_0, cderi_1, cderi_2, cderi_3
):
    stream = cupy.cuda.get_current_stream()
    assert rows.dtype == np.int32
    assert cols.dtype == np.int32
    assert len(rows) == len(cols)
    nnz = len(rows)

    naux_blk = cderi_sparse.shape[0]
    cderi_0_ptr = cderi_0.data.ptr
    cderi_1_ptr = 0
    cderi_2_ptr = 0
    cderi_3_ptr = 0
    if num_split >= 2:
        assert cderi_1 is not None
        cderi_1_ptr = cderi_1.data.ptr
    if num_split >= 3:
        assert cderi_2 is not None
        cderi_2_ptr = cderi_2.data.ptr
    if num_split >= 4:
        assert cderi_3 is not None
        cderi_3_ptr = cderi_3.data.ptr

    err = libmxp_df_helper.unpack_sym_split_cderi(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.c_int(nnz),
        rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(naux_blk),
        ctypes.c_int(padded_nao),
        ctypes.c_int(itype_bytes),
        ctypes.cast(cderi_sparse.data.ptr, ctypes.c_void_p),
        ctypes.c_int(num_split),
        ctypes.c_int(otype_bytes),
        ctypes.cast(cderi_0_ptr, ctypes.c_void_p),
        ctypes.cast(cderi_1_ptr, ctypes.c_void_p),
        ctypes.cast(cderi_2_ptr, ctypes.c_void_p),
        ctypes.cast(cderi_3_ptr, ctypes.c_void_p),
    )

    if err != 0:
        raise RuntimeError('failed in unpack_sym_split_cderi')

def DF_K_build(
    use_Dmat, naux_blk, nao, padded_nao, padded_nocc,
    num_split, cderi_splits, C_or_D_splits, K_mat
):
    stream = cupy.cuda.get_current_stream()

    cderi_0_ptr = cderi_splits[0].data.ptr
    cderi_1_ptr = 0
    cderi_2_ptr = 0
    cderi_3_ptr = 0
    if num_split >= 2:
        assert len(cderi_splits) >= 2
        cderi_1_ptr = cderi_splits[1].data.ptr
    if num_split >= 3:
        assert len(cderi_splits) >= 3
        cderi_2_ptr = cderi_splits[2].data.ptr
    if num_split >= 4:
        assert len(cderi_splits) >= 4
        cderi_3_ptr = cderi_splits[3].data.ptr

    C_or_D_0_ptr = C_or_D_splits[0].data.ptr
    C_or_D_1_ptr = 0
    C_or_D_2_ptr = 0
    C_or_D_3_ptr = 0
    if num_split >= 2:
        assert len(C_or_D_splits) >= 2
        C_or_D_1_ptr = C_or_D_splits[1].data.ptr
    if num_split >= 3:
        assert len(C_or_D_splits) >= 3
        C_or_D_2_ptr = C_or_D_splits[2].data.ptr
    if num_split >= 4:
        assert len(C_or_D_splits) >= 4
        C_or_D_3_ptr = C_or_D_splits[3].data.ptr

    split_dtype_bytes = cderi_splits[0].itemsize

    err = libmxp_df_helper.DF_K_build(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.c_int(use_Dmat),
        ctypes.c_int(naux_blk),
        ctypes.c_int(nao),
        ctypes.c_int(padded_nao),
        ctypes.c_int(padded_nocc),
        ctypes.c_int(num_split),
        ctypes.c_int(split_dtype_bytes),
        ctypes.cast(cderi_0_ptr, ctypes.c_void_p),
        ctypes.cast(cderi_1_ptr, ctypes.c_void_p),
        ctypes.cast(cderi_2_ptr, ctypes.c_void_p),
        ctypes.cast(cderi_3_ptr, ctypes.c_void_p),
        ctypes.cast(C_or_D_0_ptr, ctypes.c_void_p),
        ctypes.cast(C_or_D_1_ptr, ctypes.c_void_p),
        ctypes.cast(C_or_D_2_ptr, ctypes.c_void_p),
        ctypes.cast(C_or_D_3_ptr, ctypes.c_void_p),
        ctypes.cast(K_mat.data.ptr, ctypes.c_void_p),
    )
    #pdb.set_trace()

    if err != 0:
        raise RuntimeError('failed in unpack_sym_split_cderi')