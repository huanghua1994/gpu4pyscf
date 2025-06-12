import os
import sys
import functools
import ctypes
import numpy as np
import cupy

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

libmxp_df_helper = load_library('libmxp_df_helper')

import pdb
def unpack_sym_split_cderi(
    rows, cols, itype_bytes, cderi_sparse, padded_nao, num_split,
    otype_bytes, cderi_0, cderi_1, cderi_2, cderi_3
):
    stream = cupy.cuda.get_current_stream()
    assert rows.dtype == np.int32
    assert cols.dtype == np.int32
    assert len(rows) == len(cols)
    nnz = len(rows)

    block_size = cderi_sparse.shape[0]
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

    #pdb.set_trace()
    err = libmxp_df_helper.unpack_sym_split_cderi(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.c_int(nnz),
        rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(block_size),
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

