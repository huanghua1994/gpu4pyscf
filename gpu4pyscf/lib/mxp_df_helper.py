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
    if mxp_df_level == 0:
        return (1, cupy.float64)
    elif mxp_df_level == 1:
        if is_flagship_gpu():
            # Use 3x FP16 to simulate FP64, ~1e-9 accuracy
            return (3, cupy.float16)
        else:
            # Use 2x FP32 to simulate FP64, ~1e-12 accuracy
            return (2, cupy.float32)
    elif mxp_df_level == 2:
        # Use 2x FP16 to simulater FP32, ~1e-6 accuracy
        return (2, cupy.float16)
    else:
        raise ValueError(f"Unsupported MxP DF level: {mxp_df_level}. Supported levels are 0, 1, 2.")


class FpSplitHelper:
    num_split: int
    input_dtype: cupy.dtype
    split_dtype: cupy.dtype
    split_tensors: List[cupy.ndarray] = []

    def __init__(
        self,
        input_dtype: cupy.dtype,
        num_split: int = 0,
        split_dtype: cupy.dtype = cupy.float64,
        split_tensors: Union[List[cupy.ndarray], None] = None
    ):
        self.input_dtype = input_dtype
        self.num_split = num_split
        self.split_dtype = split_dtype
        self.split_tensors = [] if split_tensors is None else split_tensors


    def split(self, num_split: int, split_dtype: cupy.dtype, input_tensor: cupy.ndarray):
        """
        Split the input tensor into multiple tensors of a specified dtype.

        Args:
            num_split (int): Number of splits to create.
            split_dtype (cupy.dtype): The dtype for the split tensors.
            input_tensor (cupy.ndarray): The tensor to be split.
        """
        if num_split <= 0:
            raise ValueError("num_split must be a positive integer.")
        input_dtype = input_tensor.dtype

        if split_dtype == input_dtype:
            num_split = 1

        self.num_split = num_split
        self.input_dtype = input_dtype
        self.split_dtype = split_dtype

        scaling_factor = 1
        if split_dtype == cupy.float16:
            # e5m9, 10 bits of effective precision
            scaling_factor = 2 ** 10
        if split_dtype == cupy.float32:
            # e8m23, 24 bits of effective precision
            scaling_factor = 2 ** 24

        curr_tensor = input_tensor
        for i in range(num_split):
            split_tensor = curr_tensor.astype(split_dtype)
            self.split_tensors.append(split_tensor)
            if i < num_split - 1:
                curr_tensor = curr_tensor - split_tensor.astype(input_dtype)
                curr_tensor *= scaling_factor

    def reshape(self, new_shape: Union[List[int], Tuple[int]]):
        """
        Reshape the split tensors to a new shape.

        Args:
            new_shape (Union[List[int], Tuple[int]]): The new shape for the split tensors.
        """
        self.split_tensors = [tensor.reshape(new_shape) for tensor in self.split_tensors]

    def transpose(self, axes: Union[List[int], Tuple[int]]):
        """
        Transpose the split tensors according to the specified axes.

        Args:
            axes (Union[List[int], Tuple[int]]): The axes to transpose the split tensors.
        """
        self.split_tensors = [tensor.transpose(axes) for tensor in self.split_tensors]

    def upcast(self) -> cupy.ndarray:
        """
        Return the split tensors cast back to the original input dtype.
        """
        output_tensor = cupy.zeros_like(self.split_tensors[0], dtype=self.input_dtype)
        scale_inv = 1
        if self.split_dtype == cupy.float16:
            # e5m9, 10 bits of effective precision
            scale_inv = 2 ** (-10)
        if self.split_dtype == cupy.float32:
            # e8m23, 24 bits of effective precision
            scale_inv = 2 ** (-24)
        curr_scale_inv = 1
        for i in range(self.num_split):
            upcast_tensor = self.split_tensors[i].astype(self.input_dtype)
            output_tensor += curr_scale_inv * upcast_tensor
            curr_scale_inv *= scale_inv
        return output_tensor


def get_gemm_out_shape(lhs_shape, rhs_shape, trans_op):
    """
    Get GEMM output shape based on input shapes and transposition operation.
    """
    assert len(lhs_shape) == 2 and len(rhs_shape) == 2
    if trans_op == 'NN':
        assert lhs_shape[1] == rhs_shape[0], (
            f"LHS and RHS matrix contract dimensions mismatch for GEMM NN: "
            "LHS shape {lhs_shape}, RHS shape {rhs_shape}"
        )
        out_shape = (lhs_shape[0], rhs_shape[1])
    elif trans_op == 'TN':
        assert lhs_shape[0] == rhs_shape[0], (
            f"LHS and RHS matrix contract dimensions mismatch for GEMM NN: "
            "LHS shape {lhs_shape}, RHS shape {rhs_shape}"
        )
        out_shape = (lhs_shape[1], rhs_shape[1])
    else:
        raise ValueError(f"Not implemented trans_op = {trans_op}")
    return out_shape


def fp_split_gemm(lhs: FpSplitHelper, rhs: FpSplitHelper, out_num_split: int, trans_op: str = 'NN'):
    """
    Multiply two FpSplitHelper objects using GEMM and return a new FpSplitHelper.

    Args:
        lhs: FpSplitHelper for the left-hand side operand.
        rhs: FpSplitHelper for the right-hand side operand.
        out_num_split: Number of splits for the output.
        trans_op: GEMM transposition operation, 'NN' or 'TN'.

    Returns:
        A new FpSplitHelper containing the result of the GEMM operation.
    """
    assert lhs.input_dtype == rhs.input_dtype, (
        f"Mismatched LHS and RHS input dtypes: "
        f"LHS {lhs.input_dtype}, RHS {rhs.input_dtype}"
    )
    assert lhs.split_dtype == rhs.split_dtype, (
        f"Mismatched LHS and RHS split dtypes: "
        f"LHS {lhs.split_dtype}, RHS {rhs.split_dtype}"
    )
    out_shape = get_gemm_out_shape(lhs.split_tensors[0].shape, rhs.split_tensors[0].shape, trans_op)
    out_split_tensors = [cupy.zeros(out_shape, dtype=lhs.split_dtype) for _ in range(out_num_split)]
    for i in range(lhs.num_split):
        for j in range(rhs.num_split - i):
            k = i + j
            if trans_op == 'NN':
                out_split_tensors[k] += cupy.dot(lhs.split_tensors[i], rhs.split_tensors[j])
            elif trans_op == 'TN':
                out_split_tensors[k] += cupy.dot(lhs.split_tensors[i].T, rhs.split_tensors[j])
    return FpSplitHelper(lhs.input_dtype, out_num_split, lhs.split_dtype, out_split_tensors)


def cuda_jk_build(
    naux, nao, nocc, num_split, split_dtype, use_Dmat,
    rows, cols, cderi_sparse, C_or_D_mat, D_mat_sparse,
    build_J, build_K, J_mat_sparse, K_mat, avail_mem_bytes
):
    """
    Build J and/or K matrices using the CUDA implementation.

    Args:
        naux (int): Number of auxiliary basis functions.
        nao (int): Number of atomic orbitals.
        nocc (int): Number of occupied orbitals.
        num_split (int): Number of splits for mixed-precision.
        split_dtype (cupy.dtype): Data type for the split tensors.
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
        avail_mem_bytes (int): Available memory in bytes.
    """
    stream = cupy.cuda.get_current_stream()
    assert rows.dtype == np.int32
    assert cols.dtype == np.int32
    assert len(rows) == len(cols)
    npair = len(rows)

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
    blk_size = avail_mem_bytes // (split_dtype_bytes * num_split)
    blk_size -= padded_nao * padded_nao  # C or D
    blk_size = blk_size // (3 * padded_nao * padded_nao)  # cderi, rho_K, K
    if blk_size < 32:
        raise RuntimeError(f"Not enough memory for CUDA DF JK build with minimal blk_size = 32, got {blk_size=}")
    blk_size = get_gemm_padding(blk_size-32, alignment=32)
    if blk_size > 1024:
        blk_size = 1024
    workbuf_bytes = padded_nao * padded_nao  # C or D
    workbuf_bytes += 3 * padded_nao * padded_nao * blk_size  # cderi, rho_K, K
    workbuf_bytes *= split_dtype_bytes * num_split
    workbuf_bytes += 32 * 1024 * 1024 + 1024  # cuBLAS workspace and padding for aligment
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
        rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
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
