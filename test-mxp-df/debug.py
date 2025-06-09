import numpy as np
import math
import pdb

def check_error(x_ref, x, name=None):
    err = x_ref - x
    abserr = np.abs(err)
    err_fnorm = np.linalg.norm(err.flatten())
    relerr_fnorm = err_fnorm / np.linalg.norm(x_ref.flatten())
    elem_max_abserr = np.max(abserr)
    elem_relerr = abserr / np.abs(x_ref)
    elem_relerr[np.isnan(elem_relerr)] = 0.0
    elem_max_relerr = np.max(elem_relerr)
    if name is not None:
        print(f"\nArray {name}:")
        print(f"  Element-wise max abs error : {elem_max_abserr:.9e}")
        print(f"  Element-wise max rel error : {elem_max_relerr:.9e}")
        print(f"  Frobenius norm rel error   : {relerr_fnorm:.9e}")

def buf_to_split_arrays(buf, num_split, shape):
    split_size = np.prod(shape)
    split_arrays = []
    for i in range(num_split):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        split_arrays.append(buf[start_idx:end_idx].reshape(shape))
    return split_arrays

def os_split_c(num_split, x, split_dtype=np.float16, contract_dim=1):
    curr_x = x
    x_splits = []
    c_splits = []
    # FP64, FP32, FP16 mantissa bits 52, 23, 10
    cd_bits = math.log2(contract_dim)
    rho = math.ceil(52.0 - min(10.0, (23.0 - cd_bits) * 0.5))
    mu = np.max(np.abs(x))
    #mu = 1.0
    for i in range(num_split):
        tau = math.ceil(math.log2(mu))
        sigma = math.pow(2.0, rho + tau)
        x_tmp = (curr_x + sigma) - sigma
        curr_x = curr_x - x_tmp
        split_i = math.pow(2.0, -tau) * x_tmp
        c_splits.append(tau)
        x_splits.append(split_i.astype(split_dtype))
        if i < num_split - 1:
            mu = np.max(np.abs(curr_x))
            #mu = mu / 1024.0
    return x_splits, c_splits

def os_upcast_c(num_split, x_splits, c_splits, out_dtype=np.float64):
    output = np.zeros_like(x_splits[0], dtype=out_dtype)
    for i in range(num_split):
        scale = math.pow(2.0, c_splits[i])
        output += scale * x_splits[i].astype(out_dtype)
    return output

def os_gemm_c(A_splits, A_c, B_splits, B_c, enable_fast_mode=True):
    out_splits = []
    c_splits = []
    num_splits = min(len(A_splits), len(B_splits))
    for i in range(len(A_splits)):
        for j in range(len(B_splits)):
            if (i + j >= num_splits) and enable_fast_mode:
                continue
            out_ij = np.matmul(A_splits[i], B_splits[j])
            c_ij = A_c[i] + B_c[j]
            out_splits.append(out_ij)
            c_splits.append(c_ij)
    return out_splits, c_splits

def os_split(num_split, x, split_dtype=np.float16, contract_dim=1):
    curr_x = x
    x_splits = []
    #c_splits = []
    # FP64, FP32, FP16 mantissa bits 52, 23, 10
    cd_bits = math.log2(contract_dim)
    rho = math.ceil(52.0 - min(10.0, (23.0 - cd_bits) * 0.5))
    #mu = np.max(np.abs(x))
    mu = 1.0
    for i in range(num_split):
        tau = math.ceil(math.log2(mu))
        sigma = math.pow(2.0, rho + tau)
        x_tmp = (curr_x + sigma) - sigma
        curr_x = curr_x - x_tmp
        split_i = math.pow(2.0, -tau) * x_tmp
        #c_splits.append(tau)
        x_splits.append(split_i.astype(split_dtype))
        if i < num_split - 1:
            #mu = np.max(np.abs(curr_x))
            mu = mu / 1024.0
    return x_splits #, c_splits

def os_upcast(num_split, x_splits, out_dtype=np.float64):
    output = np.zeros_like(x_splits[0], dtype=out_dtype)
    for i in range(num_split):
        scale = math.pow(2.0, -10 * i)
        output += scale * x_splits[i].astype(out_dtype)
    return output

def os_gemm(num_split, A_splits, B_splits):
    assert len(A_splits) == len(B_splits) == num_split
    out_shape = (A_splits[0].shape[0], B_splits[0].shape[1])
    out_splits = [np.zeros(out_shape, dtype=A_splits[0].dtype) for _ in range(num_split)]
    for i in range(num_split):
        for j in range(num_split - i):
            k = i + j
            out_splits[k] += np.matmul(A_splits[i], B_splits[j])
    return out_splits

if __name__ == "__main__":
    num_split = 3
    nao = 537
    padded_nao = 544
    naux_blk = 128

    cderi_fp64 = np.fromfile('cderi_fp64.bin', dtype=np.float64)
    cderi_fp64 = cderi_fp64.reshape((naux_blk, padded_nao, padded_nao))

    C_or_D_fp64 = np.fromfile('C_or_D_fp64.bin', dtype=np.float64)
    C_or_D_fp64 = C_or_D_fp64.reshape(nao, nao)
    tmp = np.zeros((padded_nao, padded_nao), dtype=np.float64)
    tmp[:nao, :nao] = C_or_D_fp64
    C_or_D_fp64 = tmp

    rho_K_fp64 = np.fromfile('rho_K_fp64.bin', dtype=np.float64)
    rho_K_fp64 = rho_K_fp64.reshape((naux_blk, padded_nao, padded_nao))

    rho_K_fp64_calc = np.einsum('Pij,jk->Pki', cderi_fp64, C_or_D_fp64)
    check_error(rho_K_fp64, rho_K_fp64_calc, name='rho_K_fp64 calc vs ref')

    #pdb.set_trace()
    C_or_D_fp16 = np.fromfile('C_or_D_fp16.bin', dtype=np.float16)
    C_or_D_fp16_splits = buf_to_split_arrays(C_or_D_fp16, num_split, (padded_nao, padded_nao))
    C_or_D_fp16_upcast = os_upcast(num_split, C_or_D_fp16_splits, out_dtype=np.float64)
    check_error(C_or_D_fp64, C_or_D_fp16_upcast, name='C_or_D_fp16 upcast vs ref')

    cderi_fp16 = np.fromfile('cderi_fp16.bin', dtype=np.float16)
    cderi_fp16_splits = buf_to_split_arrays(cderi_fp16, num_split, (naux_blk, padded_nao, padded_nao))
    cderi_fp16_upcast = os_upcast(num_split, cderi_fp16_splits, out_dtype=np.float64)
    check_error(cderi_fp64, cderi_fp16_upcast, name='cderi_fp16 upcast vs ref')

    C_or_D_fp32_splits = [a.astype(np.float32) for a in C_or_D_fp16_splits]
    cderi_fp32_splits = [a.astype(np.float32) for a in cderi_fp16_splits]
    cderi_fp32_splits = [a.reshape(naux_blk * padded_nao, padded_nao) for a in cderi_fp32_splits]
    rho_K_fp32_calc_splits = os_gemm(num_split, cderi_fp32_splits, C_or_D_fp32_splits)
    rho_K_fp32_calc_splits = [a.reshape(naux_blk, padded_nao, padded_nao) for a in rho_K_fp32_calc_splits]
    rho_K_fp32_calc_splits = [a.transpose(0, 2, 1) for a in rho_K_fp32_calc_splits]
    rho_K_fp32_calc_upcast = os_upcast(num_split, rho_K_fp32_calc_splits, out_dtype=np.float64)
    check_error(rho_K_fp64, rho_K_fp32_calc_upcast, name='rho_K_fp32 calc upcast vs ref')

    rho_K_fp32 = np.fromfile('rho_K_fp32.bin', dtype=np.float32)
    rho_K_fp32_splits = buf_to_split_arrays(rho_K_fp32, num_split, (naux_blk, padded_nao, padded_nao))
    rho_K_fp32_upcast = os_upcast(num_split, rho_K_fp32_splits, out_dtype=np.float64)
    check_error(rho_K_fp64, rho_K_fp32_upcast, name='rho_K_fp32 upcast vs ref')

    rho_K_fp16 = np.fromfile('rho_K_fp16.bin', dtype=np.float16)
    rho_K_fp16_splits = buf_to_split_arrays(rho_K_fp16, num_split, (naux_blk, padded_nao, padded_nao))
    rho_K_fp16_upcast = os_upcast(num_split, rho_K_fp16_splits, out_dtype=np.float64)
    check_error(rho_K_fp64, rho_K_fp16_upcast, name='rho_K_fp16 upcast vs ref')

    mat_K_fp64_calc = np.einsum('Pki,Pkj->ij', cderi_fp64, rho_K_fp64)
    mat_K_fp64 = np.fromfile('mat_K_fp64.bin', dtype=np.float64)
    mat_K_fp64 = mat_K_fp64.reshape((nao, nao))
    check_error(mat_K_fp64, mat_K_fp64_calc[0:nao, 0:nao], name='mat_K_fp64 calc vs ref')

    cderi_fp16_splits = [a.reshape(naux_blk * padded_nao, padded_nao) for a in cderi_fp16_splits]
    cderi_fp16_splits = [a.transpose(1, 0) for a in cderi_fp16_splits]
    rho_K_fp16_splits = [a.reshape(naux_blk * padded_nao, padded_nao) for a in rho_K_fp16_splits]
    cderi_fp32_splits = [a.astype(np.float32) for a in cderi_fp16_splits]
    rho_K_fp32_splits = [a.astype(np.float32) for a in rho_K_fp16_splits]
    mat_K_fp32_calc_splits = os_gemm(num_split, cderi_fp32_splits, rho_K_fp32_splits)
    mat_K_fp32_calc_upcast = os_upcast(num_split, mat_K_fp32_calc_splits, out_dtype=np.float64)
    mat_K_fp32_calc_upcast = mat_K_fp32_calc_upcast[0:nao, 0:nao]
    check_error(mat_K_fp64, mat_K_fp32_calc_upcast, name='mat_K_fp32 calc upcast vs ref')

    mat_K_fp32 = np.fromfile('padded_K_fp32.bin', dtype=np.float32)
    mat_K_fp32_splits = buf_to_split_arrays(mat_K_fp32, num_split, (padded_nao, padded_nao))
    mat_K_fp32_upcast = os_upcast(num_split, mat_K_fp32_splits, out_dtype=np.float64)
    mat_K_fp32_upcast = mat_K_fp32_upcast[0:nao, 0:nao]
    check_error(mat_K_fp64, mat_K_fp32_upcast, name='mat_K_fp32 upcast vs ref')

    mat_K_upcast = np.fromfile('mat_K_upcast.bin', dtype=np.float64)
    mat_K_upcast = mat_K_upcast.reshape((nao, nao))
    check_error(mat_K_fp64, mat_K_upcast, name='mat_K_upcast vs ref')
