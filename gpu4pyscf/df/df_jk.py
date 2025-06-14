# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by Xiaojie Wu <wxj6000@gmail.com>

import copy
from concurrent.futures import ThreadPoolExecutor
import cupy
import numpy
from pyscf import lib, __config__
from pyscf.scf import dhf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, transpose_sum, reduce_to_device
from gpu4pyscf.dft import rks, uks, numint
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.df import df, int3c2e
from gpu4pyscf.__config__ import _streams, num_devices
from gpu4pyscf.mxp_df_helper.helper import (
    OzakiSchemeHelper,
    mxp_df_level_to_split_dtype,
    ozaki_scheme_gemm,
    get_gemm_padding,
)
import pdb
from gpu4pyscf.lib.mxp_df_helper import DF_K_build

def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = numpy.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def _density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding density fitting integrals.
    Args:
        mf : an SCF object
    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, optimal auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        only_dfj : str
            Compute Coulomb integrals only and no approximation for HF
            exchange. Same to RIJONX in ORCA
    Returns:
        An SCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K
    Examples:
    '''

    assert isinstance(mf, hf.SCF)

    if with_df is None:
        if isinstance(mf, dhf.UHF):
            with_df = df.DF4C(mf.mol)
        else:
            with_df = df.DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    if isinstance(mf, _DFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            #logger.warn(mf, 'DF might have been initialized twice.')
            mf = mf.copy()
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf

    dfmf = _DFHF(mf, with_df, only_dfj)
    return lib.set_class(dfmf, (_DFHF, mf.__class__))

from gpu4pyscf.lib import utils
class _DFHF:
    '''
    Density fitting SCF class
    Attributes for density-fitting SCF:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        with_df : DF object
            Set mf.with_df = None to switch off density fitting mode.
    '''
    to_gpu = utils.to_gpu
    device = utils.device
    __name_mixin__ = 'DF'
    _keys = {'rhoj', 'rhok', 'disp', 'screen_tol', 'with_df', 'only_dfj'}

    def __init__(self, mf, dfobj, only_dfj):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.rhoj = None
        self.rhok = None
        self.direct_scf = False
        self.with_df = dfobj
        self.only_dfj = only_dfj

    def undo_df(self):
        '''Remove the DFHF Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFHF))
        del obj.rhoj, obj.rhok, obj.with_df, obj.only_dfj
        return obj

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None: dm = self.make_rdm1()
        if self.with_df and self.only_dfj:
            vj = vk = None
            if with_j:
                vj, vk = self.with_df.get_jk(dm, hermi, True, False,
                                             self.direct_scf_tol, omega)
            if with_k:
                vk = super().get_jk(mol, dm, hermi, False, True, omega)[1]
        elif self.with_df:
            vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_k,
                                         self.direct_scf_tol, omega, self.mxp_df_level)
        else:
            vj, vk = super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        return vj, vk

    def nuc_grad_method(self):
        if self.istype('_Solvation'):
            raise NotImplementedError(
                'Gradients of solvent are not computed. '
                'Solvent must be applied after density fitting method, e.g.\n'
                'mf = mol.RKS().to_gpu().density_fit().PCM()')
        if isinstance(self, rks.RKS):
            from gpu4pyscf.df.grad import rks as rks_grad
            return rks_grad.Gradients(self)
        if isinstance(self, hf.RHF):
            from gpu4pyscf.df.grad import rhf as rhf_grad
            return rhf_grad.Gradients(self)
        if isinstance(self, uks.UKS):
            from gpu4pyscf.df.grad import uks as uks_grad
            return uks_grad.Gradients(self)
        if isinstance(self, uhf.UHF):
            from gpu4pyscf.df.grad import uhf as uhf_grad
            return uhf_grad.Gradients(self)
        raise NotImplementedError()

    Gradients = nuc_grad_method

    def Hessian(self):
        if self.istype('_Solvation'):
            raise NotImplementedError(
                'Hessian of solvent are not computed. '
                'Solvent must be applied after density fitting method, e.g.\n'
                'mf = mol.RKS().to_gpu().density_fit().PCM()')
        from gpu4pyscf.dft.rks import KohnShamDFT
        if isinstance(self, hf.RHF):
            if isinstance(self, KohnShamDFT):
                from gpu4pyscf.df.hessian import rks as rks_hess
                return rks_hess.Hessian(self)
            else:
                from gpu4pyscf.df.hessian import rhf as rhf_hess
                return rhf_hess.Hessian(self)
        elif isinstance(self, uhf.UHF):
            if isinstance(self, KohnShamDFT):
                from gpu4pyscf.df.hessian import uks as uks_hess
                return uks_hess.Hessian(self)
            else:
                from gpu4pyscf.df.hessian import uhf as uhf_hess
                return uhf_hess.Hessian(self)
        else:
            raise NotImplementedError

    @property
    def auxbasis(self):
        return getattr(self.with_df, 'auxbasis', None)

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        '''
        effective potential
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        # for DFT
        if isinstance(self, rks.KohnShamDFT):
            if dm.ndim == 2:
                return rks.get_veff(self, dm=dm)
            elif dm.ndim == 3:
                return uks.get_veff(self, dm=dm)

        if dm.ndim == 2:
            if self.direct_scf:
                ddm = cupy.asarray(dm) - dm_last
                vj, vk = self.get_jk(mol, ddm, hermi=hermi)
                return vhf_last + vj - vk * .5
            else:
                vj, vk = self.get_jk(mol, dm, hermi=hermi)
                return vj - vk * .5
        elif dm.ndim == 3:
            if self.direct_scf:
                ddm = cupy.asarray(dm) - dm_last
                vj, vk = self.get_jk(mol, ddm, hermi=hermi)
                vhf = vj[0] + vj[1] - vk
                vhf += cupy.asarray(vhf_last)
                return vhf
            else:
                vj, vk = self.get_jk(mol, dm, hermi=hermi)
                return vj[0] + vj[1] - vk
        else:
            raise NotImplementedError("Please check the dimension of the density matrix, it should not reach here.")

    def to_cpu(self):
        obj = self.undo_df().to_cpu().density_fit()
        return utils.to_cpu(self, obj)

def _jk_task_with_mo(dfobj, dms, mo_coeff, mo_occ,
                     with_j=True, with_k=True, hermi=0, device_id=0, mxp_df_level=0):
    ''' Calculate J and K matrices on single GPU
    '''
    with cupy.cuda.Device(device_id), _streams[device_id]:
        assert isinstance(dfobj.verbose, int)
        log = logger.new_logger(dfobj.mol, dfobj.verbose)
        t0 = log.init_timer()
        dms = cupy.asarray(dms)
        mo_coeff = cupy.asarray(mo_coeff)
        mo_occ = cupy.asarray(mo_occ)
        nao = dms.shape[-1]
        intopt = dfobj.intopt
        rows = intopt.cderi_row
        cols = intopt.cderi_col
        nset = dms.shape[0]
        dms_shape = dms.shape
        vj = vk = None
        if with_j:
            dm_sparse = dms[:,rows,cols]
            if hermi == 0:
                dm_sparse += dms[:,cols,rows]
            else:
                dm_sparse *= 2
            dm_sparse[:, intopt.cderi_diag] *= .5

        if with_k:
            vk = cupy.zeros_like(dms)

        num_split, split_dtype = mxp_df_level_to_split_dtype(mxp_df_level)
        padded_nao = get_gemm_padding(nao)

        # SCF K matrix with occ
        if mo_coeff is not None:
            assert hermi == 1
            nocc = 0
            occ_coeff = [0]*nset
            for i in range(nset):
                occ_idx = mo_occ[i] > 0
                occ_coeff[i] = mo_coeff[i][:,occ_idx] * mo_occ[i][occ_idx]**0.5
                nocc += mo_occ[i].sum()
            if with_k:
                occ_coeff_os = []
                for i in range(nset):
                    nocc = occ_coeff[i].shape[1]
                    padded_nocc = get_gemm_padding(nocc)
                    occ_coeff_i = cupy.zeros([padded_nao, padded_nocc], dtype=occ_coeff[i].dtype)
                    occ_coeff_i[0:nao, 0:nocc] = occ_coeff[i]
                    occ_coeff_os_i = OzakiSchemeHelper(cupy.float64)
                    occ_coeff_os_i.split(num_split, split_dtype, occ_coeff_i)
                    occ_coeff_os.append(occ_coeff_os_i)
            blksize = dfobj.get_blksize(extra=padded_nao*nocc)
            if with_j:
                vj_packed = cupy.zeros_like(dm_sparse)
            for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k, mxp_df_level=mxp_df_level):
                # leading dimension is 1
                if with_j:
                    rhoj = dm_sparse.dot(cderi_sparse)
                    vj_packed += cupy.dot(rhoj, cderi_sparse.T)
                cderi_sparse = rhoj = None
                if with_k:
                    for i in range(nset):
                        padded_nocc = occ_coeff_os[i].split_tensors[0].shape[1]
                        if mxp_df_level > 0:
                            cderi_os = cderi
                            # Except for the last split, curr_bs is always a multiplier of 128, no need to worry about
                            curr_bs = cderi_os.split_tensors[0].shape[0]

                            vk_i0 = cupy.copy(vk[i])

                            #pdb.set_trace()
                            # Lij,jk -> Lik
                            cderi_os.reshape([-1, padded_nao])
                            rhok_os = ozaki_scheme_gemm(cderi_os, occ_coeff_os[i], num_split, 'NN')
                            rhok_os.reshape([curr_bs, padded_nao, padded_nocc])
                            # Lik -> Lki
                            rhok_os.transpose((0, 2, 1))
                            rhok_os.reshape([-1, padded_nao])
                            # Lki,Lkj -> ij
                            vk_i_os = ozaki_scheme_gemm(rhok_os, rhok_os, num_split, 'TN')
                            vk_i = vk_i_os.upcast()
                            vk[i] += vk_i[0:nao, 0:nao]
                            
                            """
                            DF_K_build(
                                0, curr_bs, nao, padded_nao, padded_nocc, num_split, 
                                cderi_os.split_tensors, occ_coeff_os[i].split_tensors, vk_i0
                            )
                            pdb.set_trace()
                            """
                        else:
                            curr_bs = cderi.shape[0]
                            cderi = cderi.reshape([-1, padded_nao])
                            occ_coeff_i = occ_coeff_os[i].split_tensors[0]
                            rhok = cupy.dot(cderi, occ_coeff_i)             # Lij,jk -> Lik
                            rhok = rhok.reshape([curr_bs, padded_nao, padded_nocc])
                            rhok = cupy.transpose(rhok, axes=(0, 2, 1))     # Lik -> Lki
                            rhok = rhok.reshape([-1, padded_nao])
                            vk_i = cupy.dot(rhok.T, rhok)                   # Lki,Lkj -> ij
                            vk[i] += vk_i[0:nao, 0:nao]
                    rhok = None

            if with_j:
                vj = cupy.zeros(dms_shape)
                vj[:,rows,cols] = vj_packed
                vj[:,cols,rows] = vj_packed
        t0 = log.timer_debug1(f'vj and vk on Device {device_id}', *t0)
    return vj, vk

def _jk_task_with_mo1(dfobj, dms, mo1s, occ_coeffs,
                      with_j=True, with_k=True, hermi=0, device_id=0):
    ''' Calculate J and K matrices with mo response
        For CP-HF or TDDFT
    '''
    vj = vk = None
    with cupy.cuda.Device(device_id), _streams[device_id]:
        assert isinstance(dfobj.verbose, int)
        log = logger.new_logger(dfobj.mol, dfobj.verbose)
        t0 = log.init_timer()
        dms = cupy.asarray(dms)
        mo1s = [cupy.asarray(mo1) for mo1 in mo1s]
        occ_coeffs = [cupy.asarray(occ_coeff) for occ_coeff in occ_coeffs]

        nao = dms.shape[-1]
        intopt = dfobj.intopt
        rows = intopt.cderi_row
        cols = intopt.cderi_col
        dms_shape = dms.shape
        if with_j:
            dm_sparse = dms[:,rows,cols]
            if hermi == 0:
                dm_sparse += dms[:,cols,rows]
            else:
                dm_sparse *= 2
            dm_sparse[:, intopt.cderi_diag] *= .5

        if with_k:
            vk = cupy.zeros_like(dms)

        if with_j:
            vj_sparse = cupy.zeros_like(dm_sparse)

        nocc = max([mo1.shape[2] for mo1 in mo1s])
        blksize = dfobj.get_blksize(extra=2*nao*nocc)
        for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k):
            if with_j:
                rhoj = dm_sparse.dot(cderi_sparse)
                vj_sparse += cupy.dot(rhoj, cderi_sparse.T)
                rhoj = None
            cderi_sparse = None
            if with_k:
                iset = 0
                for occ_coeff, mo1 in zip(occ_coeffs, mo1s):
                    rhok = contract('Lij,jk->Lki', cderi, occ_coeff).reshape([-1,nao])
                    for i in range(mo1.shape[0]):
                        rhok1 = contract('Lij,jk->Lki', cderi, mo1[i]).reshape([-1,nao])
                        #contract('Lki,Lkj->ij', rhok1, rhok, alpha=1.0, beta=1.0, out=vk[iset])
                        vk[iset] += cupy.dot(rhok1.T, rhok)
                        iset += 1
                mo1 = rhok1 = rhok = None
            cderi = None
        mo1s = None
        if with_j:
            vj = cupy.zeros(dms_shape)
            vj[:,rows,cols] = vj_sparse
            vj[:,cols,rows] = vj_sparse
        if with_k and hermi:
            transpose_sum(vk)
        vj_sparse = None

        t0 = log.timer_debug1(f'vj and vk on Device {device_id}', *t0)
    return vj, vk

def _jk_task_with_dm(dfobj, dms, with_j=True, with_k=True, hermi=0, device_id=0, mxp_df_level=0):
    ''' Calculate J and K matrices with density matrix
    '''
    with cupy.cuda.Device(device_id), _streams[device_id]:
        assert isinstance(dfobj.verbose, int)
        log = logger.new_logger(dfobj.mol, dfobj.verbose)
        t0 = log.init_timer()
        dms = cupy.asarray(dms)
        intopt = dfobj.intopt
        rows = intopt.cderi_row
        cols = intopt.cderi_col
        nao = dms.shape[-1]
        dms_shape = dms.shape
        vj = vk = None
        if with_j:
            dm_sparse = dms[:,rows,cols]
            if hermi == 0:
                dm_sparse += dms[:,cols,rows]
            else:
                dm_sparse *= 2
            dm_sparse[:, intopt.cderi_diag] *= .5
            vj_sparse = cupy.zeros_like(dm_sparse)

        if with_k:
            vk = cupy.zeros_like(dms)

        nset = dms.shape[0]

        num_split, split_dtype = mxp_df_level_to_split_dtype(mxp_df_level)
        padded_nao = get_gemm_padding(nao)
        if with_k:
            dms_os = []
            for i in range(nset):
                dms_i = cupy.zeros([padded_nao, padded_nao], dtype=dms[i].dtype)
                dms_i[0:nao, 0:nao] = dms[i]
                dms_os_i = OzakiSchemeHelper(cupy.float64)
                dms_os_i.split(num_split, split_dtype, dms_i)
                dms_os.append(dms_os_i)

        blksize = dfobj.get_blksize()
        for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k, mxp_df_level=mxp_df_level):
            if with_j:
                rhoj = dm_sparse.dot(cderi_sparse)
                vj_sparse += cupy.dot(rhoj, cderi_sparse.T)
            if with_k:
                for k in range(nset):
                    if mxp_df_level > 0:
                        cderi_os = cderi
                        # Except for the last split, curr_bs is always a multiplier of 128, no need to worry about
                        curr_bs = cderi_os.split_tensors[0].shape[0]
                        # Lij,jk -> Lik
                        cderi_os.reshape([-1, padded_nao])
                        rhok_os = ozaki_scheme_gemm(cderi_os, dms_os[i], num_split, 'NN')
                        rhok_os.reshape([curr_bs, padded_nao, padded_nao])
                        # Lik -> Lki
                        rhok_os.transpose((0, 2, 1))
                        rhok_os.reshape([-1, padded_nao])
                        # Lki,Lkj -> ij
                        vk_k_os = ozaki_scheme_gemm(rhok_os, cderi_os, num_split, 'TN')
                        vk_k = vk_k_os.upcast()
                    else:
                        curr_bs = cderi.shape[0]
                        cderi = cderi.reshape([-1, padded_nao])
                        dms_k = dms_os[k].split_tensors[0]
                        rhok = cupy.dot(cderi, dms_k)                   # Lij,jk -> Lik
                        rhok = rhok.reshape([curr_bs, padded_nao, padded_nao])
                        rhok = cupy.transpose(rhok, axes=(0, 2, 1))     # Lik -> Lki
                        rhok = rhok.reshape([-1, padded_nao])
                        vk_k = cupy.dot(rhok.T, cderi)                  # Lki,Lkj -> ij
                    vk[k] += vk_k[0:nao, 0:nao]
        if with_j:
            vj = cupy.zeros(dms_shape)
            vj[:,rows,cols] = vj_sparse
            vj[:,cols,rows] = vj_sparse

        t0 = log.timer_debug1(f'vj and vk on Device {device_id}', *t0)
    return vj, vk

def get_jk(dfobj, dms_tag, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-14, omega=None, mxp_df_level=0):
    '''
    get jk with density fitting
    outputs and input are on the same device
    TODO: separate into three cases: j only, k only, j and k
    '''

    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    out_shape = dms_tag.shape
    out_cupy = isinstance(dms_tag, cupy.ndarray)
    if not isinstance(dms_tag, cupy.ndarray):
        dms_tag = cupy.asarray(dms_tag)

    assert(with_j or with_k)
    if dms_tag is None: logger.error("dm is not given")
    nao = dms_tag.shape[-1]
    t1 = t0 = log.init_timer()
    if dfobj._cderi is None:
        log.debug('Build CDERI ...')
        dfobj.build(direct_scf_tol=direct_scf_tol, omega=omega)
        t1 = log.timer_debug1('init jk', *t0)

    assert nao == dfobj.nao
    intopt = dfobj.intopt

    nao = dms_tag.shape[-1]
    dms = dms_tag.reshape([-1,nao,nao])
    intopt = dfobj.intopt
    dms = intopt.sort_orbitals(dms, axis=[1,2])

    cupy.cuda.get_current_stream().synchronize()
    if getattr(dms_tag, 'mo_coeff', None) is not None:
        mo_occ = dms_tag.mo_occ
        mo_coeff = dms_tag.mo_coeff
        nmo = mo_occ.shape[-1]
        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
        mo_occ   = mo_occ.reshape(-1,nmo)
        mo_coeff = intopt.sort_orbitals(mo_coeff, axis=[1])

        futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for device_id in range(num_devices):
                future = executor.submit(
                    _jk_task_with_mo,
                    dfobj, dms, mo_coeff, mo_occ,
                    hermi=hermi, device_id=device_id,
                    with_j=with_j, with_k=with_k, mxp_df_level=mxp_df_level)
                futures.append(future)

    elif hasattr(dms_tag, 'mo1'):
        occ_coeffs = dms_tag.occ_coeff
        mo1s = dms_tag.mo1
        if not isinstance(occ_coeffs, (tuple, list)):
            # *2 for double occupancy in RHF/RKS
            occ_coeffs = [occ_coeffs * 2.0]
        if not isinstance(mo1s, (tuple, list)):
            mo1s = [mo1s]
        occ_coeffs = [intopt.sort_orbitals(occ_coeff, axis=[0]) for occ_coeff in occ_coeffs]
        mo1s = [intopt.sort_orbitals(mo1, axis=[1]) for mo1 in mo1s]

        futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for device_id in range(num_devices):
                future = executor.submit(
                    _jk_task_with_mo1,
                    dfobj, dms, mo1s, occ_coeffs,
                    hermi=hermi, device_id=device_id,
                    with_j=with_j, with_k=with_k)
                futures.append(future)

    # general K matrix with density matrix
    else:
        futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for device_id in range(num_devices):
                future = executor.submit(
                    _jk_task_with_dm, dfobj, dms,
                    hermi=hermi, device_id=device_id,
                    with_j=with_j, with_k=with_k, mxp_df_level=mxp_df_level)
                futures.append(future)

    vj = vk = None
    if with_j:
        vj = [future.result()[0] for future in futures]
        vj = reduce_to_device(vj, inplace=True)
        vj = intopt.unsort_orbitals(vj, axis=[1,2])
        vj = vj.reshape(out_shape)

    if with_k:
        vk = [future.result()[1] for future in futures]
        vk = reduce_to_device(vk, inplace=True)
        vk = intopt.unsort_orbitals(vk, axis=[1,2])
        vk = vk.reshape(out_shape)

    t1 = log.timer_debug1('vj and vk', *t1)
    if out_cupy:
        return vj, vk
    else:
        if vj is not None:
            vj = vj.get()
        if vk is not None:
            vk = vk.get()
        return vj, vk

def get_j(dfobj, dm, hermi=1, direct_scf_tol=1e-13):
    intopt = getattr(dfobj, 'intopt', None)
    if intopt is None:
        dfobj.build(direct_scf_tol=direct_scf_tol)
        intopt = dfobj.intopt
    j2c = dfobj.j2c
    rhoj = int3c2e.get_j_int3c2e_pass1(intopt, dm)
    if dfobj.cd_low.tag == 'eig':
        rhoj, _, _, _ = cupy.linalg.lstsq(j2c, rhoj)
    else:
        rhoj = cupy.linalg.solve(j2c, rhoj)

    rhoj *= 2.0
    vj = int3c2e.get_j_int3c2e_pass2(intopt, rhoj)
    return vj

density_fit = _densit