#!/usr/bin/env python

from functools import reduce
import numpy
from pyscf import lib
from pyscf.scf import cphf

def gen_vind(mf, mo_coeff, mo_occ):
    '''Induced potential'''
    vresp = mf.gen_response(hermi=1)
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    nocc = orbo.shape[1]
    nao, nmo = mo_coeff.shape
    def vind(mo1):
        dm1 = lib.einsum('xai,pa,qi->xpq', mo1.reshape(-1,nmo,nocc), mo_coeff,
                             orbo.conj())
        dm1 = (dm1 + dm1.transpose(0,2,1).conj()) * 2
        v1mo = lib.einsum('xpq,pi,qj->xij', vresp(dm1), mo_coeff.conj(), orbo)
        return v1mo.ravel()
    return vind

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao,
              fx=None, atmlst=None, max_memory=4000, verbose=None,
              max_cycle=50, level_shift=0):
    '''Solve the first order equation

    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind.
    '''

    #h1vo = numpy.einsum('xuv, uj, vi-> xji', h1ao, mo_coeff[:,mo_occ==0], mo_coeff[:,mo_occ>0])
    h1vo = numpy.einsum('xuv, uj, vi-> xji', h1ao, mo_coeff, mo_coeff[:,mo_occ>0])
    s1 = numpy.zeros_like(h1vo)

    fx = gen_vind(mf, mo_coeff, mo_occ)

    # mo1 shape: (3,nvir,nocc)
    mo1, _ = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1, verbose=mf.verbose,
                            max_cycle=max_cycle, level_shift=level_shift)
    
    e1 = numpy.einsum('xuv, ui, vi -> xi', h1ao, mo_coeff[:,mo_occ>0], mo_coeff[:,mo_occ>0])
    
    print(mo1[:,mo_occ>0])

    return mo1, e1

def dipole_grad(mf):
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    natm = mf.mol.natm
    nao = mol.nao
    dm = mf.make_rdm1()

    de = numpy.zeros((natm, 3, 3))

    # contribution from nuclei
    for i in range(natm): 
        de[i] = numpy.eye(3) * mol.atom_charge(i) 

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center): 
        h1ao =  mol.intor_symmetric('int1e_r', comp=3)

    # solve CPHF    
    mo1, e1 = solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao)
    
    # contribution from core hamiltonian
    
    h2ao = numpy.zeros((natm, 3, 3, nao, nao))
    with mol.with_common_orig(charge_center): 
        int1e_irp =  mol.intor("int1e_irp").reshape(3, 3, nao, nao).swapaxes(0, 1)

    s1a = - mol.intor('int1e_ipovlp')

    mf_grad = mf.nuc_grad_method()
    mf_hess = mf.Hessian()

    h1ao = mf_hess.make_h1(mo_coeff, mo_occ) # 1st order derivative of Fock matrix

    vhf = mf_grad.get_veff(mol, dm)

    for a in range(natm):
        p0, p1 = mol.aoslice_by_atom()[a, 2:]
        h2ao[a, :, :, :, p0:p1] = int1e_irp[:, :, :, p0:p1]

        #h1ao = hcore_deriv(a)
        h1vo = numpy.einsum('xuv, ui, vj -> xij', h1ao[a], mo_coeff[:,mo_occ>0], mo_coeff)
        de[a] -= 4 * numpy.einsum('xij,tji->xt', h1vo, mo1)

        v1 = numpy.zeros((3, nao, nao))
        v1[:,p0:p1] += vhf[:,p0:p1]
        v1[:,:,p0:p1] += vhf[:,p0:p1].transpose(0,2,1)
        vhf_vo = numpy.einsum('xuv, ui, vj -> xij', v1, mo_coeff[:,mo_occ>0], mo_coeff)

        #de[a] -= 4 * numpy.einsum('xij,tji->xt', vhf_vo, mo1)

        s1ao = numpy.zeros((3, nao, nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        s1ii = numpy.einsum('ui, vi, xuv -> xi', mo_coeff[:,mo_occ>0], mo_coeff[:,mo_occ>0], s1ao)
        de[a] += 2*numpy.einsum('xi, ti -> xt', s1ii, e1) 

        s1ij = numpy.einsum('ui, vj, xuv -> xij', mo_coeff[:,mo_occ>0], mo_coeff, s1ao)
        de[a] += 4*numpy.einsum('i, tji, xij -> xt', mo_energy[mo_occ>0], mo1, s1ij)

    h2ao += h2ao.swapaxes(-1, -2)
    de += numpy.einsum("Axtuv, uv -> Axt", h2ao, dm)

    return de

    
if __name__ == '__main__':
    from pyscf import gto, scf
    #mol = gto.M(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="ccpvdz")
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz')

    mf = scf.RHF(mol).run()
 
    de = dipole_grad(mf)
    print(de)

