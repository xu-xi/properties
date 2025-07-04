#!/usr/bin/env python

from functools import reduce
import numpy
from pyscf import lib, dft, grad, scf
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


def dipole_grad(mf):
    '''
    Analytical nuclear gradients of dipole moments

    Ref: J. Chem. Phys. 84, 2262-2278 (1986)
    '''
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
        int1e_r =  mol.intor_symmetric('int1e_r', comp=3)

    h1vo = numpy.einsum('xuv, uj, vi-> xji', int1e_r, mo_coeff, mo_coeff[:,mo_occ>0])
    s1 = numpy.zeros_like(h1vo)

    fx = gen_vind(mf, mo_coeff, mo_occ)

    # solve CPHF
    mo1, e1 = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1, verbose=mf.verbose,
                            max_cycle=30, level_shift=mf.level_shift)

    with mol.with_common_orig(charge_center): 
        int1e_irp = - mol.intor("int1e_irp", comp=9)

    s1a = - mol.intor('int1e_ipovlp')

    mf_hess = mf.Hessian()
    h1ao = mf_hess.make_h1(mo_coeff, mo_occ) # 1st order skeleton derivative of Fock matrix

    for a in range(natm):
        p0, p1 = mol.aoslice_by_atom()[a, 2:]
        
        h2ao = numpy.zeros((9, nao, nao))
        h2ao[:,:,p0:p1] += int1e_irp[:,:,p0:p1] # nable is on ket in int1e_irp
        h2ao[:,p0:p1] += int1e_irp[:,:,p0:p1].transpose(0, 2, 1)
        de[a] -= numpy.einsum('xuv,uv->x', h2ao, dm).reshape(3, 3).T

        h1vo = numpy.einsum('xuv, ui, vj -> xij', h1ao[a], mo_coeff[:,mo_occ>0], mo_coeff)
        de[a] -= 4 * numpy.einsum('xij,tji->xt', h1vo, mo1)

        s1ao = numpy.zeros((3, nao, nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        s1ii = numpy.einsum('ui, vj, xuv -> xij', mo_coeff[:,mo_occ>0], mo_coeff[:,mo_occ>0], s1ao)
        de[a] += 2*numpy.einsum('xij, tij -> xt', s1ii, e1) 

        s1ij = numpy.einsum('ui, vj, xuv -> xij', mo_coeff[:,mo_occ>0], mo_coeff, s1ao)
        de[a] += 4*numpy.einsum('i, tji, xij -> xt', mo_energy[mo_occ>0], mo1, s1ij)

    return de

class SCFwithEfield(dft.rks.RKS):
    ' SCF with external electric field '
    _keys = {'efield'}

    def __init__(self, mol, **kwargs):
        dft.rks.RKS.__init__(self, mol, **kwargs)
        self.efield = numpy.array([0, 0, 0]) # unit: a.u. ( 1 a.u. = 5.14e11 V/m ? )
        self.mol = mol


    def get_hcore(self, mol):

        with mol.with_common_orig([0, 0, 0]):
            h = numpy.einsum('x,xij->ij', self.efield, mol.intor('int1e_r', comp=3))

        h += super().get_hcore(mol)

        return h 
    
    def energy_nuc(self):        
        charges = self.mol.atom_charges()  
        coords = self.mol.atom_coords()    

        E_nuc_field = -numpy.sum([Z * numpy.dot(self.efield, R) for Z, R in zip(charges, coords)])

        return self.mol.enuc + E_nuc_field
    
    def nuc_grad_method(self):
        return GradwithEfield(self)
    

class GradwithEfield(grad.rks.Gradients):
    ' Gradients with external electric field '
    _keys = {'mf'}
    def __init__(self, mf):
        grad.rks.Gradients.__init__(self, mf)

        self.mf = self.base = mf
        self._efield = mf.efield

        mol = self.mol
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        self._charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mf.mol
        nao = mol.nao

        with mol.with_common_orig([0, 0, 0]):
            int1e_irp = - mol.intor("int1e_irp", comp=9).reshape(3, 3, nao, nao)

        h = super().get_hcore(mol)
        h += numpy.einsum('z,zxij->xji', self._efield, int1e_irp) # nable is on ket in int1e_irp

        return h
    
    def grad_nuc(self, atmlst=None):
        gs = super().grad_nuc(atmlst)
        charges = self.mf.mol.atom_charges()
        gs -= numpy.einsum('i,x->ix', charges, self._efield)
        if atmlst is not None:
            gs = gs[atmlst]
        return gs



SCFwithEfield.Gradients = lib.class_as_method(GradwithEfield)
    
if __name__ == '__main__':
    from pyscf import gto
    #mol = gto.M(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="ccpvdz")
    mol = gto.M(atom='H 0 0 0; F 0 0 0.8', basis='ccpvdz')

    mf = SCFwithEfield(mol)
    mf.efield = numpy.array([0, 0, 0.01])
    mf.run()

    grad = mf.Gradients()
    grad.grid_response = True
    g = grad.kernel()