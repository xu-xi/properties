#!/usr/bin/env python

import unittest
from pyscf import gto, lib, scf
from pyscf.prop.infrared.rhf import kernel_dipderiv, Infrared
from pyscf.prop.infrared.dipole_grad import dipole_grad

class KnownValues(unittest.TestCase):
    def test_grad_fd(self):
        mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz')
        mf = scf.RHF(mol)
        mf.run()

        de2 = dipole_grad(mf)

        mf_ir = Infrared(mf).run()
        de1 = kernel_dipderiv(mf_ir)


        mol1 = gto.M(atom='H 0 0 -0.001; F 0 0 0.9', basis='ccpvdz')
        mf1 = scf.RHF(mol1)
        mf1.scf()

        mol2 = gto.M(atom='H 0 0 0.001; F 0 0 0.9', basis='ccpvdz')
        mf2 = scf.RHF(mol2)
        mf2.scf()

        de = (scf.hf.dip_moment(mol2, mf2.make_rdm1(), unit='au')[-1] - scf.hf.dip_moment(mol1, mf1.make_rdm1(), unit='au')[-1])/0.002*lib.param.BOHR
        self.assertAlmostEqual(de1[0,-1,-1], de, 5)
        self.assertAlmostEqual(de2[0,-1,-1], de, 5)

if __name__ == "__main__":
    print("Full Tests for gradients of molecular dipole moments")
    unittest.main()