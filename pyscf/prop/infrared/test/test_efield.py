#!/usr/bin/env python

import numpy
import unittest
from pyscf import gto, lib, scf, dft
from pyscf.prop.infrared.rhf import kernel_dipderiv, Infrared
from pyscf.prop.infrared.efield import dipole_grad, GradWithEfield, SCFWithEfield

class KnownValues(unittest.TestCase):
    def test_dipole_grad(self):
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


    def test_scf_with_efield(self):
        mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz')
        mf0 = dft.RKS(mol)
        mf0.scf()

        mf1 = SCFWithEfield(mol)
        mf1.efield = numpy.array([0, 0, 0.001])
        e1 = mf1.scf()

        mf2 = SCFWithEfield(mol)
        mf2.efield = numpy.array([0, 0, -0.001])
        e2 = mf2.scf()

        dipole = (e2 - e1) / 0.002
        self.assertAlmostEqual(dipole, scf.hf.dip_moment(mol, mf0.make_rdm1(), unit='au')[-1], 5)


    def test_grad_with_efield(self):
        mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz')
        mf0 = dft.RKS(mol)
        mf0.efield = numpy.array([0, 0, 0.001])
        mf0.scf()
        grad = GradWithEfield(mf0)
        de = grad.kernel()

        mol1 = gto.M(atom='H 0 0 -0.001; F 0 0 0.9', basis='ccpvdz')
        mf1 = SCFWithEfield(mol1)
        mf1.efield = numpy.array([0, 0, 0.001])
        e1 = mf1.scf()

        mol2 = gto.M(atom='H 0 0 0.001; F 0 0 0.9', basis='ccpvdz')
        mf2 = SCFWithEfield(mol2)
        mf2.efield = numpy.array([0, 0, 0.001])
        e2 = mf2.scf()

        de_fd = (e2 - e1) / 0.002*lib.param.BOHR
        self.assertAlmostEqual(de[0, -1], de_fd, 5)


if __name__ == "__main__":
    print("Full Tests for efield")
    unittest.main()