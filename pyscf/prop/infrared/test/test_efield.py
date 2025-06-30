#!/usr/bin/env python

import numpy
import unittest
from pyscf import gto, lib, scf, dft
from pyscf.prop.infrared.rhf import kernel_dipderiv
from pyscf.prop.infrared.rks import Infrared
from pyscf.prop.infrared.efield import dipole_grad, GradwithEfield, SCFwithEfield
from pyscf.prop.polarizability.rhf import Polarizability

class KnownValues(unittest.TestCase):
    def test_dipole_grad(self):
        mol = gto.M(atom='''O 0.0000 0.0000 0.0000;
                    H 0.75740 0.58680 0.0000;
                    H -0.75740 0.58680 0.0000;
                    ''', basis='ccpvdz')
        mf = dft.RKS(mol, xc='b3lyp')
        mf.run()

        de1 = dipole_grad(mf)
        print(de1)

        mf_ir = Infrared(mf).run()
        de2 = kernel_dipderiv(mf_ir)
        print(de2)

        self.assertAlmostEqual(abs(de1 - de2).max(), 0, 7)

        mol1 = gto.M(atom='''O 0.0000 0.0000 0.0000;
                    H 0.75740 0.58580 0.0000;
                    H -0.75740 0.58680 0.0000;
                    ''', basis='ccpvdz')
        mf1 = dft.RKS(mol1, xc='b3lyp')
        mf1.scf()

        mol2 = gto.M(atom='''O 0.0000 0.0000 0.0000;
                    H 0.75740 0.58780 0.0000;
                    H -0.75740 0.58680 0.0000;
                    ''', basis='ccpvdz')
        mf2 = dft.RKS(mol2, xc='b3lyp')
        mf2.scf()

        de_fd = (mf2.dip_moment(unit='au') - mf1.dip_moment(unit='au'))/0.002*lib.param.BOHR
        print(de_fd)

        
        self.assertAlmostEqual(abs(de1[1, 1] - de_fd).max(), 0, 4)
        self.assertAlmostEqual(abs(de2[1, 1] - de_fd).max(), 0, 4)


    def test_scf_with_efield(self):
        mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz')
        mf0 = dft.RKS(mol)
        mf0.scf()

        mf1 = SCFwithEfield(mol)
        mf1.efield = numpy.array([0, 0, 0.001])
        e1 = mf1.scf()

        mf2 = SCFwithEfield(mol)
        mf2.efield = numpy.array([0, 0, -0.001])
        e2 = mf2.scf()

        dipole = (e2 - e1) / 0.002
        self.assertAlmostEqual(dipole, scf.hf.dip_moment(mol, mf0.make_rdm1(), unit='au')[-1], 5)

        dipole1 = scf.hf.dip_moment(mol, mf1.make_rdm1(), unit='au')[-1]
        dipole2 = scf.hf.dip_moment(mol, mf2.make_rdm1(), unit='au')[-1]

        polar = Polarizability(mf0).polarizability()
        self.assertAlmostEqual(polar[-1,-1], (dipole1 - dipole2) / 0.002, 5)



    def test_grad_with_efield(self):
        mol = gto.M(atom='''O 0.0000 0.0000 0.0000;
                    H 0.75740 0.58680 0.0000;
                    H -0.75740 0.58680 0.0000;
                    ''', basis='ccpvdz')
        mf0 = SCFwithEfield(mol)
        mf0.efield = numpy.array([0, 0.01, 0])
        mf0.scf()
        grad = GradwithEfield(mf0)
        grad.grid_response = True
        de = grad.kernel()

        mol1 = gto.M(atom='''O 0.0000 0.0000 0.0000;
                    H 0.75740 0.58580 0.0000;
                    H -0.75740 0.58680 0.0000;
                    ''', basis='ccpvdz')
        mf1 = SCFwithEfield(mol1)
        mf1.efield = numpy.array([0, 0.01, 0])

        e1 = mf1.scf()

        mol2 = gto.M(atom='''O 0.0000 0.0000 0.0000;
                    H 0.75740 0.58780 0.0000;
                    H -0.75740 0.58680 0.0000;
                    ''', basis='ccpvdz')
        mf2 = SCFwithEfield(mol2)
        mf2.efield = numpy.array([0, 0.01, 0])

        e2 = mf2.scf()

        de_fd = (e2 - e1) / 0.002*lib.param.BOHR
        self.assertAlmostEqual(de[1, 1], de_fd, 6)


if __name__ == "__main__":
    print("Full Tests for efield")
    unittest.main()