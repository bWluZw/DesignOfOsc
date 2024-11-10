import pyscf
from pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = rks.RKS(mol, xc='LDA').density_fit().to_gpu()  # move PySCF object to GPU4PySCF object
e_dft = mf.kernel()  # compute total energy

print(mf.mo_energy[mf.mo_occ > 0][-1])