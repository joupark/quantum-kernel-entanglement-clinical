"""
verify_kernels_qiskit.py -- checks that the NumPy kernels in
compare_quantum_kernels.py equal fidelity kernels of the corresponding Qiskit
circuits (statevector simulation), on random inputs and input scales.

  fqk_cos2  RY(s x) layer, then CNOT chain  (and the closed form prod cos^2)
  reup2     RY - CNOT chain - RY
  reup3     RY - CNOT chain - RY - CNOT chain - RY
  zz        qiskit.circuit.library.ZZFeatureMap(6, reps=2, entanglement="linear") on s*x
  ctrl2     RY - RY - CNOT chain           (gate-matched control for reup2)
  ctrl3     RY - RY - RY - CNOT chain - CNOT chain   (gate-matched control for reup3)

Usage: python verify_kernels_qiskit.py   (needs qiskit >= 1.0)
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

import compare_quantum_kernels as C


def kernel_from(circuits):
    S = np.array([Statevector(c).data for c in circuits])
    return np.abs(S.conj() @ S.T) ** 2


def ry_cnot(x, s, layers):
    qc = QuantumCircuit(6)
    for l in range(layers):
        for i in range(6):
            qc.ry(s * x[i], i)
        if l < layers - 1 or layers == 1:
            for i in range(5):
                qc.cx(i, i + 1)
    return qc


def matched(x, s, layers):
    qc = QuantumCircuit(6)
    for _ in range(layers):
        for i in range(6):
            qc.ry(s * x[i], i)
    for _ in range(layers - 1):
        for i in range(5):
            qc.cx(i, i + 1)
    return qc


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    for s in (0.01, 0.1, 0.7, 2.0):
        X = rng.normal(size=(8, 6))
        fm = ZZFeatureMap(6, reps=2, entanglement="linear")
        checks = {
            "fqk_cos2": kernel_from([ry_cnot(x, s, 1) for x in X]),
            "reup2": kernel_from([ry_cnot(x, s, 2) for x in X]),
            "reup3": kernel_from([ry_cnot(x, s, 3) for x in X]),
            "zz": kernel_from([fm.assign_parameters(s * x) for x in X]),
            "ctrl2": kernel_from([matched(x, s, 2) for x in X]),
            "ctrl3": kernel_from([matched(x, s, 3) for x in X]),
        }
        for k, Kq in checks.items():
            e = np.abs(Kq - C.gram(k, X, X, s)).max()
            worst = max(worst, e)
            print(f"s={s:<5} {k:9s} max |K_qiskit - K_numpy| = {e:.1e}")
        cos2 = np.prod(np.cos(s * (X[:, None] - X[None]) / 2) ** 2, -1)
        e = np.abs(checks["fqk_cos2"] - cos2).max(); worst = max(worst, e)
        print(f"s={s:<5} {'closed form':9s} max |K_qiskit - prod cos^2| = {e:.1e}")
    print("ALL KERNELS MATCH QISKIT" if worst < 1e-12 else "MISMATCH")


if __name__ == "__main__":
    main()
