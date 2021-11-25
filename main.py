from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit.visualization import (
    plot_histogram,
    plot_state_qsphere,
    plot_bloch_multivector,
    plot_bloch_vector,
)
def multiPhase(qc, q, theta):
    # multi-qubit controlled phase rotation
    # applies a phase factor exp(i*theta) if all the qubits are 1.
    # Note that it doesn't matter which qubits are the controls and which is the target.
    # qc is a quantum circuit
    # q is a quantum register in qc
    # theta is a float
    n = len(q)
    q = [q[i] for i in range(n)]
    if n == 1:
        qc.u1(theta, q[0])
    elif n == 2:
        qc.cu1(theta, q[0], q[1])
    else:
        qc.cu1(theta / 2, q[1], q[0])
        multiCX(qc, q[2:], q[1])
        qc.cu1(-theta / 2, q[1], q[0])
        multiCX(qc, q[2:], q[1])
        multiPhase(qc, [q[0]] + q[2:], theta / 2)

    return
    
def multiCX(qc, q_controls, q_target, sig=None):
    # multi-qubit controlled X gate
    # applies an X gate to q_target if q_controls[i]==sig[i] for all i
    # qc is a quantum circuit
    # q_controls is the quantum register of the controlling qubits
    # q_target is the quantum register of the target qubit
    # sig is the signature of the control (defaults to sig=[1,1,...,1])

    # default signature is an array of n 1s
    n = len(q_controls)
    if sig is None:
        sig = [1] * n

    # use the fact that H Z H = X to realize a controlled X gate
    qc.h(q_target)
    multiCZ(qc, q_controls, q_target, sig)
    qc.h(q_target)

    return


def multiCZ(qc, q_controls, q_target, sig=None):
    # multi-qubit controlled Z gate
    # applies a Z gate to q_target if q_controls[i]==sig[i] for all i
    # qc is a quantum circuit
    # q_controls is the quantum register of the controlling qubits
    # q_target is the quantum register of the target qubit
    # sig is the signature of the control (defaults to sig=[1,1,...,1])

    # default signature is an array of n 1s
    n = len(q_controls)
    if sig is None:
        sig = [1] * n

    # apply signature
    for i in range(n):
        if sig[i] == 0:
            qc.x(q_controls[i])

    q = [q_controls[i] for i in range(len(q_controls))] + [q_target]
    multiPhase(qc, q, np.pi)

    # undo signature gates
    for i in range(n):
        if sig[i] == 0:
            qc.x(q_controls[i])

    return

def GroverDiffusion():
    W = QuantumCircuit(n)
    for i in range(n):
        W.h(i)
        W.x(i)
    multiPhase(W, range(n), np.pi)
    for i in range(n):
        W.x(i)
        W.h(i)

    return W

n = 5
N = 2 ** n
R = int(np.floor(np.pi * np.sqrt(N) / 4))
x = QuantumRegister(n)
g = QuantumRegister(1)
c = QuantumRegister(2)
output = QuantumRegister(1)
qc = QuantumCircuit(x, g, c, output)


def GroverSearch(s):
    assert s >= 0 and s < N, "Invalid Search Parameter"
    qc = QuantumCircuit(n)
    sig = ("{0:b}".format(s))[::-1]
    W = GroverDiffusion()

    qc.x(output[0])
    qc.h(output[0])

    for i in range(R):
        multiCX(qc, x, output, sig)  # unitary
        qc.append(W, x)  # W

    qc.h(output[0])
    qc.x(output[0])
    qc.barrier()

    for i in range(n): 
        qc.measure(i + 1, i)




