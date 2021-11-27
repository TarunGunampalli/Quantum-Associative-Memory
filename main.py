from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit.visualization import (
    plot_histogram,
    plot_state_qsphere,
    plot_bloch_multivector,
    plot_bloch_vector,
)

n = 2
N = 2 ** n
R = int(np.floor(np.pi * np.sqrt(N) / 4))
x = QuantumRegister(n)
g = QuantumRegister(1)
c = QuantumRegister(2)
xc = ClassicalRegister(n)
gc = ClassicalRegister(1)
cc = ClassicalRegister(2)
output = QuantumRegister(1)
qc = QuantumCircuit(x, g, c, output, xc, gc, cc)


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
        qc.cp(theta, q[0], q[1])
    else:
        qc.cp(theta / 2, q[1], q[0])
        multiCX(qc, q[2:], q[1])
        qc.cp(-theta / 2, q[1], q[0])
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
    # sig is the signature of the control (defaults to sig="11...1")

    # default signature is a string of 1s
    n = len(q_controls)
    if sig is None:
        sig = "1" * n

    # apply signature
    for i in range(n):
        if sig[i] == "0":
            qc.x(q_controls[i])

    q = [q_controls[i] for i in range(len(q_controls))] + [q_target]
    multiPhase(qc, q, np.pi)

    # undo signature gates
    for i in range(n):
        if sig[i] == "0":
            qc.x(q_controls[i])

    return


def GroverDiffusion():
    for i in range(n):
        qc.h(x[i])
        qc.x(x[i])
    multiPhase(qc, x, np.pi)
    for i in range(n):
        qc.x(x[i])
        qc.h(x[i])


def Flip(patterns, index):
    prevPattern = ""
    if index == 0:
        prevPattern = "0" * n
    else:
        prevPattern = patterns[index - 1]

    pattern = patterns[index]
    pattern = pattern[::-1]
    prevPattern = prevPattern[::-1]
    for i in range(n):
        if pattern[i] != prevPattern[i]:
            multiCX(qc, c, x[i], "00")

    multiCX(qc, x, c[1], pattern)

def S(p):
    theta = 2 * np.arccos(np.sqrt((p - 1) / p))
    qc.cu(theta, 0, 0, 0, c[1], c[0])
    
def Save(pattern):
    sig = pattern[::-1]
    multiCX(qc, x, c[1], sig)


def getState():
    result = ""
    probs = Statevector.from_instruction(qc).probabilities()
    for state in range(len(probs)):
        bformat = "{0:0" + str(n + 4) + "b}"
        state_str = bformat.format(state)
        if np.abs(probs[state]) < 0.0001:
            continue
        elif probs[state] == 1:
            return "|" + state_str + ">"
        coefficient = np.sqrt(probs[state])
        coefficient = "{0:.2f}".format(coefficient)
        result += coefficient + " |" + state_str + "> + "
    result = result[: len(result) - 3]
    return result


def SavePatterns(patterns):
    m = len(patterns)
    assert m < 2 ** n
    for i in range(m):
        pattern = patterns[i]
        assert len(pattern) == n
        Flip(patterns, i)
        S(m - i)
        Save(pattern)


def GroverSearch(s):
    assert s >= 0 and s < N, "Invalid Search Parameter"
    sig = (("{0:0" + str(n) + "b}").format(s))[::-1]
    print(sig)
    qc.x(output[0])
    qc.h(output[0])

    for i in range(R):
        multiCX(qc, x, output, sig)  # unitary
        GroverDiffusion()  # W
        if i == 0:  # after first iteration
            multiCX(qc, c, output, "10")  # phase rotate saved patterns
            GroverDiffusion()  # W
        print(getState())

    qc.h(output[0])
    qc.x(output[0])
    qc.barrier()

    # for i in range(n):
    #     qc.measure(i + 1, i)

patterns = ["01","10","11"]
SavePatterns(patterns)
GroverSearch(1)
qc.measure(x, xc)
qc.measure(g, gc)
qc.measure(c, cc)

# execute the quantum circuit
backend = Aer.get_backend("qasm_simulator")
job = execute(qc, backend, shots=1024)
data = job.result().get_counts(qc)
print(data)
