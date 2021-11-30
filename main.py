from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit.visualization import (
    plot_histogram,
    plot_state_qsphere,
    plot_bloch_multivector,
    plot_bloch_vector,
)

patterns = ["0000", "0011", "0110", "1001", "1100", "1111"]
n = 4
N = 2 ** n
R = int(np.floor(np.pi * np.sqrt(N) / 4))
x = QuantumRegister(n)
c = QuantumRegister(2)
xc = ClassicalRegister(n)
cc = ClassicalRegister(2)
output = QuantumRegister(1)
qc = QuantumCircuit(x, c, output, xc, cc)


def getBitstring(s):
    if type(s) is int:
        assert s >= 0 and s < N, "Invalid Search Parameter"
        return ("{0:0" + str(n) + "b}").format(s)
    elif type(s) is str:
        assert len(s) == n
        return s
    else:
        raise Exception("Invalid bitstring type")


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
    qc.h(x)
    qc.x(x)
    multiPhase(qc, x, np.pi)
    qc.x(x)
    qc.h(x)


def Flip(index):
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
    return result + "\n"


def SavePatterns():
    m = len(patterns)
    assert m <= 2 ** n
    for i in range(m):
        pattern = getBitstring(patterns[i])
        assert len(pattern) == n
        Flip(i)
        S(m - i)
        Save(pattern)


def GroverSearch(s):
    s = getBitstring(s)[::-1]

    qc.x(output[0])
    qc.h(output[0])

    multiCX(qc, x, output, s)  # unitary
    GroverDiffusion()  # W

    # modified iteration
    for pattern in patterns:
        multiCX(qc, x, output, getBitstring(pattern)[::-1])  # phase rotate saved patterns
    GroverDiffusion()  # W

    for i in range(R - 2):
        multiCX(qc, x, output, s)  # unitary
        GroverDiffusion()  # W

    qc.h(output[0])
    qc.x(output[0])
    qc.barrier()

    qc.measure(x, xc)
    qc.measure(c, cc)


SavePatterns()
GroverSearch("0110")

# execute the quantum circuit
backend = Aer.get_backend("qasm_simulator")
job = execute(qc, backend, shots=1024)
data = job.result().get_counts(qc)
print(data)
