import numpy as np
import itertools

def tensor(*args):
    if len(args) == 0:
        return np.array([1])
    result = args[0]
    for a in args[1:]:
        result = np.kron(result, a)
    return result

def channel_output(input_state, channel1, channel2=None):
    n = int(np.log2(input_state.shape[0])) // 2
    if channel2 is None:
        channel2 = channel1
    N = 2 * n
    def apply_channel_to_qubits(state, channel, qubit_indices):
        kraus_per_qubit = [channel for _ in qubit_indices]
        new_state = np.zeros_like(state)
        for combo in itertools.product(*kraus_per_qubit):
            ops = [np.eye(2) for _ in range(N)]
            for i, kraus in zip(qubit_indices, combo):
                ops[i] = kraus
            full_op = tensor(*ops)
            new_state += full_op @ state @ full_op.conj().T
        return new_state
    state = apply_channel_to_qubits(input_state, channel1, list(range(n)))
    state = apply_channel_to_qubits(state, channel2, list(range(n, 2*n)))
    return state