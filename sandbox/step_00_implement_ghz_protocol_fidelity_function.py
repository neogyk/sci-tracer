import numpy as np

def ghz_protocol_fidelity(input_state, channel1, channel2=None):
    if channel2 is None:
        channel2 = channel1
    n = int(np.log2(input_state.shape[0]) // 2)
    dims = [2] * (2 * n)
    state = apply_channel(input_state, channel1, sys=list(range(n)), dim=dims[:n])
    state = apply_channel(state, channel2, sys=list(range(n, 2*n)), dim=dims[n:])
    two_qubit = ghz_protocol(state)
    phi_plus = np.array([[0.5, 0, 0, 0.5],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0.5, 0, 0, 0.5]])
    return fidelity(two_qubit, phi_plus)