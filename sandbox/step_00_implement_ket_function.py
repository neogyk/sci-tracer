import numpy as np

def ket(dim, *args):
    if isinstance(dim, int):
        out = np.zeros(dim)
        if args:
            idx = args[0] - 1 if isinstance(args[0], int) else args[0][0] - 1
            out[idx] = 1
        else:
            out[0] = 1
        return out
    elif isinstance(dim, list):
        vecs = [ket(d, args[0][i] if args and len(args[0]) > i else 1) for i, d in enumerate(dim)]
        out = vecs[0]
        for v in vecs[1:]:
            out = np.kron(out, v)
        return out