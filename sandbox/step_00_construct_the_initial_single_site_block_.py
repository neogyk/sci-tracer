def block_initial(model_d: int) -> 'Block':
    import numpy as np
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=float)
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    H1 = np.zeros((2, 2), dtype=float)
    assert model_d == 2, "Spin-1/2 requires model_d=2"
    block = Block(length=1, basis_size=2, operator_dict={"H": H1, "conn_Sz": Sz, "conn_Sp": Sp})
    return block