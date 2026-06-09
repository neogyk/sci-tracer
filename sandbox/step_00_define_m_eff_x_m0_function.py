def m_eff(x, m0=1.0):
    m_e = (0.0637 + 0.083*x) * m0
    m_lh = (0.087 + 0.063*x) * m0
    m_hh = (0.50 + 0.29*x) * m0
    mr_lh = 1.0 / (1.0/m_e + 1.0/m_lh)
    mr_hh = 1.0 / (1.0/m_e + 1.0/m_hh)
    mr = (mr_lh**1.5 + mr_hh**1.5) ** (2.0/3.0)
    return mr