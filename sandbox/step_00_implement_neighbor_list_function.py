def neighbor_list(site, N):
    i, j = site
    return [((i-1)%N, j), (i, (j+1)%N), ((i+1)%N, j), (i, (j-1)%N)]