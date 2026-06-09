def neighbor_list(i, j, N):
    return [((i-1)%N, j), ((i+1)%N, j), (i, (j-1)%N), (i, (j+1)%N)]