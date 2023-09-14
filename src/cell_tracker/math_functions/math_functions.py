import numpy    as np
import networkx as nx
import pandas   as pd

# Creates an adjacency matrix computing the 
# distance between pairs of points
def pdist2(P, Q):
    Np = P.shape[0]
    Nq = Q.shape[0]
    
    if (len(P.shape) == 1):
        return np.square(P.reshape((Np, 1)) - Q.reshape((1, Nq))) 
    if P.shape[1] < 2:
        return np.square(P.reshape((Np, 1)) - Q.reshape((1, Nq)))
        
    return np.square(P[:, 0].reshape((Np, 1)) - Q[:, 0].reshape((1, Nq))) + \
           np.square(P[:, 1].reshape((Np, 1)) - Q[:, 1].reshape((1, Nq)))

# Splits up groups of objects with a specified cutoff
def connected_components(adj, threshold = None, min_size = 50):

    if threshold is not None:
        adj = adj < threshold

    G  = nx.from_numpy_array(adj)
    cc = nx.connected_components(G)

    labels = {ii:0 for ii in range(adj.shape[0])}
    for cntr, c in enumerate(cc):
        clist = np.array(list(c))
        if len(clist) < min_size:
            continue
        labels.update(dict.fromkeys(clist, cntr + 1))

    return pd.DataFrame.from_dict(labels)
