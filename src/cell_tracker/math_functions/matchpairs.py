
import scipy.optimize     as sciopt
import numpy              as np

# This should mimic the function of MATLAB's matchpairs routine ...
def matchpairs(D, b = None, D_cutoff = None, MAX_VALUE = None):
    if MAX_VALUE is None:
        MAX_VALUE = np.finfo(np.float64).max
    
    if D_cutoff is not None:
        D[D >= D_cutoff] = MAX_VALUE

    if b is None:
        try:
            b = 1.05 * np.max(D[D < MAX_VALUE])
        except ValueError:
            # print(" **** No Matches ****")
            return None, None, None

    cost                             = np.zeros((np.sum(D.shape), np.sum(D.shape)))
    cost[0:D.shape[0], 0:D.shape[1]] = D
    cost[D.shape[0]:,  0:D.shape[1]] = MAX_VALUE * (1 - np.eye(D.shape[1])) + b * np.eye(D.shape[1])
    cost[0:D.shape[0],  D.shape[1]:] = MAX_VALUE * (1 - np.eye(D.shape[0])) + b * np.eye(D.shape[0])
    cost[D.shape[0]:,   D.shape[1]:] = np.transpose(D)

    rows, cols = sciopt.linear_sum_assignment(cost)

    uR = rows[0:D.shape[0]][cols[0:D.shape[0]] >= D.shape[1]]
    uC = cols[D.shape[0]:][cols[D.shape[0]:] < D.shape[1]]

    idx = ((rows >= D.shape[0]) + (cols >= D.shape[1])) > 0
    M   = np.transpose(np.array([rows[idx == 0], cols[idx == 0]]))

    return M, uR, uC



if __name__ == "__main__":
    pass
