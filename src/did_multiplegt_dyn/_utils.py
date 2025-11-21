import numpy as np

def gaussian_elimination(A, b, tol=1e-7, verbose=False):
    """Perform Gaussian elimination on augmented matrix [A|b]."""
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    m, n = A.shape
    Ab = np.hstack([A, b])
    
    for i in range(min(m, n)):
        # Pivot
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if abs(Ab[max_row, i]) < tol:
            continue
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Normalize pivot row
        Ab[i] = Ab[i] / Ab[i, i]
        
        # Eliminate other rows
        for j in range(m):
            if j != i:
                Ab[j] -= Ab[j, i] * Ab[i]
    
    return Ab

def Ginv(A, tol=np.sqrt(np.finfo(float).eps), verbose=False):
    """Generalized inverse using Gaussian elimination (R's Ginv)."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    
    # First elimination
    B = gaussian_elimination(A, np.eye(m), tol=tol, verbose=verbose)
    L = B[:, n:]      # columns after first n
    AR = B[:, :n]     # first n columns
    
    # Second elimination
    C = gaussian_elimination(AR.T, np.eye(n), tol=tol, verbose=verbose)
    R = C[:, m:].T    # columns after first m, then transpose
    AC = C[:, :m].T   # first m columns, then transpose
    
    # Construct generalized inverse
    ginv = R @ AC.T @ L
    return ginv
