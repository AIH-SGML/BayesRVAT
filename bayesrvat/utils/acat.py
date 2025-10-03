import warnings
import scipy.stats as st
import numpy as np
import warnings

def compute_acat(pvs):
    """
    pvs is n_different_tests x n_tests_to_integrate
    """
    RV =  np.ones(pvs.shape[0])
    for i in range(pvs.shape[0]):
        _pvs = pvs[i]
        is_nan = np.isnan(_pvs)
        if is_nan.all():
            RV[i] = np.nan
        else:
            _pvs = _pvs[~is_nan]
            RV[i] = acat_test(_pvs)
    return RV

def acat_test(pvals, weights=None, is_check=True, low_pval=1e-15, high_stat=1e15):
    """
    Combine p-values per row using the Aggregated Cauchy Association Test (ACAT).

    Parameters
    ----------
    pvals : ndarray, shape (n_tests, n_pvals)
        Matrix of p-values. Each row contains the p-values to combine into one.
    weights : None, ndarray of shape (n_pvals,), or ndarray of shape (n_tests, n_pvals)
        Non-negative weights for each p-value. If None, equal weights are used.
        If 1D, the same set of weights is applied to every row.
        If 2D, each row must match the corresponding p-value row.
    is_check : bool, default=True
        Whether to validate inputs: no NaNs, p-values in [0,1], weights non-negative,
        and no row contains both 0 and 1 p-values.
    low_pval : float, default=1e-15
        Threshold below which to approximate tan-transform by 1/(p * pi).
    high_stat : float, default=1e15
        Threshold above which to approximate the tail p-value by 1/(stat * pi).

    Returns
    -------
    combined : ndarray, shape (n_tests,)
        Combined p-values for each row, valid under arbitrary correlation.
    """
    dtype = np.float64
    eps=1e-12
    pvals = np.array(pvals, dtype=dtype)
    if pvals.ndim == 1:
        pvals = pvals.reshape(1, -1)
    n, m = pvals.shape

    # Prepare weights
    if weights is None:
        w = np.ones((n, m), dtype=dtype)
    else:
        w = np.array(weights, dtype=dtype)
        if w.ndim == 1:
            if w.shape[0] != m:
                raise ValueError("1D weights must have length n_pvals={}.".format(m))
            w = np.broadcast_to(w, (n, m))
        elif w.shape != (n, m):
            raise ValueError("2D weights must match pvals shape {}.".format(pvals.shape))

    # Input checks
    if is_check:
        if np.isnan(pvals).any():
            raise ValueError("Cannot have NaN in p-values.")
        if (pvals < 0).any() or (pvals > 1).any():
            raise ValueError("P-values must be between 0 and 1.")

    # Detect zeros and ones
    zero = pvals == 0
    one = pvals == 1
    both = zero.any(axis=1) & one.any(axis=1)
    if both.any():
        rows = np.where(both)[0]
        raise ValueError(f"Rows {rows} have both 0 and 1 p-values, invalid for ACAT.")

    # Replace exact ones by (1 - eps)
    if one.any():
        pvals[one] = 1 - eps
        warnings.warn(f"Adapted {one.sum()} p-values equal to 1 to {1 - eps}", stacklevel=2)

    # Normalize weights so each row sums to 1
    w_sum = w.sum(axis=1, keepdims=True)
    if (w_sum <= 0).any():
        raise ValueError("Each row of weights must sum to a positive value.")
    w = w / w_sum

    # Cauchy transformation
    small = pvals < low_pval
    t = np.empty_like(pvals)
    # small p-values
    t[small] = w[small] / (pvals[small] * np.pi)
    # regular p-values
    t[~small] = w[~small] * np.tan((0.5 - pvals[~small]) * np.pi)

    # Test statistic per row
    stat = t.sum(axis=1)

    # Compute combined p-values
    surv = np.empty(n)
    # high stat approx
    high = stat > high_stat
    surv[high] = 1.0 / (stat[high] * np.pi)
    # exact survival function
    surv[~high] = st.cauchy.sf(stat[~high])

    return surv