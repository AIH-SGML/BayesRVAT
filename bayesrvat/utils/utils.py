import numpy as np
import scipy.special as ss
import scipy.stats as st

def softplus(x, beta=2):
    return (1 / beta) * np.log1p(np.exp(beta * x))

def softplus_derivative(x, beta=2):
    return 1 / (1 + np.exp(-beta * x))

def expit_derivative(x):
    sig_x = ss.expit(x)
    return sig_x * (1 - sig_x)

def normal_pdf(residuals, sigma):
    _quad_term = -0.5 * (residuals**2) / sigma**2
    _logdet_term = - np.log(sigma)
    _const_term = -0.5 * np.log(2 * np.pi)
    _loglik = _quad_term + _logdet_term + _const_term
    return _loglik

def d_T_std(T, d_T):
    N = T.shape[0]
    _s = T.std(0)[None, :, None]
    C_T = (T - T.mean(0, keepdims=True))[:, :, None]  # (N, S, 1)
    C_d_T = d_T - d_T.mean(0, keepdims=True)
    C3 = C_T * np.einsum("nsq,nsq->sq", C_T, C_d_T)
    return C_d_T / _s - C3 / (_s**3 * N)


def toRanks(A):
    """
    converts the columns of A to ranks
    """
    AA = np.zeros_like(A)
    for i in range(A.shape[1]):
        AA[:, i] = st.rankdata(A[:, i])
    AA = np.array(np.around(AA), dtype="int") - 1
    return AA


def gaussianize(Y):
    """
    Gaussianize X: [samples x phenotypes]
    - each phentoype is converted to ranks and transformed back to normal using the inverse CDF
    """
    N, P = Y.shape

    YY = toRanks(Y)
    quantiles = (np.arange(N) + 0.5) / N
    gauss = st.norm.isf(quantiles)
    Y_gauss = np.zeros((N, P))
    for i in range(P):
        Y_gauss[:, i] = gauss[YY[:, i]]
    Y_gauss *= -1
    return Y_gauss