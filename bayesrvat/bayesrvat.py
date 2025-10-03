import numpy as np
import scipy.stats as st
import scipy.special as ss
from .utils.utils import softplus, softplus_derivative, expit_derivative, normal_pdf
import time


class BayesRVAT():
    """
    Bayesian Rare Variant Association Test (BayesRVAT).

    Variational Bayesian framework for gene-level rare variant association testing with
    quantitative (Gaussian) or binary traits. Follows the BayesRVAT formulation that:
    (i) models fixed effects and a learned burden term; (ii) partitions samples into
    carriers and non-carriers for a null/alternative decomposition; (iii) optimizes a
    Monte Carlo–estimated ELBO with reparameterization; and (iv) supports importance-
    weighted ELBO for tighter bounds.

    Parameters
    ----------
    Y : np.ndarray, shape (N,) or (N, K)
        Phenotype vector or matrix.
    F : np.ndarray, shape (N, p)
        Fixed-effect design matrix (covariates, intercept optional).
    X : np.ndarray, shape (N, L)
        Variant features aggregated per sample (e.g., per-variant annotations or genotypes).
    idxs : np.ndarray, shape (Nc,)
        Integer indices of carrier samples to define burden vs. non-burden partitions.
    prior_mean : np.ndarray, shape (L,)
        Prior mean of burden weights.
    prior_std : np.ndarray, shape (L,)
        Prior standard deviation of burden weights (positive).
    positive : np.ndarray of {0,1}, shape (L,)
        Mask indicating which weight coordinates are constrained to be positive (softplus).
    S : int, default=16
        Number of Monte Carlo samples for variational expectations.
    trait_type : {'gaussian', 'binary'}, default='gaussian'
        Likelihood model for the trait.
    compute_null : bool, default=True
        If True, constructs an internal null BayesRVAT over non-carriers to enable
        null likelihood and combined objective.
    eps : np.ndarray or None, shape (L, S), default=None
        Optional antithetic or fixed noise for reproducibility; sampled if None.

    Notes
    -----
    - Burden weights use a location-scale reparameterization and optional softplus positivity.
    - Null model augments F0 with a small-probability column to absorb the burden intercept.
    - See the BayesRVAT preprint for methodological details and motivations.
    """

    def __init__(self, Y, F, X, idxs, prior_mean, prior_std, positive, S=16, trait_type='gaussian', compute_null: bool = True, eps=None):
        assert trait_type in ['gaussian', 'binary'], 'trait_type not valid!'
        self.trait_type = trait_type

        # full data
        self.Y = Y
        self.F = F
        
        # data burden
        self.idxs = idxs
        self.Y1 = Y[self.idxs]
        self.F1 = F[self.idxs]
        self.X = X
        self.X1 = X[self.idxs]
        # data no burden
        self.idxs0 = np.setdiff1d(np.arange(Y.shape[0]), idxs)
        self.Y0 = Y[self.idxs0]
        self.F0 = F[self.idxs0]
        extra_col = np.ones((self.F0.shape[0],1))*ss.expit(-6)
        self.F0 = np.concatenate([self.F0,extra_col],axis=1)
        if compute_null:
            self.brvat0 = self.__class__(self.Y0, self.F0, np.ones([self.Y0.shape[0], 1]), np.arange(self.Y0.shape[0]),prior_mean,prior_std,positive,compute_null=False)
        else: self.brvat0 = None 

        # prior
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.positive = positive

        # MC samples
        if eps is not None:
            self.eps = eps
        else:
            self.eps = np.random.randn(X.shape[1], S)

        # other params
        self.b0 = -6

        # init params
        self.alpha = 1e-3 * np.random.randn(F.shape[1])
        self.beta = 1e-3
        self.sigma_n = 0.9
        self.post_mean = self.prior_mean
        self.post_std = self.prior_std

    # --- Property definitions ---
    @property
    def alpha(self):
        """
        Fixed-effect coefficients.

        Returns
        -------
        np.ndarray, shape (p,)
            Current fixed-effect coefficients.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """
        Set fixed-effect coefficients.

        Parameters
        ----------
        value : np.ndarray, shape (p,)
            New coefficients.
        """
        self._alpha = value

    @property
    def beta(self):
        """
        Burden effect scalar.

        Returns
        -------
        float
            Current burden effect size.
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        """
        Set burden effect scalar.

        Parameters
        ----------
        value : float
            New burden effect size.
        """
        self._beta = value

    @property
    def sigma_n(self):
        """
        Observation noise standard deviation (Gaussian model).

        Stored internally as log_sigma_n for positivity.

        Returns
        -------
        float
            Noise standard deviation in natural scale.
        """
        return np.exp(self._log_sigma_n)

    @sigma_n.setter
    def sigma_n(self, value):
        """
        Set observation noise standard deviation.

        Parameters
        ----------
        value : float
            Positive standard deviation.

        Raises
        ------
        ValueError
            If value <= 0.
        """
        if value <= 0:
            raise ValueError("sigma_n must be positive")
        self._log_sigma_n = np.log(value)

    @property
    def post_mean(self):
        """
        Variational mean of burden weights.

        Returns
        -------
        np.ndarray, shape (L,)
            Mean parameter of q(w).
        """
        return self._post_mean

    @post_mean.setter
    def post_mean(self, value):
        """
        Set variational mean of burden weights.

        Parameters
        ----------
        value : np.ndarray, shape (L,)
            New mean parameter.
        """
        self._post_mean = value

    @property
    def post_std(self):
        """
        Variational std of burden weights.

        Stored internally as post_log_std for positivity.

        Returns
        -------
        np.ndarray, shape (L,)
            Standard deviation in natural scale.
        """
        return np.exp(self._post_log_std)

    @post_std.setter
    def post_std(self, value):
        """
        Set variational std of burden weights.

        Parameters
        ----------
        value : np.ndarray, shape (L,)
            Positive standard deviations.

        Raises
        ------
        ValueError
            If any entry <= 0.
        """
        if (value <= 0).any():
            raise ValueError("post_std must be positive")
        self._post_log_std = np.log(value)

    def get_w(self, eps=None, return_all=False):
        """
        Sample burden weights via reparameterization.

        w_raw = post_mean + post_std * eps
        w = w_raw (free coords) or softplus(w_raw) (positive coords)

        Parameters
        ----------
        eps : np.ndarray or None, shape (L, S), default=None
            Noise draws. Uses self.eps if None.
        return_all : bool, default=False
            If True, also return intermediates.

        Returns
        -------
        np.ndarray, shape (L, S)
            Sampled weights w.
        dict, optional
            Intermediates including 'raw_w' if return_all=True.
        """
        if eps is None:
            eps = self.eps
        _raw_w = self.post_mean[:,None] + self.post_std[:,None] * eps
        _softplus_w = softplus(_raw_w)
        _w = _raw_w * (1. - self.positive[:,None]) + _softplus_w * self.positive[:,None]
        if return_all:
            return _w, {'raw_w': _raw_w}
        return _w
        
    def get_burden(self, eps=None, return_all=False, only_carriers=False):
        """
        Compute sample-wise burden probabilities.

        Burden = sigmoid(X @ w + b0), averaged over S samples if used downstream.

        Parameters
        ----------
        eps : np.ndarray or None, shape (L, S), default=None
            Optional noise for w sampling.
        return_all : bool, default=False
            If True, also return intermediates from get_w.
        only_carriers : bool, default=False
            If True, uses X1; else uses X.

        Returns
        -------
        np.ndarray, shape (N, S) or (Nc, S)
            Burden values per sample and MC draw.
        dict, optional
            Intermediates including 'raw_w' and 'w' if return_all=True.
        """
        _w, _all = self.get_w(eps=eps, return_all=True)
        _all['w'] = _w
        _X = self.X1 if only_carriers else self.X
        _burden = ss.expit(_X.dot(_w) + self.b0)
        if return_all:
            return _burden, _all
        return _burden

    def null_lml(self, gradients=False):
        """
        Null-model log-likelihood and gradients.

        Fits the fixed-effect-only model over the provided data: mean = F @ alpha.
        For Gaussian traits uses Normal with sigma_n; for binary traits uses logistic.

        Parameters
        ----------
        gradients : bool, default=False
            If True, also return analytical gradients w.r.t. alpha and sigma_n.

        Returns
        -------
        float
            Null-model log-likelihood estimate (averaged over MC dimension if present).
        dict, optional
            Gradients with keys {'alpha', 'sigma_n'} when gradients=True.
        """
        # log lik
        _mean = self.F.dot(self.alpha[:,None])
        if self.trait_type=='gaussian':
            _residuals = self.Y - _mean
            _loglik = normal_pdf(_residuals, self.sigma_n).sum(0).mean()
        else:
            _residuals = self.Y - ss.expit(_mean)
            log_p = -np.logaddexp(0, -_mean)
            log_1mp = -np.logaddexp(0, _mean)
            _loglik = (self.Y * log_p + (1 - self.Y) * log_1mp).sum(0).mean()

        # compute gradients
        if gradients:

            gradients = {}
            
            # alpha
            d_quad_term_d_alpha = (self.F.T.dot(_residuals)).mean(1)
            if self.trait_type=='gaussian':
                d_quad_term_d_alpha /= self.sigma_n**2
            gradients['alpha'] = d_quad_term_d_alpha

            # sigma_n
            if self.trait_type=='gaussian':
                d_quad_term_d_sigman = (_residuals**2).sum(0).mean() / self.sigma_n**3
                d_logdet_term_d_sigman = - self.Y.shape[0] / self.sigma_n
                gradients['sigma_n'] = d_quad_term_d_sigman + d_logdet_term_d_sigman
            else:
                gradients['sigma_n'] = 0

            return _loglik, gradients

        return _loglik
        
    
    def elbo(self, gradients=False):
        """
        Evidence lower bound for the alternative model.

        Computes ELBO = E_q[log p(Y | F, X, w, alpha, beta, sigma)] - KL(q(w) || p(w)),
        with the data split into non-carriers (null submodel) and carriers (burden model).
        For Gaussian traits, residual terms scale with sigma_n; for binary traits, uses
        logistic likelihood. Supports analytical gradients via the reparameterization trick.

        Parameters
        ----------
        gradients : bool, default=False
            If True, return ELBO and gradient dict for optimization.

        Returns
        -------
        float
            ELBO value.
        (float, dict), optional
            ELBO and gradients when gradients=True. Gradient keys:
            {'alpha','beta','sigma_n','post_mean','post_std'}.
        """
        verbose = False
        t0 = time.time()
        self.brvat0.alpha = np.concatenate([self.alpha,np.asarray([self.beta])])
        self.brvat0.sigma_n = self.sigma_n
        _loglik0, _gradients0 = self.brvat0.null_lml(gradients=True)
        
        # sample burden from posterior
        _burden1, _all = self.get_burden(return_all=True, only_carriers=True)
        _raw_w = _all['raw_w']

        # compute mean, std and loglik
        _mean1 = self.F1.dot(self.alpha[:,None]) + _burden1 * self.beta
        if self.trait_type=='gaussian':
            _residuals1 = self.Y1 - _mean1
            _loglik1 = normal_pdf(_residuals1, self.sigma_n).sum(0).mean()
        else:
            _residuals1 = self.Y1 - ss.expit(_mean1)
            log_p = -np.logaddexp(0, -_mean1)
            log_1mp = -np.logaddexp(0, _mean1)
            _loglik1 = (self.Y1 * log_p + (1 - self.Y1) * log_1mp).sum(0).mean()
        _loglik = _loglik0 + _loglik1

        # Compute KLD
        _kld = (np.log(self.prior_std / self.post_std) + (self.post_std**2 + (self.post_mean - self.prior_mean) ** 2) / (2 * self.prior_std**2) - 0.5).sum()
        
        # ELBO
        _elbo = _loglik - _kld

        if verbose: print('elbo:', time.time() - t0)
        t0 = time.time()

        # compute gradients
        if gradients:

            gradients1 = {}

            # alpha
            d_quad_term_d_alpha = (self.F1.T.dot(_residuals1)).mean(1)
            if self.trait_type=='gaussian':
                d_quad_term_d_alpha /= self.sigma_n**2
            gradients1['alpha'] = d_quad_term_d_alpha

            if verbose: print('alpha:', time.time() - t0)
            t0 = time.time()
            
            # beta
            d_quad_term_d_beta = np.einsum('nk,nk->k', _burden1, _residuals1).mean(0)
            if self.trait_type=='gaussian':
                d_quad_term_d_beta /= self.sigma_n**2
            gradients1['beta'] = d_quad_term_d_beta

            if verbose: print('beta:', time.time() - t0)
            t0 = time.time()

            # sigma_n
            if self.trait_type=='gaussian':
                d_quad_term_d_sigman = (_residuals1**2).sum(0).mean() / self.sigma_n**3
                d_logdet_term_d_sigman = - self.Y1.shape[0] / self.sigma_n
                gradients1['sigma_n'] = d_quad_term_d_sigman + d_logdet_term_d_sigman
            else:
                gradients1['sigma_n'] = 0

            if verbose: print('sigma:', time.time() - t0)
            t0 = time.time()

            # precompute
            d_w_d_wraw = (1. - self.positive[:,None]) + softplus_derivative(_raw_w) * self.positive[:,None] # LxK
            if verbose: print('precomp1:', time.time() - t0)
            t0 = time.time()
            burden_derivative1 = _burden1 * (1 - _burden1)
            _brx = np.einsum('nk,nl->kl', burden_derivative1 * _residuals1, self.X1)
            d_quad_term_d_wraw = self.beta * _brx * d_w_d_wraw.T
            if self.trait_type=='gaussian':
                d_quad_term_d_wraw /= self.sigma_n**2
            if verbose: print('precomp4:', time.time() - t0)
            t0 = time.time()
            
            # post_mean
            d_quad_term_d_pmean = d_quad_term_d_wraw.mean(0)
            d_kld_d_pmean = (self.post_mean - self.prior_mean) / self.prior_std**2
            gradients1['post_mean'] = d_quad_term_d_pmean - d_kld_d_pmean

            if verbose: print('post_mean:', time.time() - t0)
            t0 = time.time()

            # post_std
            d_quad_term_d_pstd = (d_quad_term_d_wraw * self.eps.T).mean(0)
            d_kld_d_pstd = - 1 / self.post_std + 2 * self.post_std / (2 * self.prior_std**2)
            gradients1['post_std'] = d_quad_term_d_pstd - d_kld_d_pstd

            gradients = {}
            for key in (_gradients0.keys()-set(['alpha'])):
                gradients[key] = _gradients0[key] + gradients1[key]
            for key in gradients1.keys() -(_gradients0.keys() | set(['beta'])):
                gradients[key] = gradients1[key]
            gradients['alpha']= _gradients0['alpha'][:-1] + gradients1['alpha']
            gradients['beta']= _gradients0['alpha'][-1] + gradients1['beta']
            if verbose: print('post_std:', time.time() - t0)
            return _elbo, gradients  
        
        return _elbo

    def iwelbo(self, K=16, S=30, batch_size=16):
        """
        Importance-weighted ELBO.

        Estimates a tighter bound on log marginal likelihood using K importance samples
        per S groups, evaluated in batches for memory efficiency.

        Parameters
        ----------
        K : int, default=16
            Number of importance samples per S group.
        S : int, default=30
            Number of groups (also used to reshape accumulated log-weights).
        batch_size : int, default=16
            Number of weight samples evaluated per batch; requires (K*S) % batch_size == 0.

        Returns
        -------
        np.ndarray, shape (S,)
            IWELBO values for each group; typical use aggregates via mean().
        """
        # Ensure that K*S is divisible by the batch_size
        assert (K * S) % batch_size == 0, 'K*S should be divisible by batch_size'
        
        n_batches = int(K * S / batch_size)
        log_weights_list = []
        
        for _ in range(n_batches):
            
            # Sample weights
            _eps = np.random.randn(self.X.shape[1], batch_size)
            _burden, _all = self.get_burden(eps=_eps, return_all=True)
            _raw_w = _all['raw_w']

            # compute log lik
            _mean = self.F.dot(self.alpha[:,None]) + _burden * self.beta
            if self.trait_type=='gaussian':
                _loglik = normal_pdf(self.Y - _mean, self.sigma_n).sum(0).mean()
            else:
                log_p = -np.logaddexp(0, -_mean)
                log_1mp = -np.logaddexp(0, _mean)
                _loglik = (self.Y * log_p + (1 - self.Y) * log_1mp).sum(0).mean()
            
            # Compute prior and posterior probabilities
            _log_prior = normal_pdf(_raw_w - self.prior_mean[:,None], self.prior_std[:,None]).sum(0)
            _log_post  = normal_pdf(_raw_w - self.post_mean[:,None], self.post_std[:,None]).sum(0)
            
            # Calculate log-weights for this batch
            _log_weights = _loglik + _log_prior - _log_post
            log_weights_list.append(_log_weights)
        
        # Concatenate log-weights from all batches
        log_weights = np.concatenate(log_weights_list, axis=0)
        
        # Reshape into shape (K, S)
        log_weights = log_weights.reshape(K, S)
        
        # Compute the importance weighted ELBO:
        # For numerical stability, we use scipy's logsumexp which computes
        # log(sum(exp(x))) in a stable manner.
        iwelbo = ss.logsumexp(log_weights, axis=0) - np.log(K)
        
        return iwelbo

    def getPv(self, K=16, S=30, batch_size=16):
        """
        P-value from IWELBO vs null.

        Computes 2*(mean(IWELBO) - lml0) and evaluates a chi-square(1) survival function.

        Parameters
        ----------
        K : int, default=16
            Importance samples per group.
        S : int, default=30
            Number of groups.
        batch_size : int, default=16
            Batch size used in iwelbo.

        Returns
        -------
        float
            One-degree-of-freedom chi-square p-value.
        """
        iwelbo = self.iwelbo(K=K, S=S, batch_size=batch_size)
        return st.chi2(1).sf(2 * (iwelbo.mean() - self.lml0))

    #################################################
    # The following methods are for optimization
    #################################################
    
    def _unravel_params(self, x):
        """
        Unpack flat parameter vector.

        Parameters
        ----------
        x : np.ndarray, shape (P,)
            Concatenated parameters in the order returned by getParams().

        Returns
        -------
        tuple
            (alpha, beta, log_sigma_n, post_mean, post_log_std) with shapes:
            (p,), (), (), (L,), (L,)
        """
        nf, nx = self.F.shape[1], self.X.shape[1]
        alpha = x[:nf]
        beta = x[nf]
        log_sigma_n = x[nf+1]
        post_mean = x[nf+2:nf+2+nx]
        post_log_std = x[nf+2+nx:nf+2+2*nx]
        return alpha, beta, log_sigma_n, post_mean, post_log_std
    
    def setRandomParams(self):
        """
        Initialize parameters with standard Normal draws.

        Side Effects
        ------------
        Overwrites internal parameters via setParams().
        """
        x0 = np.random.randn(self.getNumberParams())
        self.setParams(x0)

    def getNumberParams(self):
        """
        Total number of free parameters.

        Returns
        -------
        int
            p (alpha) + 1 (beta) + 1 (log_sigma_n) + L (post_mean) + L (post_log_std).
        """
        n_fixed_effs = self.F.shape[1] + 1
        n_noise_pars = 1
        n_var_pars = 2 * self.X.shape[1]
        return n_fixed_effs + n_noise_pars + n_var_pars

    def setParams(self, x):
        """
        Load parameters from a flat vector.

        Parameters
        ----------
        x : np.ndarray, shape (P,)
            Concatenated parameters ordered as in getParams().

        Side Effects
        ------------
        Sets internal attributes:
        _alpha, _beta, _log_sigma_n, _post_mean, _post_log_std.
        """
        alpha, beta, log_sigma_n, post_mean, post_log_std = self._unravel_params(x)

        # set model parameters
        self._alpha = alpha
        self._beta = beta
        self._log_sigma_n = log_sigma_n

        # variational parameters
        self._post_mean = post_mean
        self._post_log_std = post_log_std

    def getParams(self):
        """
        Export parameters as a flat vector.

        Returns
        -------
        np.ndarray, shape (P,)
            Concatenation of:
            [alpha, beta, log_sigma_n, post_mean, post_log_std].
        """
        # When retrieving parameters, we return sigma_n and post_std in log-space internally
        # so that getParams() remains consistent with setParams() ordering.
        x = np.concatenate([
            self._alpha, 
            np.array([self._beta]), 
            np.array([self._log_sigma_n]), 
            self._post_mean, 
            self._post_log_std
        ])
        return x

    def loss(self, x):
        """
        Negative ELBO and gradient for optimizer.

        Parameters
        ----------
        x : np.ndarray, shape (P,)
            Current parameter vector.

        Returns
        -------
        tuple
            (-ELBO, -grad) where grad matches x's layout:
            [alpha, beta, log_sigma_n, post_mean, post_log_std].
        """
        self.setParams(x)
        elbo, G = self.elbo(gradients=True)
        G['log_sigma_n'] = G['sigma_n'] * self.sigma_n
        G['post_log_std'] = G['post_std'] * self.post_std
    
        gradients = np.concatenate([
            G['alpha'],
            np.array([G['beta']]),
            np.array([G['log_sigma_n']]),
            G['post_mean'],
            G['post_log_std']
        ])
    
        return -elbo, -gradients

    def optimize(self, **kwargs):
        """
        L-BFGS-B optimization of the full model.

        Parameters
        ----------
        **kwargs
            Passed to scipy.optimize.fmin_l_bfgs_b (e.g., maxiter, factr, pgtol).

        Returns
        -------
        tuple
            (conv, info) where conv is bool for convergence and info is the optimizer info dict.

        Notes
        -----
        Uses getParams()/loss()/setParams() contract. Initializes from current state.
        """
        from scipy.optimize import fmin_l_bfgs_b as optimize
    
        # Retrieve initial parameters.
        x0 = self.getParams()
    
        # Run the optimization.
        result = optimize(self.loss, x0, **kwargs)
        
        # Extract optimization info.
        info = result[2]
        conv = info['warnflag'] == 0
    
        return conv, info

    def optimize_null(self, **kwargs):
        """
        L-BFGS-B optimization of the null model.

        Optimizes alpha and log_sigma_n under the fixed-effect-only likelihood and caches
        the optimum null log-likelihood in self.lml0.

        Parameters
        ----------
        **kwargs
            Passed to scipy.optimize.fmin_l_bfgs_b.

        Returns
        -------
        tuple
            (conv, info) for optimizer convergence and info dict.

        Side Effects
        ------------
        Updates _alpha, _log_sigma_n; sets self.lml0 to null_lml() at the optimum.
        """
        from scipy.optimize import fmin_l_bfgs_b as optimize

        def setNullParams(x):
            self._alpha = x[:self.F.shape[1]]
            self._log_sigma_n = x[-1]
    
        def getNullParams():
            x = np.concatenate([
                self._alpha,
                np.array([self._log_sigma_n]), 
            ])
            return x
        
        def null_loss(x):
        
            setNullParams(x)
            lml, G = self.null_lml(gradients=True)
            G['log_sigma_n'] = G['sigma_n'] * self.sigma_n
        
            gradients = np.concatenate([
                G['alpha'],
                np.array([G['log_sigma_n']])
            ])
        
            return -lml, -gradients
    
        # Run the optimization.
        x0 = getNullParams()
        result = optimize(null_loss, x0, **kwargs)

        # store optimum
        self.lml0 = self.null_lml(gradients=False)
        
        # Extract optimization info.
        info = result[2]
        conv = info['warnflag'] == 0
    
        return conv, info

    def run_simple_burden(self, Y, F, Xt, annots, prior_mean, prior_std, positive, trait_type='gaussian'):
        """
        Run simple univariate burden tests for each annotation.

        For each column in Xt (corresponding to an annotation), this fits a univariate
        BayesRVAT model by augmenting the fixed-effect design F with that single annotation.
        It then computes the burden coefficient estimate (beta) and a likelihood ratio
        test p-value against the null model.

        Parameters
        ----------
        Y : np.ndarray, shape (N,)
            Phenotype vector.
        F : np.ndarray, shape (N, p)
            Fixed-effect design matrix.
        Xt : np.ndarray, shape (N, K)
            Annotation matrix, each column is one candidate annotation.
        annots : list of str
            Names of the annotations (length K).
        prior_mean : np.ndarray, shape (L,)
            Prior mean for burden weights.
        prior_std : np.ndarray, shape (L,)
            Prior standard deviation for burden weights.
        positive : np.ndarray, shape (L,)
            Indicator array for positive constraints on burden weights.
        trait_type : {'gaussian','binary'}, default='binary'
            Trait model to use.

        Returns
        -------
        dict
            Mapping from annotation name to a dict with keys:
            'beta' : float or np.nan
                Estimated burden coefficient.
            'pv' : float or np.nan
                Likelihood ratio test p-value.
        """
        results = {}
        # fit null model once
        brvat0 = BayesRVAT(Y, F, Xt, np.arange(Y.shape[0]), prior_mean, prior_std, positive, S=16, trait_type=trait_type)
        conv0, info0 = brvat0.optimize_null(factr=1e3)
        loglik_null = brvat0.null_lml()

        for ik, key in enumerate(annots):
            _x = Xt[:, [ik]]
            if (_x == 0).all():
                results[key] = {'beta': np.nan, 'pv': np.nan}
                continue

            F1 = np.concatenate([F, _x], axis=1)
            brvat = BayesRVAT(Y, F1, Xt, np.arange(Y.shape[0]), prior_mean, prior_std, positive, S=16, trait_type=trait_type)
            brvat.alpha = np.append(brvat0.alpha, 0)
            conv1, info1 = brvat.optimize_null(factr=1e3)
            loglik_alt = brvat.null_lml()
            beta = brvat.alpha[-1]
            pv = st.chi2(1).sf(2 * (loglik_alt - loglik_null))
            results[key] = {'beta': beta, 'pv': pv}

        return results
