import numpy as np
import scipy.stats as st
import pandas as pd
from bayesrvat.utils.utils import gaussianize
from bayesrvat import BayesRVAT

def simulate_genetics(df, n_samples, seed=0):
    """
    Simulate a PLINK-like genotype matrix from variant MAFs.

    Each genotype is simulated under Hardy–Weinberg equilibrium, where
    dosages (0, 1, 2) represent the count of the minor allele (a1).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['chrom', 'snp', 'cm', 'pos', 'a0', 'a1', 'maf'].
    n_samples : int
        Number of individuals to simulate.
    seed : int, optional
        Random generator seed (default=0).

    Returns
    -------
    G : pandas.DataFrame, shape (n_samples, n_variants)
        Genotype matrix with rows as individuals and columns indexed by SNP ID.
        Values are minor-allele dosages in {0,1,2}.
    """
    # Ensure only required columns
    required_cols = ['chrom', 'snp', 'cm', 'pos', 'a0', 'a1', 'maf']
    df = df[required_cols].copy()

    # Minor allele frequencies
    maf = np.clip(df['maf'].to_numpy(float), 0.0, 1.0)

    # Random generator
    rng = np.random.default_rng(seed)

    # HWE simulation: dosage ~ Binomial(2, maf)
    genotypes = rng.binomial(2, maf[None, :], size=(n_samples, maf.size)).astype(np.int32)

    # Build DataFrame with SNPs as columns
    G = pd.DataFrame(genotypes, columns=df['snp'].astype(str))

    return G


def simulate_phenotype(XA, vg,num_annots=5,plof_mean=1,missense_mean=1,other_mean=1,annots_mean=1,plof_std=1,missense_std=2,other_std=2,annots_std=2):

    Y = np.ones((XA.shape[0], 1))

    # pick some annots
    annots_idxs = np.arange(3, XA.shape[1])
    sel_annots = np.sort(np.random.permutation(annots_idxs)[:num_annots])
    all_sel_annots = np.concatenate([np.arange(3), sel_annots], axis=0)
    XA = XA[:, all_sel_annots]

    # define w_mean, w_std, w_pos
    w_mean = np.array([plof_mean, missense_mean, other_mean] + [annots_mean] * num_annots)
    w_std = np.array([plof_std, missense_std, other_std] + [annots_std] * num_annots)
    positive_w = np.array([0, 0, 0] + [1] * num_annots)

    nonzero_rows = np.unique(XA.nonzero()[0])  
    
    brvat = BayesRVAT(Y=Y, F=Y, X=XA, idxs=nonzero_rows, prior_mean=w_mean, prior_std=w_std, positive=positive_w)
    
    burden = brvat.get_burden().mean(1)
    
    y_g = np.sqrt(vg / (np.var(burden) + 1e-08)) * burden
    y_g = y_g.reshape(-1, 1)

    y_n = np.random.randn(y_g.shape[0], 1)
    y_n = np.sqrt((1 - vg) / np.var(y_n)) * y_n
    y = y_g + y_n
    y = (y - y.mean(0)) / y.std(0)
    
    return gaussianize(y)
