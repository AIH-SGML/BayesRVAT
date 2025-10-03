# BayesRVAT Tutorial

## Introduction
This repository contains the `Tutorial.ipynb` notebook, which demonstrates how to use the **BayesRVAT** framework for rare variant association testing (RVAT). The notebook walks through data preparation, simulation of phenotypes, model fitting, and comparison against baseline aggregation methods.  

The tutorial is designed to be fully reproducible with provided synthetic data, so users can explore BayesRVAT’s functionalities without requiring direct access to large biobank datasets.

## Inputs to BayesRVAT Tutorial
The notebook utilizes the following data files located under `notebooks/data/`:

- **APOB_annots.csv**  
  Variant-level annotations for a single gene. Columns include SNP identifiers (`SNP_ID`), position, alleles, and functional annotations (e.g., consequence, missense, loss-of-function). Float values are exported with six decimal places.

- **APOB_freq.csv**  
  Allele frequency information per variant. Contains frequency and count fields for simulated case-control or cohort data.

- **model_choices/all.csv**  
  Specification of models considered by BayesRVAT. Each row corresponds to an annotation set with associated prior mean, standard deviation, and sign constraints.

These files mirror the structure described in the BayesRVAT paper, allowing quick replication of the methods.

## Workflow in the Tutorial
The tutorial proceeds through the following steps:

1. **Load input data**: Import variant frequencies, annotations, and model specification.  
2. **Simulate genotypes and phenotypes**: Generate a synthetic cohort and phenotype under a controlled genetic architecture.  
3. **Preprocess annotations**: Construct annotation matrices, normalize features, and compute variant-specific weights.  
4. **Fit BayesRVAT**: Run the variational Bayesian RVAT procedure, obtain gene-level P values, and extract posterior parameter estimates.  
5. **Compare to baselines**: Compute ACAT-consequence and ACAT-MultiAnnots P values for reference.  
6. **Visualization**:  Violin plot of posterior weight distributions for `plof`, `missense`, and `other1` annotations.  

## Example with Provided Data
To illustrate data formats and model behavior, the tutorial uses APOB as an example gene. Variants, annotations, and allele frequencies are simulated and shipped with the repository. 
