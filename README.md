# BayesRVAT

Code for the paper: "Bayesian Aggregation of Multiple Annotations Enhances Rare Variant Association Testing" presented at [RECOMB 2025](https://recomb.org/recomb2025/accepted_papers.html) and now published in [Genome Research](https://genome.cshlp.org/content/35/12/2682).

## Installation
If you want to run the showcase notebook and adapt the code please run:
```sh
git clone https://github.com/AIH-SGML/BayesRVAT.git
cd BayesRVAT
conda env create --file ./environment.yaml
conda activate bayesrvat
pip install -e .
```

## Showcase notebook
Check out `./notebooks/README.md` on how to run BayesRVAT on your data.

## License
This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.

## Citation
```
@article {Nappi2682-2690,
title = {BayesRVAT enhances rare-variant association testing through Bayesian aggregation of functional annotations},
author = {Nappi, Antonio and Shilova, Liubov and Karaletsos, Theofanis and Cai, Na and Casale, Francesco Paolo},
journal = {Genome Research},
year = {2025},
doi = {10.1101/gr.280689.125}
```
}
