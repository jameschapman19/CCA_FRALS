import sys
from typing import Iterable

import pandas as pd
from joblib import dump

sys.path.append("C:/Users/chapm/PycharmProjects/FlexibleALSCCA")
sys.path.append("/home/jchapman/projects/FlexibleALSCCA")
from experiments.cvscore import outer_cv

sys.path.append("/home/jchapman/projects/braindata")
sys.path.append("C:/Users/chapm/PycharmProjects/braindata")
import wandb
from braindata import HCP, ABCD, ABIDE, ADNI
from cca_zoo import rCCA, SCCA_PMD, SCCA_IPLS, PLS
from cca_frals import CCA_FRALS,CCA_PCA
import numpy as np
from sklearn.linear_model import SGDRegressor, ElasticNet
from cca_zoo.models._base import _BaseCCA
from sklearn.impute import SimpleImputer


defaults = {
    "data": "hcp",
    "random_seed": 42,
    "model": "flexals",
    "c1": 1e-4,
    "c2": 1e-4,
    "l1_ratio1": 0.5,
    "l1_ratio2": 0.5,
    "connectome_size": 25,
}

datasets = {
    "hcp": HCP,
}

models = {
    "rcca": rCCA,
    "pmd": SCCA_PMD,
    "ipls": SCCA_IPLS,
    "pls": PLS,
    "pca": CCA_PCA,
    "flexals": FRALS,
}

HUE_ORDER = [
    "Cognition",
    "Personality",
    "Emotion",
    "Alertness",
    "Sensory",
    "Substance Use",
    "Psychiatric and Life Function",
    "Health and Family History",
    "Motor",
]


def main():
    wandb.init(config=defaults)
    config = wandb.config
    dataset = datasets[config.data]()
    base_model = models[config.model](latent_dims=1)
    if config.model == "rcca":
        base_model.set_params(c=[config.c1, config.c2])
    elif config.model == "pmd":
        base_model.set_params(c=[config.c1, config.c2])
    elif config.model == "ipls":
        base_model.set_params(c=[config.c1, config.c2])
    elif config.model == "flexals":
        regressor_1 = ElasticNet(warm_start=True)
        regressor_1.set_params(alpha=config.c1, l1_ratio=config.l1_ratio1)
        regressor_2 = ElasticNet(warm_start=True)
        regressor_2.set_params(alpha=config.c2, l1_ratio=config.l1_ratio2)
        base_model.set_params(regressors=[regressor_1, regressor_2])
    elif config.model in ["pls", "pca"]:
        pass
    else:
        raise ValueError("Model not recognized")
    X = dataset.X
    Y = dataset.Y
    ip = SimpleImputer(missing_values=np.nan, strategy="mean")
    Y = ip.fit_transform(Y)
    X = ip.fit_transform(X)
    ocv = outer_cv(
        base_model,
        X,
        Y,
        outer_splits=1,
        inner_splits=3,
        verbose=0,
        jobs=3,
        random_state=0,
    )
    dump(ocv, f"{config.model}.pkl")


flexals_config = {
    "method": "grid",
    "parameters": {
        "model": {"values": ["flexals"]},
        "c1": {"values": [1e-3]},
        "c2": {"values": [1e-2]},
        "l1_ratio1": {"values": [0.9]},
        "l1_ratio2": {"values": [0.9]},
    },
}

pmd_config = {
    "method": "grid",
    "parameters": {
        "model": {"values": ["pmd"]},
        "c1": {"values": [0.9]},
        "c2": {"values": [0.9]},
    },
}

rcca_config = {
    "method": "grid",
    "parameters": {
        "model": {"values": ["rcca"]},
        "c1": {"values": [1e-1]},
        "c2": {"values": [1e-1]},
    },
}

pca_config = {
    "method": "grid",
    "parameters": {
        "model": {"values": ["pca"]},
    },
}

pls_config = {
    "method": "grid",
    "parameters": {
        "model": {"values": ["pls"]},
    },
}

if __name__ == "__main__":
    # PCA
    sweep_id = wandb.sweep(pca_config, project="HCP")
    wandb.agent(sweep_id, function=main)
    # PLS
    sweep_id = wandb.sweep(pls_config, project="HCP")
    wandb.agent(sweep_id, function=main)
    # RCCA
    sweep_id = wandb.sweep(rcca_config, project="HCP")
    wandb.agent(sweep_id, function=main)
    # PMD
    sweep_id = wandb.sweep(pmd_config, project="HCP")
    wandb.agent(sweep_id, function=main)
    # FlexALS
    sweep_id = wandb.sweep(flexals_config, project="HCP")
    wandb.agent(sweep_id, function=main)
