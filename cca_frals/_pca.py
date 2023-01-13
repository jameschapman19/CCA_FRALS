from typing import Union, Iterable

import numpy as np
from cca_zoo.models._base import _BaseCCA
from sklearn.decomposition import PCA as PCA_


class CCA_PCA(_BaseCCA):
    def __init__(
        self,
        latent_dims: int = 1,
        scale=True,
        centre=True,
        copy_data=True,
        accept_sparse=False,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        super().__init__(
            latent_dims, scale, centre, copy_data, accept_sparse, random_state
        )

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self.pca_models = [
            PCA_(n_components=self.latent_dims, copy=self.copy_data) for _ in views
        ]
        for pca_model, view in zip(self.pca_models, views):
            pca_model.fit(view)
        self.weights = [pca_model.components_.T for pca_model in self.pca_models]
        return self
