import copy
from typing import Union, Iterable

import numpy as np
from cca_zoo.models._iterative._base import _BaseIterative, _default_initializer
from cca_zoo.utils import _check_views, _process_parameter
from sklearn.base import RegressorMixin


class CCA_FRALS(_BaseIterative):
    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        deflation="cca",
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-3,
        regressors: Union[Iterable[RegressorMixin], RegressorMixin] = None,
        regressors_params=None,
        verbose=0,
    ):
        self.regressors = regressors
        if regressors_params is None:
            self.regressors_params = [
                regressor.get_params() for regressor in self.regressors
            ]
        else:
            self.regressors_params = regressors_params
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            deflation=deflation,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )
        initializer_scores = np.stack(initializer.fit_transform(views))
        residuals = copy.deepcopy(list(views))
        self.track = {"objective": {}}
        # if all regressors are multioutput then no need for deflation
        if self.deflation:
            raise NotImplementedError
        else:
            self.regressors, self.track["objective"] = self._fit(
                residuals, initializer_scores, self.regressors
            )
        return self

    def _check_params(self):
        self.regressors_params = _process_parameter(
            "regressors_params", self.regressors_params, {}, self.n_views
        )
        self.regressors = [
            regressor.set_params(**regressor_params)
            for regressor, regressor_params in zip(
                self.regressors, self.regressors_params
            )
        ]
        if (
            all(
                [
                    regressor._get_tags().get("multioutput", True)
                    for regressor in self.regressors
                ]
            )
            or self.latent_dims == 1
        ):
            self.deflation = False
        else:
            self.deflation = True

    def _update(self, views, scores, regressors):
        target = self._update_target(scores)
        for view_index, (view, regressor) in enumerate(zip(views, self.regressors)):
            regressor.fit(view, target)
            pred = regressor.predict(view)
            if pred.ndim == 1:
                pred = pred[:, np.newaxis]
            scores[view_index] = pred
        return scores, regressors

    def _update_target(self, scores):
        target = scores.sum(axis=0)
        if target.ndim == 1:
            target = target[:, np.newaxis]
        U, _, Vt = np.linalg.svd(target, full_matrices=False)
        return U @ Vt * np.sqrt(self.n)

    def transform(self, views: Iterable[np.ndarray], **kwargs):
        """

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        transformed_views : list of numpy arrays

        """
        views = _check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale_transform(views)
        transformed_views = []
        if self.deflation:
            raise NotImplementedError
        else:
            for view, regressor in zip(views, self.regressors):
                transformed_view = regressor.predict(view)
                transformed_views.append(transformed_view)
        return transformed_views

    @property
    def weights(self):
        weights = []
        for regressor in self.regressors:
            try:
                weights.append(regressor.coef_)
            except:
                weights.append(None)
        return weights
