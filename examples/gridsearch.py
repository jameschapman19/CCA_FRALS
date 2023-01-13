import numpy as np

from cca_frals import als_param2grid, CCA_FRALS

if __name__ == "__main__":
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)
    from sklearn.linear_model import Ridge, Lasso

    # any sklearn regressor can be used
    regressors = [Ridge(), Lasso()]
    # parameter grid for each regressor
    regressor_params = [{"alpha": [1e-3, 1e-2]}, {"alpha": [1e-2, 5e-2]}]
    # create the grid
    param_grid = als_param2grid(regressor_params)
    from cca_zoo.model_selection import GridSearchCV

    def scorer(estimator, X):
        dim_corrs = estimator.score(X)
        return dim_corrs.sum()

    a = GridSearchCV(
        CCA_FRALS(regressors=regressors, latent_dims=5), param_grid, scoring=scorer
    ).fit((X, Y))
