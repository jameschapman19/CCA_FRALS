from cca_zoo.model_selection._search import param2grid
from sklearn.model_selection import ParameterGrid


def als_param2grid(regressors_params=None):
    params = {"regressors_params": []}
    for regressor_param_grid in regressors_params:
        regressor_params = []
        regressor_param_grid = ParameterGrid(regressor_param_grid)
        for candidate in regressor_param_grid:
            regressor_params.append(candidate)
        params["regressors_params"].append(regressor_params)
    return param2grid(params)
