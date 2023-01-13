import wandb
from cca_zoo.model_selection import cross_validate
from sklearn.model_selection import KFold, ShuffleSplit


def outer_cv(
    estimator,
    X,
    Y,
    outer_splits=3,
    inner_splits=3,
    verbose=0,
    jobs=None,
    random_state=0,
):
    if outer_splits == 1:
        outer_splitter = ShuffleSplit(
            n_splits=outer_splits, random_state=random_state, test_size=0.2
        )
    else:
        outer_splitter = KFold(
            n_splits=outer_splits, random_state=random_state, shuffle=True
        )
    for outer_split_number, split in enumerate(outer_splitter.split(X)):
        if inner_splits == 1:
            splitter = ShuffleSplit(
                n_splits=inner_splits, random_state=random_state, test_size=0.2
            )
        else:
            splitter = KFold(
                n_splits=inner_splits, random_state=random_state, shuffle=True
            )
        inner_cv = cross_validate(
            estimator,
            (X[split[0]], Y[split[0]]),
            return_estimator=True,
            return_train_score=True,
            cv=splitter,
            verbose=verbose,
            n_jobs=jobs,
        )
        inner_cv_log(inner_cv, outer_split_number)
    outer_cv = cross_validate(
        estimator,
        (X, Y),
        return_estimator=True,
        return_train_score=True,
        cv=outer_splitter,
        verbose=verbose,
        n_jobs=jobs,
    )
    outer_cv_log(outer_cv)
    return outer_cv


def inner_cv_log(results, outer_split_number):
    for key in ["train_score", "test_score"]:
        # append outer split number to key
        wandb.log({f"fold_{outer_split_number}_{key}_inner": results[key].mean()})


def outer_cv_log(results):
    for key in ["train_score", "test_score", "fit_time"]:
        # log each value associated with key
        for outer_split_number, value in enumerate(results[key]):
            wandb.log({f"fold_{outer_split_number}_{key}_outer": value})
        # log mean of each key
        wandb.log({f"{key}": results[key].mean()})
