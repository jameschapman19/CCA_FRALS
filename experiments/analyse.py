# load the models from hcp/results and analyse the model weights

import sys

import pandas as pd
from joblib import load
from sklearn.impute import SimpleImputer

sys.path.append("C:/Users/chapm/PycharmProjects/braindata")
from braindata import HCP
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(font_scale=1)
PLOT_BRAIN = True
PLOT_BEHAVIOUR = False
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
data = HCP(connectome_size=25)
models = ["pca", "pmd", "rcca", "flexals"]
model_title_dict = {
    "pls": "PLS",
    "pmd": "PMD",
    "rcca": "RCCA",
    "pca": "PCA",
    "flexals": "FRALS",
}
loading_fig, loading_axs = plt.subplots(
    len(models), 1, figsize=(30, 5 * len(models)), sharex=True
)
for i, model in enumerate(models):
    try:
        results = load(
            f"C:/Users/chapm/PycharmProjects/FlexibleALSCCA/experiments/{model}.pkl"
        )
    except FileNotFoundError:
        continue
    try:
        estimator = results["estimator"][0].best_estimator_
        behaviour_weights = results["estimator"][0].best_estimator_.weights[1]
        brain_weights = results["estimator"][0].best_estimator_.weights[0]
    except:
        estimator = results["estimator"][0]
        behaviour_weights = results["estimator"][0].weights[1]
        brain_weights = results["estimator"][0].weights[0]
    # flip the pca behaviour weights
    if model == "pca":
        behaviour_weights = -behaviour_weights
    # normalize behaviour weights
    behaviour_weights = behaviour_weights / np.linalg.norm(behaviour_weights)
    base_df = data.y_labels
    base_df["weights"] = behaviour_weights
    # get data.Y and use scikit imputer to mean impute
    Y = data.Y
    ip = SimpleImputer(missing_values=np.nan, strategy="mean")
    Y = ip.fit_transform(Y)
    Y -= np.mean(Y, axis=0)
    Y /= np.linalg.norm(Y, axis=0)
    base_df["loadings"] = np.corrcoef(Y, Y @ behaviour_weights, rowvar=False)[
        : Y.shape[1], Y.shape[1] :
    ]
    brain_loadings = np.corrcoef(
        data.X, results["estimator"][0].transform((data.X, Y))[0], rowvar=False
    )[: data.X.shape[1], data.X.shape[1] :]
    if PLOT_BEHAVIOUR:
        att = "loadings"
        # sort base_df on att
        base_df = base_df.sort_values(att, ascending=False)
        top_df = base_df.head(15)
        bottom_df = base_df.tail(15)
        # concatenate top and bottom and do the same plot
        df = pd.concat([top_df, bottom_df])
        # make text on ylabels bigger
        sns.barplot(
            y="Label",
            x=att,
            hue="Category",
            data=df,
            dodge=False,
            hue_order=HUE_ORDER,
            ax=loading_axs[i],
        )
        loading_axs[i].set_title(f"{model_title_dict[model]} {att}", fontsize=30)
        # labels = [textwrap.fill(label.get_text(), 40) for label in loading_axs[i].get_xticklabels()]
        # loading_axs[i].set_xticklabels(labels)
        loading_axs[i].set_xlim(-1, 1)
    if PLOT_BRAIN:
        fig = data.plot_connectome_weights(
            np.squeeze(brain_loadings),
            title=f"{model_title_dict[model]} Brain Loadings",
        )
        plt.savefig(
            f"C:/Users/chapm/PycharmProjects/FlexibleALSCCA/experiments/hcp/results/{model}_brain_loadings.png"
        )
    # store the model scores from model.transform in an array which will be used to plot the model similarities as a heatmap
    brain_scores_, behaviour_scores_ = estimator.transform((data.X, Y))
    # make brain scores_ and behaviour scores_ 2d if not already
    if len(brain_scores_.shape) == 1:
        brain_scores_ = brain_scores_[:, np.newaxis]
    if len(behaviour_scores_.shape) == 1:
        behaviour_scores_ = behaviour_scores_[:, np.newaxis]

    # append the model scores to the array
    try:
        brain_scores = np.hstack((brain_scores, brain_scores_))
    except:
        brain_scores = brain_scores_
    try:
        behaviour_scores = np.hstack((behaviour_scores, behaviour_scores_))
    except:
        behaviour_scores = behaviour_scores_
plt.tight_layout()
plt.savefig(
    f"C:/Users/chapm/PycharmProjects/FlexibleALSCCA/experiments/hcp/results/all_top_and_bottom_loadings.png"
)
# get list of model titles associated with models in models list
model_titles = [model_title_dict[model] for model in models]
# plot the model similarities as a heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(
    np.abs(np.corrcoef(brain_scores, rowvar=False)),
    xticklabels=model_titles,
    yticklabels=model_titles,
    cmap="viridis",
)
plt.title("Brain model similarities", fontsize=48)
plt.savefig(
    f"C:/Users/chapm/PycharmProjects/FlexibleALSCCA/experiments/hcp/results/brain_model_similarities.png"
)
plt.figure(figsize=(20, 20))
sns.heatmap(
    np.abs(np.corrcoef(behaviour_scores, rowvar=False)),
    xticklabels=model_titles,
    yticklabels=model_titles,
    cmap="viridis",
)
plt.title("Behaviour model similarities", fontsize=48)
plt.savefig(
    f"C:/Users/chapm/PycharmProjects/FlexibleALSCCA/experiments/hcp/results/behaviour_model_similarities.png"
)
print()
