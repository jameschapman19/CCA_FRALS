from PIL import Image

models = ["pca", "pmd", "rcca", "flexals"]
behaviour_imgs = []
brain_imgs = []
sim_imgs = [
    Image.open(f"hcp/results/behaviour_model_similarities.png"),
    Image.open(f"hcp/results/brain_model_similarities.png"),
]
for model in models:
    behaviour_imgs.append(
        Image.open(f"hcp/results/{model}_top_and_bottom_loadings.png")
    )
    brain_imgs.append(Image.open(f"hcp/results/{model}_brain_loadings.png"))
result = Image.new(
    "RGB",
    (behaviour_imgs[0].size[0], behaviour_imgs[0].size[1] * len(models)),
    color="white",
)
for i, img in enumerate(behaviour_imgs):
    result.paste(img, (0, i * behaviour_imgs[0].size[1]))
result.save("hcp/results/all_behaviour_loadings.png")
# combine brain imgs on one side and similarity imgs on the other into on figure called results2.png
results2 = Image.new(
    "RGB",
    (brain_imgs[0].size[0] + sim_imgs[0].size[0], sim_imgs[0].size[1] * 2),
    color="white",
)
for i, img in enumerate(brain_imgs):
    results2.paste(img, (0, 2 * int(i * sim_imgs[0].size[1] / len(brain_imgs))))
for i, img in enumerate(sim_imgs):
    results2.paste(img, (sim_imgs[0].size[0], i * sim_imgs[0].size[1]))
results2.save("hcp/results/all_brain_sim.png")
