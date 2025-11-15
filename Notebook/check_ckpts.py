import torch
import os

os.chdir("Notebook")

ckpt0 = torch.load(
    "FinalPipelineLogs/Study_TEST_TEST2/TEST_TEST2_trial_3_fold_0/version_0/checkpoints/trial-3-fold-0-epoch=01-val_F1=0.819.ckpt",
    map_location="cpu",
    weights_only=False,
)
ckpt1 = torch.load(
    "FinalPipelineLogs/Study_TEST_TEST2/TEST_TEST2_trial_3_fold_1/version_0/checkpoints/trial-3-fold-1-epoch=35-val_F1=0.812.ckpt",
    map_location="cpu",
    weights_only=False,
)

ff0 = [
    k
    for k in ckpt0["state_dict"].keys()
    if "feedforward.network" in k and "weight" in k
]
ff1 = [
    k
    for k in ckpt1["state_dict"].keys()
    if "feedforward.network" in k and "weight" in k
]

print(f"Fold 0 feedforward layers: {len(ff0)}")
print(f"Fold 1 feedforward layers: {len(ff1)}")
print(f"\nFold 0 last 3 layers:")
for layer in ff0[-3:]:
    print(f"  {layer}")
print(f"\nFold 1 last 3 layers:")
for layer in ff1[-3:]:
    print(f"  {layer}")

print(
    f"\nFold 0 hyper_parameters: {ckpt0['hyper_parameters']['params']['FeedForwardParams']}"
)
print(
    f"\nFold 1 hyper_parameters: {ckpt1['hyper_parameters']['params']['FeedForwardParams']}"
)
