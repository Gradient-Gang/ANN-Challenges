import torch

ckpt0_path = "FinalPipelineLogs/Study_TEST_TEST2/TEST_TEST2_trial_3_fold_0/version_0/checkpoints/trial-3-fold-0-epoch=01-val_F1=0.819.ckpt"
ckpt1_path = "FinalPipelineLogs/Study_TEST_TEST2/TEST_TEST2_trial_3_fold_1/version_0/checkpoints/trial-3-fold-1-epoch=35-val_F1=0.812.ckpt"

ckpt0 = torch.load(ckpt0_path, map_location="cpu")
ckpt1 = torch.load(ckpt1_path, map_location="cpu")

print("=" * 60)
print("CHECKPOINT 0 (fold 0, epoch 01)")
print("=" * 60)
print(f"Epoch: {ckpt0.get('epoch', 'N/A')}")
print(f"Global step: {ckpt0.get('global_step', 'N/A')}")
print(f"Keys: {list(ckpt0.keys())}")

state_dict_0 = ckpt0["state_dict"]
ff_layers_0 = [k for k in state_dict_0.keys() if "feedforward" in k]
print(f"\nFeedforward layers: {len([k for k in ff_layers_0 if 'weight' in k])}")
print(f"Last feedforward layer: {max([k for k in ff_layers_0 if 'weight' in k])}")

print("\n" + "=" * 60)
print("CHECKPOINT 1 (fold 1, epoch 35)")
print("=" * 60)
print(f"Epoch: {ckpt1.get('epoch', 'N/A')}")
print(f"Global step: {ckpt1.get('global_step', 'N/A')}")

state_dict_1 = ckpt1["state_dict"]
ff_layers_1 = [k for k in state_dict_1.keys() if "feedforward" in k]
print(f"\nFeedforward layers: {len([k for k in ff_layers_1 if 'weight' in k])}")
print(f"Last feedforward layer: {max([k for k in ff_layers_1 if 'weight' in k])}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
if len([k for k in ff_layers_0 if "weight" in k]) != len(
    [k for k in ff_layers_1 if "weight" in k]
):
    print("❌ DIFFERENT ARCHITECTURES!")
    print(f"   Fold 0: {len([k for k in ff_layers_0 if 'weight' in k])} layers")
    print(f"   Fold 1: {len([k for k in ff_layers_1 if 'weight' in k])} layers")
    print("\nThis confirms: These checkpoints were NOT trained in the same trial run.")
    print("Possible causes:")
    print("- Trial was interrupted and resumed with different hyperparameters")
    print("- Database params were modified after training")
    print("- Checkpoints were manually renamed/moved")
else:
    print("✓ Same architecture")
