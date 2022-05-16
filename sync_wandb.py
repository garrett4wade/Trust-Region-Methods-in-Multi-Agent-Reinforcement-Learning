import os
import torch


def recursive_wandb_sync(path):
    prefix = path
    for file_or_dir in os.listdir(path):
        if os.path.isdir(os.path.join(prefix, file_or_dir)):
            recursive_wandb_sync(os.path.join(prefix, file_or_dir))
        else:
            if file_or_dir.endswith(".wandb"):
                cmd = f"wandb sync --include-online --include-offline --no-include-synced --mark-synced {os.path.join(prefix, file_or_dir)}"
                print(cmd)
                os.system(cmd)


if __name__ == "__main__":
    for gpu in range(torch.cuda.device_count()):
        if os.path.exists(f"/root/happo_{gpu}/scripts/results"):
            recursive_wandb_sync(f"/root/happo_{gpu}/scripts/results")