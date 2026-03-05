import logging
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from hydra.utils import instantiate
from pathlib import Path
import json
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset 
import h5py
from typing import Tuple
import math
import torch
import torch.nn as nn

from model.hsa import HSA



class Corpus(Dataset):
    def __init__(self, corpus_file: str, split=None, **kwargs):
        super().__init__()

        self.h5_path   = corpus_file
        self.split     = split

        with h5py.File(self.h5_path, "r") as h5:
            self.keys = sorted(list(h5[split].keys()))

        self._h5 = None

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple:
        h5 = self._get_h5()
        grp = h5[self.split][self.keys[idx]]
        spec = grp["spec"][()]
        spec = torch.tensor(spec, dtype=torch.float32)
        return spec



def save_run(model: nn.Module, loss: dict, epoch: int) -> None:
    epoch_dir = Path("./checkpoints")
    epoch_dir.mkdir(parents=True, exist_ok=True)

    out_file_loss = epoch_dir / "loss.json"
    out_file_ckpt = epoch_dir / "model.pt"

    with open(out_file_loss, "w") as f:
        json.dump(loss, f, indent=2)

    state = model.state_dict()
    state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
    torch.save(state_cpu, out_file_ckpt)

    print(f"Checkpoint saved successfully at {epoch_dir}")


def get_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def train(
        model: nn.Module,
        dataloader_train: DataLoader,
        dataloader_valid: DataLoader,
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_epochs: int,
    ) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    global_step = 0
    for epoch in range(num_epochs):
        # --- train ---
        model.train()

        train_loss = 0.0
        n_train = 0
        
        loop_train = tqdm(dataloader_train, desc=f"Epoch: {epoch+1}/{num_epochs} Train", leave=False)
        for batch in loop_train:
            spec = batch
            spec = spec.to(device)
            B = spec.size(0)

            # init
            optimizer.zero_grad(set_to_none=True)

            # forward
            loss = model.forward_train(spec)

            # backward
            loss.backward()

            # clip norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # step
            optimizer.step()
            scheduler.step()
            global_step += 1

            n_train += B
            train_loss += loss.detach().item() * B
            loop_train.set_postfix(train_loss=train_loss/n_train, lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # --- valid ---
        model.eval()

        valid_loss = 0.0
        n_valid = 0

        with torch.no_grad():
            loop_valid = tqdm(dataloader_valid, desc=f"Epoch: {epoch+1}/{num_epochs} Valid", leave=False)
            for batch in loop_valid:
                spec = batch
                spec = spec.to(device)
                B = spec.size(0)

                loss = model.forward_train(spec)

                n_valid    += B
                valid_loss += loss.detach().item() * B
                loop_valid.set_postfix(valid_loss=valid_loss/n_valid)

        avg_train_loss = train_loss / n_train if n_train else 0.0
        avg_valid_loss = valid_loss / n_valid if n_valid else 0.0

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train_loss={avg_train_loss:.4f}")
        print(f"Valid_loss={avg_valid_loss:.4f}")
        print(f"lr={scheduler.get_last_lr()[0]:.2e}")

        # --- save ---
        loss = dict(
            train_loss=avg_train_loss,
            valid_loss=avg_valid_loss,
        )

        save_run(model, loss, epoch)

    print("Training complete!")


def configure_torch(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf_32
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf_32


def collate_fn(batch):
    specs = batch
    specs = torch.stack(specs, dim=0)
    return specs


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # --- torch ---
    configure_torch(cfg.torch)

    # --- corpus ---
    corpus_train = Corpus(corpus_file=cfg.corpus.corpus_file, split="train")  
    corpus_valid = Corpus(corpus_file=cfg.corpus.corpus_file, split="valid")  
    dataloader_train = instantiate(cfg.dataloader, dataset=corpus_train, collate_fn=collate_fn)
    dataloader_valid = instantiate(cfg.dataloader, dataset=corpus_valid, collate_fn=collate_fn)

    # --- model ---
    model = HSA()
    model.print_num_params()

    # --- optimizer ---
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # --- scheduler ---
    num_epochs = cfg.train.epochs
    warmup_steps = cfg.train.warmup_steps
    total_steps = num_epochs * len(dataloader_train)
    lr_lambda = get_lr_lambda(warmup_steps, total_steps)
    scheduler = instantiate(cfg.scheduler, optimizer, lr_lambda)

    # -- train ---
    train(model, dataloader_train, dataloader_valid, optimizer, scheduler, cfg.train.epochs)


if __name__ == "__main__":
    main()