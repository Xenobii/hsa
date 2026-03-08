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
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


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



class Trainer():
    def __init__(self, cfg: DictConfig, load_epoch: Optional[int]):
        # --- torch ---
        self.configure_torch(cfg.torch)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- corpus ---
        corpus_train = Corpus(corpus_file=cfg.corpus.corpus_file, split="train")  
        corpus_valid = Corpus(corpus_file=cfg.corpus.corpus_file, split="valid")  
        self.dataloader_train = instantiate(cfg.dataloader, dataset=corpus_train, collate_fn=self.collate_fn)
        self.dataloader_valid = instantiate(cfg.dataloader, dataset=corpus_valid, collate_fn=self.collate_fn)

        # --- model ---
        self.model = instantiate(cfg.model)
        self.model.print_num_params()
        if load_epoch is not None:
            self.load_run(load_epoch)

        # --- optimizer ---
        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())

        # --- scheduler ---
        self.num_epochs = cfg.train.epochs
        self.warmup_steps = cfg.train.warmup_steps
        self.total_steps = self.num_epochs * len(self.dataloader_train)
        self.lr_lambda = self.get_lr_lambda(self.warmup_steps, self.total_steps)
        self.scheduler = instantiate(cfg.scheduler, self.optimizer, self.lr_lambda)


    def configure_torch(self, cfg: DictConfig):
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf_32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf_32


    def get_lr_lambda(self, warmup_steps: int, total_steps: int):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_lambda
    

    def save_run(self, loss: dict, epoch: int) -> None:
        epoch_dir = Path("./checkpoints") / f"epoch_{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        out_file_loss = epoch_dir / "loss.json"
        out_file_ckpt = epoch_dir / "model.pt"

        with open(out_file_loss, "w") as f:
            json.dump(loss, f, indent=2)

        state = self.model.state_dict()
        state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        torch.save(state_cpu, out_file_ckpt)

        print(f"Checkpoint saved successfully at {epoch_dir}")


    def load_run(self, epoch: int) -> None:
        checkpoint_dir = Path("./checkpoints") / f"epoch_{epoch+1}"
        ckpt_file = checkpoint_dir / "model.pt"
        
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
        
        state = torch.load(ckpt_file, map_location=self.device)
        state = state["state_dict"]
        self.model.load_state_dict(state)
        
        print(f"Loaded model weights from {ckpt_file}")


    def collate_fn(self, batch):
        specs = batch
        specs = torch.stack(specs, dim=0)
        return specs


    def train(self) -> None:
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        global_step = 0
        for epoch in range(self.num_epochs):
            # --- train ---
            self.model.train()

            train_loss = 0.0
            n_train = 0
            
            loop_train = tqdm(self.dataloader_train, desc=f"Epoch: {epoch+1}/{self.num_epochs} Train", leave=False)
            for batch in loop_train:
                spec = batch
                spec = spec.to(self.device)
                B = spec.size(0)

                # init
                self.optimizer.zero_grad(set_to_none=True)

                # forward
                loss = self.model.forward_train(spec)

                # backward
                loss.backward()

                # clip norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # step
                self.optimizer.step()
                self.scheduler.step()
                global_step += 1

                n_train += B
                train_loss += loss.detach().item() * B
                loop_train.set_postfix(train_loss=train_loss/n_train, lr=f"{self.scheduler.get_last_lr()[0]:.2e}")

            # --- valid ---
            self.model.eval()

            valid_loss = 0.0
            n_valid = 0

            with torch.no_grad():
                loop_valid = tqdm(self.dataloader_valid, desc=f"Epoch: {epoch+1}/{self.num_epochs} Valid", leave=False)
                for batch in loop_valid:
                    spec = batch
                    spec = spec.to(self.device)
                    B = spec.size(0)

                    loss = self.model.forward_train(spec)

                    n_valid    += B
                    valid_loss += loss.detach().item() * B
                    loop_valid.set_postfix(valid_loss=valid_loss/n_valid)

            avg_train_loss = train_loss / n_train if n_train else 0.0
            avg_valid_loss = valid_loss / n_valid if n_valid else 0.0

            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train_loss={avg_train_loss:.4f}")
            print(f"Valid_loss={avg_valid_loss:.4f}")
            print(f"lr={self.scheduler.get_last_lr()[0]:.2e}")

            # --- save ---
            loss = dict(
                train_loss=avg_train_loss,
                valid_loss=avg_valid_loss,
            )

            self.save_run(self.model, loss, epoch)

        print("Training complete!")



@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # --- torch ---
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()