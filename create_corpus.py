import os
import csv
import hydra
import h5py
import numpy as np
from tqdm import tqdm
from typing import List
from omegaconf import DictConfig
from torch.utils.data import Dataset
from hydra.utils import instantiate



class MaestroDataset(Dataset):
    def __init__(self, root: str, split: str):
        super().__init__()

        csv_path = os.path.join(root, "maestro-v3.0.0.csv")
        self.files: List[str] = []

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.files.append(os.path.join(root, row['audio_filename']))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> str:
        return self.files[index]
    


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def create_corpus(cfg: DictConfig):
    # --- save dir ---
    f_out = cfg.corpus.corpus_file
    out_dir = os.path.dirname(f_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- dataset ---
    train_dataset = MaestroDataset(root=cfg.corpus.root, split="train")
    valid_dataset = MaestroDataset(root=cfg.corpus.root, split="validation")

    # --- model ---
    model = instantiate(cfg.model)

    with h5py.File(f_out, "w") as h5:
        print(f"Creating corpus...")

        train_data = h5.create_group("train")
        valid_data = h5.create_group("valid")

        train_file_indices = np.arange(len(train_dataset))
        valid_file_indices = np.arange(len(valid_dataset))

        print(f"Instantiated split: train_len={len(train_dataset)}, valid_len={len(valid_dataset)}")
        for idx in tqdm(train_file_indices, desc="Creating training corpus"):
            wav_file = train_dataset[idx]
            
            spec = model.preprocess_input(wav_file).cpu().numpy()

            chunks = spec.shape[0]
            for i in range(chunks):
                group = train_data.create_group(f"{idx:07d}_{i:04d}")
                group.create_dataset("spec", data=spec[i], compression="lzf")

                group.attrs["wav_file"] = wav_file
        
        for idx in tqdm(valid_file_indices, desc="Creating validation corpus"):
            wav_file = valid_dataset[idx]
            
            spec = model.preprocess_input(wav_file).cpu().numpy()

            chunks = spec.shape[0]
            for i in range(chunks):
                group = valid_data.create_group(f"{idx:07d}_{i:04d}")
                group.create_dataset("spec", data=spec[i], compression="lzf")

                group.attrs["wav_file"] = wav_file

    print(f"Finished processing. Dataset saved at {f_out}")
            


if __name__ == "__main__":
    create_corpus()