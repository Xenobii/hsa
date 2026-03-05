import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from model.utils import plot_spec
from pathlib import Path
import torch



def load_checkpoint():
    ckpt_path = Path("./checkpoints/model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt



@hydra.main(version_base="1.3", config_path="config", config_name="config")
def inference(cfg: DictConfig):
    print("--- Inference ---")
    # --- model ---
    model = instantiate(cfg.model)
    ckpt = load_checkpoint()
    model.load_state_dict(ckpt)
    model.eval()

    # -- inference ---
    spec_rec = model.sequential_inference("test/test_wav.WAV")    
    
    # --- plot ---
    plot_spec(spec_rec, k=0, save_path="test/tet.png")
    
    print("--- Inference successful! ---")



if __name__ == "__main__":
    inference()