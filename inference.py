import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from model.utils import plot_spec, visualize_slots
from pathlib import Path
import torch



def load_checkpoint():
    ckpt_path = Path("./checkpoints/epoch_1/model.pt")
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
    rec_a, rec_b, masks_a, masks_b = model.sequential_inference("test/test_wav.WAV")    
    
    # --- plot ---
    plot_spec(rec_a, rec_b, k=0, save_path="test/test.png")
    visualize_slots(rec_a, masks_a, save_path="test/slot_test.png")
    visualize_slots(rec_b, masks_b, save_path="test/slot_test.png")
    
    print("--- Inference successful! ---")



if __name__ == "__main__":
    inference()