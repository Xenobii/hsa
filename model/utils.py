import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_spec(*args, k=0, cmap="magma", save_path=None):
    """
    Plot one or more spectrograms in the same figure.
    Usage:
      plot_spec(spec)                   # single spec, default k=0
      plot_spec(spec, 2)                # single spec, k=2 (backwards-compatible)
      plot_spec(spec1, spec2, k=0)      # multiple specs
    Each spec must have shape [K, T, F].
    """
    specs = list(args)
    if len(specs) == 0:
        raise ValueError("No spectrograms provided")

    if isinstance(specs[-1], (int, np.integer)):
        k = int(specs.pop())

    if len(specs) == 0:
        raise ValueError("No spectrograms provided")

    mats = []
    for idx, spec in enumerate(specs):
        if isinstance(spec, torch.Tensor):
            arr = spec.detach().cpu().numpy()
        else:
            arr = np.asarray(spec)

        if arr.ndim != 3:
            raise ValueError(f"spec {idx} must have 3 dims [K,T,F], got {arr.ndim}")

        K, T, F = arr.shape
        if not (0 <= k < K):
            raise IndexError(f"k out of range for spec {idx}: got {k}, expected 0 <= k < {K}")

        mat = arr[k]  # shape [T, F]
        if np.iscomplexobj(mat):
            mat = np.abs(mat)

        mats.append(mat)

    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 3))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # transpose so imshow gets [F, T] (frequency rows, time cols) like before
        im = ax.imshow(mats[i].T, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Spectrogram [{k}] ({i})")
        plt.colorbar(im, ax=ax, label="Amplitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_slots(
            spec_rec: torch.Tensor,
            masks_rec: torch.Tensor,
            save_path: str = None,
            k: int=0,
    ) -> None: 
        # spec_rec (B, T, F)
        # masks_rec (B, K, T, F)
        spec_rec = spec_rec[k, :, :] # (T, F)
        masks_rec = masks_rec[k, :, :, :] # (K, T, F)
        
        # hard slot map
        slot_map = torch.argmax(masks_rec, dim=0) # (T, F)
        K = masks_rec.shape[0]
        
        cmap = plt.cm.get_cmap("Set3", K) 
        colors = cmap(np.arange(K))[:, :3] 
        seg = colors[slot_map] # (T, F, 3)
        
        spec_rgb = np.stack([spec_rec]*3, axis=-1) # (T, F, 3)
        overlay = 0.6*spec_rgb + 0.4*seg

        plt.imshow(overlay.transpose(1, 0, 2), origin="lower")
            
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()    