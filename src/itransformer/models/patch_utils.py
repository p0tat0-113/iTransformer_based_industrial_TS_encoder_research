import torch


def build_patch_attn_mask(patch_count: int, n_vars: int, mode: str, local_win: int, device) -> torch.Tensor | None:
    if mode not in ("same_time", "local"):
        return None
    token_count = patch_count * n_vars
    idx = torch.arange(token_count, device=device)
    patch_idx = idx // n_vars
    if mode == "same_time":
        mask_mat = patch_idx[:, None] != patch_idx[None, :]
    else:
        win = max(0, int(local_win))
        mask_mat = (patch_idx[:, None] - patch_idx[None, :]).abs() > win
    return mask_mat.unsqueeze(0).unsqueeze(0)
