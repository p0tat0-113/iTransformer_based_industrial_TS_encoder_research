from itransformer.models import ITransformer, PatchITransformer


PATCH_VARIANTS = {"P0", "P1", "P2", "P3", "P4"}


def build_model(cfg):
    patch_enabled = bool(getattr(cfg.model.patch, "enabled", False))
    if cfg.model.variant in PATCH_VARIANTS or patch_enabled:
        return PatchITransformer(cfg)
    return ITransformer(cfg)
