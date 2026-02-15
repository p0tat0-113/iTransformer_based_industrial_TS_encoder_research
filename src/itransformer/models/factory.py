from itransformer.models import (
    DLinear,
    ITransformer,
    ITransformerM0,
    Informer,
    PatchITransformer,
    PatchTST,
    TSLibForecastAdapter,
    TiDE,
)


PATCH_VARIANTS = {"P1", "P2", "P3", "P4"}


def build_model(cfg):
    variant = str(getattr(cfg.model, "variant", "") or "")
    if variant == "M0":
        return ITransformerM0(cfg)
    if variant == "TSLIB":
        return TSLibForecastAdapter(cfg)
    if variant == "PatchTST":
        return PatchTST(cfg)
    if variant == "Informer":
        return Informer(cfg)
    if variant == "DLinear":
        return DLinear(cfg)
    if variant == "TiDE":
        return TiDE(cfg)
    if variant == "P0" and not bool(getattr(cfg.model.patch, "enabled", False)):
        return ITransformer(cfg)
    patch_enabled = bool(getattr(cfg.model.patch, "enabled", False))
    if variant in PATCH_VARIANTS or patch_enabled:
        return PatchITransformer(cfg)
    return ITransformer(cfg)
