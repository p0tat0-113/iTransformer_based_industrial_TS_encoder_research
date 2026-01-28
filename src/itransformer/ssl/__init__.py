"""Self-supervised pretraining modules."""

from itransformer.ssl.var_mae import VarMAE
from itransformer.ssl.patch_mae import PatchMAE

__all__ = ["VarMAE", "PatchMAE"]
