"""Dataset loaders for iTransformer."""

from itransformer.data.factory import data_provider
from itransformer.data.datasets import (
    DatasetETTHour,
    DatasetETTMinute,
    DatasetCustom,
    DatasetSolar,
    DatasetPEMS,
    DatasetPred,
)

__all__ = [
    "data_provider",
    "DatasetETTHour",
    "DatasetETTMinute",
    "DatasetCustom",
    "DatasetSolar",
    "DatasetPEMS",
    "DatasetPred",
]
