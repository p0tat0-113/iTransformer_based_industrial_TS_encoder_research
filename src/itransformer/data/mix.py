from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from itransformer.data.datasets import (
    DatasetETTHour,
    DatasetETTMinute,
    DatasetCustom,
    DatasetSolar,
    DatasetPEMS,
    DatasetPred,
)


_CONF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "conf"))
_DATA_CONF_ROOT = os.path.join(_CONF_ROOT, "data")


def _select_dataset(name: str, data_path: str):
    if name in ("ETTh1", "ETTh2"):
        return DatasetETTHour
    if name in ("ETTm1", "ETTm2"):
        return DatasetETTMinute
    if name == "ETT":
        return DatasetETTMinute if "ETTm" in data_path else DatasetETTHour
    if name == "Solar":
        return DatasetSolar
    if name == "PEMS":
        return DatasetPEMS
    return DatasetCustom


def _load_data_cfg(name: str) -> dict:
    path = os.path.join(_DATA_CONF_ROOT, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mix dataset config not found: {path}")
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def _build_loader(cfg, data_cfg: dict, flag: str):
    data_cls = _select_dataset(data_cfg["name"], data_cfg.get("data_path", ""))

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = data_cfg.get("freq", cfg.data.freq)
    elif flag == "val":
        shuffle_flag = False
        drop_last = False
        batch_size = cfg.train.batch_size
        freq = data_cfg.get("freq", cfg.data.freq)
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = data_cfg.get("freq", cfg.data.freq)
        data_cls = DatasetPred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = cfg.train.batch_size
        freq = data_cfg.get("freq", cfg.data.freq)

    data_set = data_cls(
        root_path=data_cfg.get("root_path", cfg.data.root_path),
        data_path=data_cfg.get("data_path", cfg.data.data_path),
        flag=flag,
        size=[cfg.data.seq_len, cfg.data.label_len, cfg.data.pred_len],
        features=data_cfg.get("features", cfg.data.features),
        target=data_cfg.get("target", cfg.data.target),
        timeenc=data_cfg.get("timeenc", cfg.data.timeenc),
        freq=freq,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=cfg.train.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader


class MixBatchLoader:
    def __init__(
        self,
        loaders: List[DataLoader],
        sample_mode: str,
        epoch_size: Optional[int],
        shuffle: bool,
        seed: Optional[int],
        deterministic: bool = False,
    ):
        self.loaders = loaders
        self.sample_mode = sample_mode
        self.epoch_size = epoch_size
        self.shuffle = shuffle
        self.seed = seed
        self.deterministic = deterministic
        self._epoch = 0
        self._length = self._compute_length()
        self._cached_schedule: Optional[List[int]] = None

    def _compute_length(self) -> int:
        lengths = [len(loader) for loader in self.loaders]
        if any(l <= 0 for l in lengths):
            raise ValueError("Mix loader received an empty dataset.")
        if self.epoch_size is not None:
            return int(self.epoch_size)
        if self.sample_mode == "balanced":
            return min(lengths) * len(lengths)
        return sum(lengths)

    def __len__(self) -> int:
        return self._length

    def _build_schedule(self) -> List[int]:
        n = len(self.loaders)
        lengths = [len(loader) for loader in self.loaders]
        total = self._length

        if self.sample_mode == "balanced":
            if self.epoch_size is None:
                per = min(lengths)
                schedule = [i for i in range(n) for _ in range(per)]
            else:
                per = total // n
                rem = total % n
                schedule = [i for i in range(n) for _ in range(per)]
                if rem:
                    schedule.extend(list(range(rem)))
        else:  # proportional
            if self.epoch_size is None:
                schedule = [i for i in range(n) for _ in range(lengths[i])]
            else:
                weights = [l / sum(lengths) for l in lengths]
                rng = random.Random((self.seed or 0) + self._epoch)
                schedule = rng.choices(range(n), weights=weights, k=total)

        if self.shuffle:
            rng = random.Random((self.seed or 0) + self._epoch)
            rng.shuffle(schedule)
        return schedule

    def __iter__(self):
        if not self.deterministic:
            self._epoch += 1
        iters = [iter(loader) for loader in self.loaders]
        if self.deterministic:
            if self._cached_schedule is None:
                self._cached_schedule = self._build_schedule()
            schedule = list(self._cached_schedule)
        else:
            schedule = self._build_schedule()
        for idx in schedule:
            try:
                batch = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self.loaders[idx])
                batch = next(iters[idx])
            yield batch


@dataclass
class MixDatasetInfo:
    name: str
    dataset_names: List[str]
    sensor_ids: None = None
    length: int = 0

    def __len__(self) -> int:
        return self.length


def mix_data_provider(cfg, flag: str):
    if getattr(cfg.metadata, "enabled", False):
        raise ValueError("Mix dataset does not support metadata yet.")

    mix_cfg = cfg.data.mix
    dataset_names = list(mix_cfg.datasets or [])
    if not dataset_names:
        raise ValueError("mix.datasets is required for Mix data.")

    data_cfgs = [_load_data_cfg(name) for name in dataset_names]
    for data_cfg in data_cfgs:
        for key in ("seq_len", "label_len", "pred_len"):
            if data_cfg.get(key) != getattr(cfg.data, key):
                raise ValueError(
                    f"Mix requires identical {key} across datasets "
                    f"(Mix={getattr(cfg.data, key)}, {data_cfg.get('name')}={data_cfg.get(key)})"
                )

    datasets = []
    loaders = []
    for data_cfg in data_cfgs:
        data_set, loader = _build_loader(cfg, data_cfg, flag=flag)
        datasets.append(data_set)
        loaders.append(loader)

    loader = MixBatchLoader(
        loaders=loaders,
        sample_mode=str(mix_cfg.get("sample_mode", "balanced")),
        epoch_size=mix_cfg.get("epoch_size"),
        shuffle=bool(mix_cfg.get("shuffle", True)),
        seed=getattr(cfg.runtime, "seed", None),
        deterministic=(flag == "val"),
    )
    info = MixDatasetInfo(name=cfg.data.name, dataset_names=dataset_names, length=len(loader))
    return info, loader
