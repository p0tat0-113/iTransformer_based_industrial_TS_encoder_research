from torch.utils.data import DataLoader

from itransformer.data.datasets import (
    DatasetETTHour,
    DatasetETTMinute,
    DatasetCustom,
    DatasetSolar,
    DatasetPEMS,
    DatasetPred,
)


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
    # Traffic/Weather/ECL/Exchange/custom are csv with date column
    return DatasetCustom


def data_provider(cfg, flag: str):
    data_cls = _select_dataset(cfg.data.name, cfg.data.data_path)

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = cfg.data.freq
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = cfg.data.freq
        data_cls = DatasetPred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = cfg.train.batch_size
        freq = cfg.data.freq

    data_set = data_cls(
        root_path=cfg.data.root_path,
        data_path=cfg.data.data_path,
        flag=flag,
        size=[cfg.data.seq_len, cfg.data.label_len, cfg.data.pred_len],
        features=cfg.data.features,
        target=cfg.data.target,
        timeenc=cfg.data.timeenc,
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
