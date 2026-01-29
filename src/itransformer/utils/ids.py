from __future__ import annotations


def build_run_id(template: str, *, code: str, dataset: str, variant: str, hparams_tag: str, seed: int) -> str:
    return template.format(
        I=code,
        D=dataset,
        V=variant,
        H=hparams_tag,
        seed=seed,
    )


def build_op_id(template: str, *, code: str, op_code: str, op_hparams: str, on_run_id: str) -> str:
    if op_hparams is None:
        op_hparams = ""
    if not isinstance(op_hparams, str):
        op_hparams = str(op_hparams)
    if op_hparams and op_hparams.startswith("."):
        op_hparams = op_hparams[1:]
    suffix = f".{op_hparams}" if op_hparams else ""
    return template.format(
        I=code,
        code=op_code,
        op_hparams=suffix,
        RunID=on_run_id,
    )


def build_cmp_id(template: str, *, dataset: str, code: str, left: str, right: str) -> str:
    return template.format(
        D=dataset,
        I=code,
        RunID_left=left,
        RunID_right=right,
    )


def build_agg_id(template: str, *, dataset: str, code: str, run_ids: list[str]) -> str:
    joined = ",".join(run_ids)
    return template.format(
        D=dataset,
        I=code,
        RunIDs=joined,
    )
