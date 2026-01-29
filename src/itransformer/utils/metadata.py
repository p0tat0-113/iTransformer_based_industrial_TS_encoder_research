import json
import os
from string import Formatter
from typing import Dict, Iterable, List, Tuple

import torch


def _with_suffix(path: str, suffix: str) -> str:
    if not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.{suffix}{ext}"


def load_metadata_jsonl(path: str) -> Dict[str, Dict]:
    if not path:
        raise ValueError("metadata.path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata.jsonl not found: {path}")

    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sensor_id = str(obj.get("sensor_id"))
            if sensor_id == "None":
                raise ValueError("metadata.jsonl entry missing sensor_id")
            meta[sensor_id] = obj
    return meta


def validate_metadata_jsonl(
    path: str,
    sensor_ids: Iterable[str],
    template: str,
    *,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    errors = []
    warnings = []

    if not os.path.exists(path):
        return [f"metadata.jsonl not found: {path}"], warnings

    seen = set()
    meta_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                errors.append(f"Line {line_no}: invalid JSON ({exc})")
                continue
            if not isinstance(obj, dict):
                errors.append(f"Line {line_no}: JSON must be an object")
                continue
            if "sensor_id" not in obj:
                errors.append(f"Line {line_no}: missing sensor_id")
                continue
            sensor_id = str(obj.get("sensor_id"))
            if sensor_id in seen:
                errors.append(f"Line {line_no}: duplicate sensor_id={sensor_id}")
                continue
            seen.add(sensor_id)
            meta_map[sensor_id] = obj

    sensor_ids = [str(sid) for sid in sensor_ids]
    missing = [sid for sid in sensor_ids if sid not in meta_map]
    if missing:
        errors.append(f"Missing metadata for {len(missing)} sensor_ids (e.g., {missing[:5]})")

    if "{json}" not in template:
        fields = _template_fields(template)
        if fields:
            for sid in sensor_ids:
                meta = meta_map.get(sid, {})
                for field in fields:
                    if field not in meta:
                        msg = f"sensor_id={sid} missing field '{field}'"
                        if strict:
                            errors.append(msg)
                        else:
                            warnings.append(msg)

    return errors, warnings


def _template_fields(template: str) -> List[str]:
    fields = []
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            fields.append(field_name)
    return fields


def serialize_metadata(meta: Dict, template: str, unk_token: str, unk_template: str) -> str:
    if "{json}" in template:
        return template.replace("{json}", json.dumps(meta, ensure_ascii=False))

    values = {}
    for field in _template_fields(template):
        value = meta.get(field, None)
        if value is None or value == "":
            value = unk_template.format(field=field) if unk_template else unk_token
        values[field] = value
    return template.format(**values)


def build_texts(
    sensor_ids: Iterable[str],
    meta_map: Dict[str, Dict],
    template: str,
    unk_token: str,
    unk_template: str,
) -> List[str]:
    texts = []
    for sensor_id in sensor_ids:
        meta = dict(meta_map.get(str(sensor_id), {}))
        meta.setdefault("sensor_id", str(sensor_id))
        text = serialize_metadata(meta, template, unk_token, unk_template)
        texts.append(text)
    return texts


def _embed_texts_gemini(
    texts: List[str],
    *,
    model: str,
    dim: int,
    api_key_env: str,
    batch_size: int = 32,
) -> torch.Tensor:
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key env: {api_key_env}")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "google-genai is required. Install with `pip install google-genai`."
        ) from exc

    client = genai.Client(api_key=api_key)
    embeddings = []

    if batch_size <= 0:
        batch_size = len(texts)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            config = None
            if dim:
                config = types.EmbedContentConfig(output_dimensionality=dim)
            result = client.models.embed_content(
                model=model,
                contents=batch,
                config=config,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini embed_content failed: {exc}") from exc

        batch_embeddings = result.embeddings
        if hasattr(batch_embeddings, "values"):
            batch_embeddings = [batch_embeddings]

        for emb in batch_embeddings:
            if hasattr(emb, "values"):
                vec = emb.values
            elif isinstance(emb, dict) and "values" in emb:
                vec = emb["values"]
            else:
                vec = emb
            embeddings.append(vec)

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
        )

    return torch.tensor(embeddings, dtype=torch.float32)


def load_or_build_embeddings(cfg, sensor_ids: Iterable[str]) -> torch.Tensor:
    if not cfg.metadata.enabled:
        return None

    sensor_ids = [str(sid) for sid in sensor_ids]
    cache_path = cfg.metadata.cache.path
    meta_source = getattr(cfg.model.meta, "source", "real")
    if meta_source == "constant":
        cache_path = _with_suffix(cache_path, "constant")

    if cache_path and os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        cached_ids = [str(sid) for sid in payload.get("sensor_ids", [])]
        cached_emb = payload.get("embeddings")
        if cached_emb is None:
            raise ValueError(f"Invalid cache format: {cache_path}")
        id_to_idx = {sid: i for i, sid in enumerate(cached_ids)}
        missing = [sid for sid in sensor_ids if sid not in id_to_idx]
        if missing:
            raise ValueError(f"Cache missing sensor_ids: {missing[:5]}...")
        order = [id_to_idx[sid] for sid in sensor_ids]
        return cached_emb[order]

    if not cfg.metadata.cache.build:
        raise FileNotFoundError(
            f"Metadata cache not found: {cache_path}. "
            "Set metadata.cache.build=true to build embeddings."
        )
    if meta_source == "constant":
        texts = [cfg.metadata.unk_token for _ in sensor_ids]
    else:
        meta_map = load_metadata_jsonl(cfg.metadata.path)
        texts = build_texts(
            sensor_ids,
            meta_map,
            cfg.metadata.template,
            cfg.metadata.unk_token,
            cfg.metadata.unk_template,
        )

    if cfg.metadata.embedding.provider != "gemini":
        raise ValueError(f"Unsupported embedding provider: {cfg.metadata.embedding.provider}")

    emb = _embed_texts_gemini(
        texts,
        model=cfg.metadata.embedding.model,
        dim=cfg.metadata.embedding.dim,
        api_key_env=cfg.metadata.embedding.api_key_env,
        batch_size=getattr(cfg.metadata.embedding, "batch_size", 32),
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"sensor_ids": sensor_ids, "embeddings": emb}, cache_path)
    return emb
