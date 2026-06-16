import json
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class MetadataRenderOptions:
    enable_column_description: bool = False
    enable_column_type: bool = False
    enable_sample_values: bool = False
    sample_values_max_items: int = 2
    sample_value_max_chars: int = 120


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _parse_json_like_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _compress_complex(value: Any, max_chars: int) -> Any:
    if isinstance(value, list):
        if not value:
            return []
        return [_compress_complex(value[0], max_chars)]

    if isinstance(value, dict):
        compact = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 3:
                compact["..."] = "..."
                break
            compact[key] = _compress_complex(item, max_chars)
        return compact

    if isinstance(value, str):
        return _truncate_text(value, max_chars)

    return value


def _field_name_only(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                out[key] = _field_name_only(item)
            else:
                out[key] = "<value>"
        return out

    if isinstance(value, list):
        if not value:
            return []
        return [_field_name_only(value[0])]

    return "<value>"


def _serialize_value(value: Any, max_chars: int) -> str:
    if isinstance(value, str):
        truncated = _truncate_text(value, max_chars)
        return json.dumps(truncated, ensure_ascii=False)
    try:
        return _truncate_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), max_chars)
    except Exception:
        return _truncate_text(str(value), max_chars)


def normalize_sample_values(
    sample_values: Optional[List[Any]],
    column_type: Optional[str],
    max_items: int,
    max_chars: int,
) -> List[str]:
    values = sample_values or []
    if max_items <= 0:
        return []

    normalized: List[str] = []
    seen = set()
    lowered_type = str(column_type or "").lower()
    is_variant_like = "variant" in lowered_type or "json" in lowered_type

    if is_variant_like:
        variant_values: List[Any] = []
        for raw_value in values:
            if raw_value is None:
                continue

            parsed = _parse_json_like_string(raw_value) if isinstance(raw_value, str) else raw_value
            marker = _serialize_value(parsed, max_chars * 4)
            if marker in seen:
                continue
            seen.add(marker)
            variant_values.append(parsed)
            if len(variant_values) >= max_items:
                break

        if not variant_values:
            return []

        # 1) Try full structural samples and reduce sample count first.
        while variant_values:
            full_encoded = [_serialize_value(v, max_chars) for v in variant_values]
            if len(", ".join(full_encoded)) <= max_chars:
                return full_encoded
            if len(variant_values) > 1:
                variant_values = variant_values[:-1]
                continue

            # 2) Still too long with one sample -> keep field names only.
            field_only = [_field_name_only(variant_values[0])]
            field_only_encoded = [_serialize_value(field_only[0], max_chars)]
            if len(", ".join(field_only_encoded)) <= max_chars:
                return field_only_encoded

            # 3) Final fallback: hard truncate serialized field-name skeleton.
            return [_truncate_text(field_only_encoded[0], max_chars)]

        return []

    for raw_value in values:
        if raw_value is None:
            continue

        parsed = _parse_json_like_string(raw_value) if isinstance(raw_value, str) else raw_value
        candidate = _compress_complex(parsed, max_chars) if is_variant_like else parsed
        encoded = _serialize_value(candidate, max_chars)

        if encoded in seen:
            continue

        seen.add(encoded)
        normalized.append(encoded)
        if len(normalized) >= max_items:
            break

    return normalized


def build_column_annotation(
    column_type: Optional[str],
    description: Optional[str],
    sample_values: Optional[List[Any]],
    options: MetadataRenderOptions,
) -> str:
    extras: List[str] = []

    if options.enable_column_type and column_type:
        extras.append(f"type={column_type}")

    if options.enable_column_description and description:
        desc = _truncate_text(description.strip(), options.sample_value_max_chars)
        if desc:
            extras.append(f"desc={desc}")

    if options.enable_sample_values:
        samples = normalize_sample_values(
            sample_values=sample_values,
            column_type=column_type,
            max_items=options.sample_values_max_items,
            max_chars=options.sample_value_max_chars,
        )
        if samples:
            extras.append(f"samples=[{'; '.join(samples)}]")

    if not extras:
        return ""

    return " {" + " | ".join(extras) + "}"


def format_examples(
    sample_values: Optional[List[Any]],
    column_type: Optional[str],
    max_items: int,
    max_chars: int,
) -> str:
    samples = normalize_sample_values(
        sample_values=sample_values,
        column_type=column_type,
        max_items=max_items,
        max_chars=max_chars,
    )
    if not samples:
        return ""
    return f"eg=[{', '.join(samples)}]"


def render_column_line(
    name: str,
    column_type: Optional[str],
    description: Optional[str],
    sample_values: Optional[List[Any]],
    options: MetadataRenderOptions,
) -> str:
    dtype = str(column_type or "TEXT")
    segments: List[str] = [f"{name}: {dtype}"]

    if options.enable_column_description:
        desc = (description or "").strip()
        segments.append(desc if desc else "NULL")

    if options.enable_sample_values:
        example_block = format_examples(
            sample_values=sample_values,
            column_type=column_type,
            max_items=options.sample_values_max_items,
            max_chars=options.sample_value_max_chars,
        )
        if example_block:
            segments.append(example_block)

    return " | ".join(segments)
