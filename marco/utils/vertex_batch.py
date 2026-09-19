import json
from typing import Any
from urllib.parse import urlparse


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri or not uri.startswith('gs://'):
        raise ValueError(f'Expected a gs:// URI, got: {uri!r}')

    parsed = urlparse(uri)
    bucket = parsed.netloc
    blob_path = parsed.path.lstrip('/')

    if not bucket or not blob_path:
        raise ValueError(f'Invalid GCS URI: {uri!r}')

    return bucket, blob_path


def build_gcs_uri(bucket: str, blob_path: str) -> str:
    bucket = (bucket or '').strip()
    blob_path = (blob_path or '').lstrip('/')

    if not bucket or not blob_path:
        raise ValueError('Both bucket and blob_path are required to build a GCS URI')

    return f'gs://{bucket}/{blob_path}'


def extract_batch_response_text(payload: Any) -> str:
    if payload is None:
        return ''

    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, list):
        parts = [extract_batch_response_text(item) for item in payload]
        return ''.join(part for part in parts if part).strip()

    if isinstance(payload, dict):
        if not payload:
            return ''

        text_value = payload.get('text')
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()

        response_value = payload.get('response')
        if response_value is not None:
            nested = extract_batch_response_text(response_value)
            if nested:
                return nested

        candidates = payload.get('candidates')
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            if isinstance(first_candidate, dict):
                content = first_candidate.get('content')
                if isinstance(content, dict):
                    parts = content.get('parts') or []
                    text_parts: list[str] = []
                    for part in parts:
                        if isinstance(part, dict):
                            part_text = part.get('text')
                            if isinstance(part_text, str) and part_text:
                                text_parts.append(part_text)
                    if text_parts:
                        return ''.join(text_parts).strip()

        parts = payload.get('parts')
        if isinstance(parts, list):
            text_parts = []
            for part in parts:
                if isinstance(part, dict):
                    part_text = part.get('text')
                    if isinstance(part_text, str) and part_text:
                        text_parts.append(part_text)
                    if text_parts:
                        return ''.join(text_parts).strip()

        if 'status' in payload and not payload.get('response'):
            return ''

    return str(payload).strip()


def extract_batch_usage_metadata(payload: Any) -> dict[str, int]:
    def _get_value(source: Any, *keys: str) -> int | None:
        if source is None:
            return None

        if isinstance(source, dict):
            for key in keys:
                value = source.get(key)
                if isinstance(value, int):
                    return value
            return None

        for key in keys:
            value = getattr(source, key, None)
            if isinstance(value, int):
                return value
        return None

    if payload is None:
        return {}

    usage_metadata = None
    if isinstance(payload, dict):
        usage_metadata = payload.get('usageMetadata') or payload.get('usage_metadata')
    else:
        usage_metadata = getattr(payload, 'usage_metadata', None) or getattr(payload, 'usageMetadata', None)

    if not usage_metadata:
        return {}

    prompt_tokens = _get_value(usage_metadata, 'promptTokenCount', 'prompt_token_count')
    output_tokens = _get_value(usage_metadata, 'candidatesTokenCount', 'candidates_token_count')
    total_tokens = _get_value(usage_metadata, 'totalTokenCount', 'total_token_count')

    result: dict[str, int] = {}
    if prompt_tokens is not None:
        result['prompt_tokens'] = prompt_tokens
    if output_tokens is not None:
        result['completion_tokens'] = output_tokens
    if total_tokens is not None:
        result['total_tokens'] = total_tokens

    return result


def load_jsonl_from_text(blob_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for line in blob_text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    return records
