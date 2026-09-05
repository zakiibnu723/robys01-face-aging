import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageFile, UnidentifiedImageError

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION")
BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")

try:
    s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION and BUCKET_NAME else None
except Exception as exc: 
    s3 = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]")
_HASH_TTL = datetime.timedelta(hours=1)
_RATE_LIMIT_WINDOW = datetime.timedelta(seconds=60)
_RATE_LIMIT_MAX = 12
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_DEDUP_PREFIX = "dedupe/"

_recent_hashes: dict[str, datetime.datetime] = {}
_rate_tracker: dict[str, list[datetime.datetime]] = {}
_lock = threading.Lock()

_REPO_ROOT = Path(__file__).resolve().parent
try:
    _EXAMPLE_ROOTS = {(_REPO_ROOT / "examples").resolve()}
except FileNotFoundError:
    _EXAMPLE_ROOTS = set()


@dataclass
class PreparedUpload:
    path: str
    suffix: str
    format_name: str
    cleanup: bool = True


def _sanitize_stem(raw_stem: str) -> str:
    sanitized = _SANITIZE_PATTERN.sub("_", raw_stem or "")
    sanitized = sanitized.strip("._-")
    trimmed = sanitized[:50]
    return trimmed or "image"


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGB", "L"):
        return image.convert("RGB")

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        alpha = rgba.split()[-1]
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=alpha)
        return canvas

    return image.convert("RGB")


def _mktemp(suffix: str) -> str:
    fd, tmp_path = tempfile.mkstemp(prefix="face-aging_", suffix=suffix)
    os.close(fd)
    return tmp_path


def _save_jpeg(image: Image.Image, *, quality: int) -> PreparedUpload:
    path = _mktemp(".jpg")
    image.save(
        path,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
    )
    return PreparedUpload(path=path, suffix=".jpg", format_name="jpeg")


def _copy_as_webp(local_path: str) -> PreparedUpload:
    path = _mktemp(".webp")
    shutil.copyfile(local_path, path)
    return PreparedUpload(path=path, suffix=".webp", format_name="webp")


def _prepare_upload(local_path: str) -> PreparedUpload:
    prepared: Image.Image | None = None
    source_format = ""
    try:
        with Image.open(local_path) as img:
            img.load()
            source_format = (img.format or "").upper()
            if source_format != "WEBP":
                prepared = _as_rgb(img)

        if source_format == "WEBP":
            return _copy_as_webp(local_path)

        if prepared is None:
            with Image.open(local_path) as img:
                prepared = _as_rgb(img)

        return _save_jpeg(prepared, quality=82)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupted image: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to process image: {exc}") from exc
    finally:
        if prepared is not None:
            prepared.close()


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_marker_key(digest: str) -> str:
    return f"{_DEDUP_PREFIX}{digest}"


def _digest_exists_remotely(digest: str) -> bool:
    if not s3 or not BUCKET_NAME:
        return False
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=_digest_marker_key(digest))
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "403", "Forbidden"}:
            if code in {"403", "Forbidden"}:
                print(
                    json.dumps(
                        {
                            "event": "dedupe_head_forbidden",
                            "digest": digest[:12],
                            "issue": "head_object_forbidden",
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
            return False
        raise


def _store_digest_marker(digest: str, metadata: dict[str, str]) -> None:
    if not s3 or not BUCKET_NAME:
        return
    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=_digest_marker_key(digest),
            Body=b"",
            ContentType="text/plain",
            Metadata=metadata,
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"403", "Forbidden"}:
            print(
                json.dumps(
                    {
                        "event": "dedupe_marker_forbidden",
                        "digest": digest[:12],
                        "issue": "put_object_forbidden",
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            return
        raise


def _prune(now: datetime.datetime) -> None:
    cutoff = now - _HASH_TTL
    stale_hashes = [digest for digest, seen in _recent_hashes.items() if seen < cutoff]
    for digest in stale_hashes:
        _recent_hashes.pop(digest, None)

    for identity, events in list(_rate_tracker.items()):
        filtered = [ts for ts in events if now - ts <= _RATE_LIMIT_WINDOW]
        if filtered:
            _rate_tracker[identity] = filtered
        else:
            _rate_tracker.pop(identity, None)


def _allow_rate(identity: str, now: datetime.datetime) -> bool:
    events = _rate_tracker.setdefault(identity, [])
    events = [ts for ts in events if now - ts <= _RATE_LIMIT_WINDOW]
    if len(events) >= _RATE_LIMIT_MAX:
        _rate_tracker[identity] = events
        return False
    events.append(now)
    _rate_tracker[identity] = events
    return True


def _clean_metadata(values: dict[str, object]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in values.items():
        if not key or value is None:
            continue
        key_norm = key.strip().lower()
        if not key_norm:
            continue
        val = str(value).strip()
        if not val:
            continue
        cleaned[key_norm[:128]] = val[:1024]
    return cleaned


def _is_example_asset(local_path: str) -> bool:
    try:
        resolved = Path(local_path).resolve(strict=False)
    except Exception:
        return False

    for root in _EXAMPLE_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


@dataclass(frozen=True)
class UploadContext:
    session: str | None = None
    ip: str | None = None
    country: str | None = None
    agent: str | None = None
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        return (self.session or "").strip() or (self.ip or "").strip() or "anonymous"

    def metadata(self) -> dict[str, str]:
        data: dict[str, str] = {}
        if self.ip:
            data["ip"] = self.ip.strip()
        if self.session:
            data["session"] = self.session.strip()
        if self.country:
            data["country"] = self.country.strip()
        if self.agent:
            data["ua"] = self.agent.strip()[:256]
        data["identity"] = self.identity
        for key, value in (self.extras or {}).items():
            if key and value:
                data[key.strip().lower()] = value.strip()
        return data


def _log_event(event: str, *, context: UploadContext, path: str, **extra: object) -> None:
    payload: dict[str, object] = {
        "event": event,
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "file": Path(path).name,
    }
    payload.update(context.metadata())
    for key, value in extra.items():
        if value is None:
            continue
        payload[key] = value

    try:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    except Exception as exc:  # pragma: no cover
        print(f"[log-failed] {event} for {path}: {exc}")


def build_context(request: Any | None = None) -> UploadContext:
    if request is None:
        return UploadContext()

    headers: Mapping[str, str] | None = None
    raw_headers = getattr(request, "headers", None)
    if isinstance(raw_headers, Mapping):
        headers = raw_headers  # type: ignore[assignment]
    elif hasattr(raw_headers, "keys"):
        try:
            headers = {k: raw_headers.get(k) for k in raw_headers.keys()}  # type: ignore[attr-defined]
        except Exception:
            headers = None

    session = getattr(request, "session_hash", None)
    ip = None
    country = None
    agent = None
    extras: dict[str, str] = {}

    if headers:
        forwarded = headers.get("x-forwarded-for") or headers.get("x-real-ip")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        country = (
            headers.get("cf-ipcountry")
            or headers.get("x-country-code")
            or headers.get("x-appengine-country")
        )
        agent = headers.get("user-agent")
        referer = headers.get("referer") or headers.get("origin")
        if referer:
            extras["referer"] = referer
        if headers.get("host"):
            extras["host"] = headers.get("host")

    if not ip:
        client = getattr(request, "client", None)
        ip = getattr(client, "host", None) if client else None

    return UploadContext(session=session, ip=ip, country=country, agent=agent, extras=extras)


def imagine(local_path: str, age: int, *, context: UploadContext | None = None) -> str | None:
    context = context or UploadContext()
    now = datetime.datetime.now()

    if _is_example_asset(local_path):
        _log_event("skip_example_asset", context=context, path=local_path)
        print(f"Skipping upload for built-in example: {local_path}")
        return None

    try:
        file_size = os.path.getsize(local_path)
    except OSError as exc:
        print(f"Unable to stat file {local_path}: {exc}")
        _log_event("stat_failed", context=context, path=local_path, error=str(exc))
        return None

    if file_size > _MAX_UPLOAD_BYTES:
        print(f"Skipping upload: {local_path} exceeds {_MAX_UPLOAD_BYTES} bytes")
        _log_event(
            "file_too_large",
            context=context,
            path=local_path,
            size=file_size,
            limit=_MAX_UPLOAD_BYTES,
        )
        return None

    digest = _hash_file(local_path)

    with _lock:
        _prune(now)
        identity = context.identity
        if not _allow_rate(identity, now):
            print(f"Rate limit exceeded for {identity}; skipping upload")
            _log_event(
                "rate_limited",
                context=context,
                path=local_path,
                identity=identity,
                rate_limit=_RATE_LIMIT_MAX,
                window_seconds=int(_RATE_LIMIT_WINDOW.total_seconds()),
            )
            return None

        seen_at = _recent_hashes.get(digest)
        if seen_at and now - seen_at <= _HASH_TTL:
            print(f"Skipping upload for duplicate digest {digest[:12]}… (recent)")
            _log_event(
                "duplicate_recent",
                context=context,
                path=local_path,
                digest=digest[:12],
            )
            return None

        _recent_hashes[digest] = now

    if _digest_exists_remotely(digest):
        print(f"Skipping upload for duplicate digest {digest[:12]}… (remote)")
        _log_event(
            "duplicate_remote",
            context=context,
            path=local_path,
            digest=digest[:12],
        )
        return None

    prepared_upload: PreparedUpload | None = None
    if not s3 or not BUCKET_NAME:
        print()
        return None
    try:
        prepared_upload = _prepare_upload(local_path)

        year = now.strftime("%Y")
        month = now.strftime("%Y-%m")
        today = now.strftime("%Y-%m-%d")
        ts = now.strftime("%H%M%S")
        stem = _sanitize_stem(pathlib.Path(local_path).stem)
        key_prefix = f"{year}/{month}/{today}"

        key = f"{key_prefix}/{age}_{ts}_{stem}{prepared_upload.suffix}"
        base_metadata = context.metadata()
        object_metadata = _clean_metadata(
            {
                **base_metadata,
                "digest": digest,
                "size": file_size,
                "source-age": age,
                "format": prepared_upload.format_name,
                "uploaded-at": now.isoformat(timespec="seconds"),
                "filename": pathlib.Path(local_path).name,
            }
        )
        content_type = "image/webp" if prepared_upload.format_name == "webp" else "image/jpeg"
        s3.upload_file(
            Filename=prepared_upload.path,
            Bucket=BUCKET_NAME,
            Key=key,
            ExtraArgs={
                "Metadata": object_metadata,
                "ContentType": content_type,
            },
        )
        marker_metadata = _clean_metadata(
            {
                **base_metadata,
                "digest": digest,
                "size": file_size,
                "object-key": key,
                "uploaded-at": now.isoformat(timespec="seconds"),
            }
        )
        _store_digest_marker(digest, marker_metadata)
        source_name = pathlib.Path(local_path).name
        print(f"Uploaded {source_name} ({prepared_upload.format_name}) to s3://{BUCKET_NAME}/{key}")

        return key
    except ValueError as exc:
        print(f"Validation error: {exc}")
        _log_event("validation_error", context=context, path=local_path, error=str(exc))
        return None
    except ClientError as exc:
        with _lock:
            _recent_hashes.pop(digest, None)
        print(f"Error uploading (client): {exc}")
        _log_event("client_error", context=context, path=local_path, error=str(exc))
        return None
    except Exception as exc:
        with _lock:
            _recent_hashes.pop(digest, None)
        print(f"Error uploading: {exc}")
        _log_event("unexpected_error", context=context, path=local_path, error=str(exc))
        return None
    finally:
        if prepared_upload and prepared_upload.cleanup and os.path.exists(prepared_upload.path):
            try:
                os.remove(prepared_upload.path)
            except OSError as cleanup_error:
                print(f"Warning: could not delete temp file {prepared_upload.path}: {cleanup_error}")
