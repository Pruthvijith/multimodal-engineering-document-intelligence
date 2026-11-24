"""Offline PNG intake endpoint. Only hash, disposition and confidence enter the audit log."""

import json
import logging
import os
import hmac
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from PIL import UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from .ingest import decode
from .model import load_reader, read_one


def create_app(reader=None, secret=None):
    secret = secret or os.getenv("READER_SECRET", "")
    if len(secret) < 24:
        raise ValueError("Missing reader secret")
    model, tokenizer = reader or load_reader(Path(os.environ["READER_WEIGHTS"]), os.environ["BERT_PATH"])
    app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

    @app.post("/v1/records")
    async def record(request: Request):
        if not hmac.compare_digest(
            request.headers.get("authorization", "").encode(), ("Bearer " + secret).encode()
        ):
            raise HTTPException(401, "Unauthorised")
        if request.headers.get("content-type") != "image/png":
            raise HTTPException(415, "Expected PNG")
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > 1_000_000:
                raise HTTPException(413, "Payload too large")
            data.extend(chunk)
        try:
            image = decode(bytes(data))
        except (ValueError, UnidentifiedImageError, OSError) as exc:
            raise HTTPException(422, "Invalid card") from exc
        result = await run_in_threadpool(read_one, model, tokenizer, image)
        logging.getLogger("records.audit").info(
            json.dumps({k: result[k] for k in ("source_sha256", "disposition", "confidence")})
        )
        return result

    return app
