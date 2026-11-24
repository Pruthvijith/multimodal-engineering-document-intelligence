# Engineering inspection card reader

Read a fixed-layout inspection card into four typed fields. The system combines actual offline OCR, a fine-tuned BERT text encoder and a convolutional image encoder. Uncertain or invalid records enter human review.

Independent, AI-assisted portfolio research milestone · v0.1.0 · Python 3.12 · CPU reference environment.

## Measured outcome

32/32 held-out synthetic cards passed the 0.80 confidence threshold with all four fields correct. These are controlled images with clean text and known crop locations. Accuracy on field photographs, rotated scans or novel layouts remains unknown.

## Architecture and data flow

PNG validation enforces the supported dimensions, format and size. Tesseract reads the text crop. A BERT encoder and small CNN jointly predict the record type and visual mark. A strict field parser extracts the asset identifier and temperature. A Pydantic schema and minimum head-confidence threshold determine acceptance. The HTTP endpoint writes only the source hash, confidence and disposition to the audit log.

## Dataset and model usage contract

128 original synthetic cards from 32 asset IDs. Each asset has two record types and two visual marks. Training used 96 records from 24 assets. Testing used the remaining eight assets and 32 records. A single renderer produces all examples. Google BERT L-2 H-128 A-2 supplies the pinned text checkpoint.

The supported purpose is offline research and human-reviewed engineering decision support. Aviation design approval, operational aircraft decisions and regulated certification are outside the validated scope.

## Reproduce

Run from this checkout on Linux x86-64. The reference CPU was an AMD EPYC 9V74. The portfolio index includes a downloader for the pinned models and UCI input.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -c constraints-cpu.txt -r requirements-dev.txt
python -m records.model work/cards
```

## Testing and deployment

```bash
python -m pytest --cov=records --cov-report=term-missing
python -m ruff check .
python -m mypy records
```

Install Tesseract and its English language data. Set BERT_PATH to the staged local base checkpoint for offline operation. The run writes reader.safetensors and report.json. To serve it, set READER_WEIGHTS to the generated weights, BERT_PATH to the pinned base, and READER_SECRET to a random secret of at least 24 characters. Run python -m uvicorn records.api:create_app --factory --host 127.0.0.1. POST one image/png body to /v1/records with bearer authentication. Bound traffic and concurrent OCR processes at the hosting boundary before any multi-user use.

## Operations and next validation

A missing field, duplicate asset string or confidence below threshold returns review. Preserve the source hash to trace a decision. A systematic increase in review rate should trigger sampling by scanner, layout and lighting condition. Add negative tests before allowing a new layout.

See [decisions and interview guide](docs/review.md) and the measured files in [reports](reports/). Hosted CI and production qualification require their own evidence. For dependencies and known advisories see the portfolio index security report.

## Provenance and licensing

Original code and original fixtures use Apache-2.0. Upstream models, data and third-party executables retain their own licences. External weights and source archives are fetched separately. References follow.

- https://huggingface.co/google/bert_uncased_L-2_H-128_A-2
- https://github.com/tesseract-ocr/tesseract

## Local verification

4 tests passed. Statement coverage of the records package was 96.61%. Ruff and mypy passed. See [verification record](reports/verification.json).
