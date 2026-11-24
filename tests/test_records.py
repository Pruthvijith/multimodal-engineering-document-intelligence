import io
import pytest
from PIL import Image
from records.ingest import decode, extract, fingerprint, render, ocr
from records.model import pixels, decode_heads, Reader, read_one
import torch
from transformers import BertConfig, BertModel


def test_validation_review_and_ocr():
    img, truth = render(27, 1, 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert fingerprint(decode(buf.getvalue())) == fingerprint(img)
    text = ocr(img)
    assert extract(text, "service", "open", 0.98) == truth
    for inp, conf in [
        (text, 0.6),
        (text + "\nAsset P0001", 0.99),
        ("Temperature 30 C", 0.99),
        ("Asset P0030 Temperature 999 C", 0.99),
    ]:
        assert extract(inp, "service", "open", conf) is None
    with pytest.raises(ValueError):
        decode(b"")
    buf = io.BytesIO()
    Image.new("L", (10, 10)).save(buf, format="PNG")
    with pytest.raises(ValueError):
        decode(buf.getvalue())


def test_head_shapes_and_review(tmp_path):
    from transformers import BertTokenizerFast

    (tmp_path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ninspection\nasset\n")
    tok = BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    BertModel(
        BertConfig(
            vocab_size=7, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
        )
    ).save_pretrained(tmp_path)
    torch.manual_seed(3)
    model = Reader(tmp_path).eval()
    img, _ = render(2, 0, 1)
    assert tuple(pixels(img).shape) == (1, 20, 20)
    assert decode_heads(torch.tensor([[5.0, 0.0], [0.0, 5.0]]))[:2] == ("inspection", "filled")
    result = read_one(model, tok, img)
    assert result["disposition"] == "review"


def test_retrain_reload_and_extract(tmp_path):
    from transformers import BertTokenizerFast
    from records.model import run, load_reader
    import json

    src = tmp_path / "model"
    src.mkdir()
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "inspection",
        "service",
        "report",
        "asset",
        "temperature",
        "c",
    ]
    (src / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(src / "vocab.txt")).save_pretrained(src)
    BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    ).save_pretrained(src)
    out = tmp_path / "experiment"
    run(src, out)
    report = json.loads((out / "report.json").read_text())
    reader, tok = load_reader(out / "reader.safetensors", src)
    img, _ = render(24, 0, 0)
    assert read_one(reader, tok, img)["attributes"] == report["cases"][0]["attributes"]
    assert report["test"] == 32


def test_api_intake(monkeypatch):
    from fastapi.testclient import TestClient
    from records import api

    buf = io.BytesIO()
    render(1, 1, 1)[0].save(buf, format="PNG")
    secret = "s" * 32
    monkeypatch.setattr(
        api,
        "read_one",
        lambda *args: dict(attributes=None, confidence=0.3, source_sha256="a" * 64, disposition="review"),
    )
    client = TestClient(api.create_app((None, None), secret))
    headers = {"Authorization": "Bearer " + secret, "Content-Type": "image/png"}
    assert (
        client.post("/v1/records", content=buf.getvalue(), headers=headers).json()["disposition"] == "review"
    )
    assert client.post("/v1/records").status_code == 401
    assert client.post("/v1/records", headers={"Authorization": "Bearer " + secret}).status_code == 415
    assert client.post("/v1/records", headers=headers, content=b"x").status_code == 422
    assert client.post("/v1/records", headers=headers, content=b"x" * 1000001).status_code == 413
