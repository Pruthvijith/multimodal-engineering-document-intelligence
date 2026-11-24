"""A fixed-format, synthetic inspection card and a bounded local OCR reader."""

import hashlib
import io
import re
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field


class Attributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset: str = Field(pattern=r"^P[0-9]{4}$")
    temperature_c: int = Field(ge=-50, le=200)
    document_type: str = Field(pattern=r"^(inspection|service)$")
    observed_mark: str = Field(pattern=r"^(open|filled)$")


def render(group, kind, mark):
    img = Image.new("L", (520, 168), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=24)
    offset = group % 4
    heading = ("Inspection report", "Service report")[kind]
    temp = 20 + ((group * 17 + kind * 7 + mark) % 65)
    asset = f"P{group:04}"
    for i, line in enumerate([heading, f"Asset {asset}", f"Temperature {temp} C"]):
        draw.text((12 + offset, 8 + i * 46), line, font=font, fill=12)
    x = 444 + offset
    draw.ellipse((x, 54, x + 40, 94), fill=25 if mark else 255, outline=25, width=4)
    return img, dict(
        asset=asset,
        temperature_c=temp,
        document_type=("inspection", "service")[kind],
        observed_mark=("open", "filled")[mark],
    )


def decode(data: bytes):
    if not 0 < len(data) <= 1_000_000:
        raise ValueError("Invalid record size")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (520, 168):
            raise ValueError("Expected 520x168 PNG card")
        return image.convert("L")


def ocr(image):
    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError("Install Tesseract with English trained data")
    buf = io.BytesIO()
    image.crop((0, 0, 398, 168)).save(buf, format="PNG")
    result = subprocess.run(
        [exe, "stdin", "stdout", "-l", "eng", "--psm", "6"],
        input=buf.getvalue(),
        capture_output=True,
        check=True,
        timeout=5,
    )
    return result.stdout.decode().strip()


def extract(text, kind, mark, confidence, threshold=0.80):
    a = re.findall(r"\bAsset\s+(P\d{4})\b", text, re.I)
    t = re.findall(r"\bTemperature\s+(-?\d{1,3})\s*C\b", text, re.I)
    if len(a) != 1 or len(t) != 1 or not 0 <= threshold <= confidence <= 1:
        return None
    fields = dict(asset=a[0].upper(), temperature_c=int(t[0]), document_type=kind, observed_mark=mark)
    try:
        return Attributes(**fields).model_dump()
    except ValueError:
        return None


def fingerprint(image):
    return hashlib.sha256(image.tobytes()).hexdigest()
