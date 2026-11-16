import base64
import io
import json
import math
import shutil
import tempfile
from contextlib import closing, contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image, UnidentifiedImageError

FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
           ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP", ".webp": "WEBP"}
# PDFium calls must be serialized even when requests open different documents.
PDFIUM_LOCK = Lock()


def detect(image):
    original = image
    scale = min(1, 1200 / max(image.size))
    if scale < 1:
        size = tuple(max(1, round(dimension * scale)) for dimension in image.size)
        image = image.resize(size, Image.Resampling.BOX)
    gray = np.asarray(image.convert("L"))
    edges = cv2.Canny(gray, 60, 160)
    width = gray.shape[1]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 65,
                            minLineLength=max(35, width // 6), maxLineGap=16)
    if lines is None:
        return None
    votes = {}
    # OpenCV returns (N, 1, 4) or (N, 4), depending on the version.
    for x, y, xx, yy in lines.reshape(-1, 4):
        if x == xx:
            continue
        angle = math.degrees(math.atan((yy - y) / (xx - x)))
        if abs(angle) <= 10:
            votes.setdefault(round(angle * 4), []).append((angle, abs(xx - x)))
    if not votes:
        return None
    peak = max(votes, key=lambda k: sum(w for j in range(k-1, k+2)
                                        for _, w in votes.get(j, [])))
    group = [pair for j in range(peak-1, peak+2) for pair in votes.get(j, [])]
    if len(group) < 5:
        return None
    estimate = sum(a*w for a, w in group) / sum(w for _, w in group)
    return refine_angle(original, estimate)


def refine_angle(image, estimate):
    # Horizontal staff edges concentrate into fewer rows at the correct angle.
    gray = image.convert("L")
    gray.thumbnail((3200, 3200), Image.Resampling.BOX)
    pixels = np.asarray(gray)
    height, width = pixels.shape
    if min(height, width) < 20:
        return round(estimate, 2)
    vertical = cv2.Sobel(pixels, cv2.CV_32F, 0, 1, ksize=3)
    horizontal = cv2.Sobel(pixels, cv2.CV_32F, 1, 0, ksize=3)
    edges = np.maximum(np.abs(vertical) - np.abs(horizontal), 0)
    # Exclude page boundaries, which may have a different angle from the staff.
    edges = edges[height//20:height-height//20, width//20:width-width//20]
    y, x = np.nonzero(edges > 100)
    if len(x) < 100:
        return round(estimate, 2)
    weights = edges[y, x]
    x = x - edges.shape[1] / 2
    padding = width // 5 + 2

    def score(angle):
        rows = (y - x * math.tan(math.radians(angle)) + padding) * 4
        lower = rows.astype(np.int32)
        fraction = rows - lower
        projection = np.bincount(lower, weights * (1 - fraction),
                                 minlength=4*(height + 2*padding) + 1)
        projection += np.bincount(lower + 1, weights * fraction, minlength=len(projection))
        projection = cv2.GaussianBlur(projection[np.newaxis, :], (17, 1), 2)[0]
        return float(np.sum(projection * projection))

    coarse = max((estimate + step * 0.05 for step in range(-10, 11)), key=score)
    fine = max((coarse + step * 0.01 for step in range(-5, 6)), key=score)
    precise = max((fine + step * 0.001 for step in range(-10, 11)), key=score)
    return round(precise, 3)


def corrected(image, angle, flip, fit, margins=(0, 0, 0, 0)):
    if len(margins) != 4 or any(not 0 <= value <= 20 for value in margins):
        raise ValueError("端の白塗りは各辺0〜20%で指定してください")
    top, right, bottom, left = (
        round(value * dimension / 100)
        for value, dimension in zip(margins, (image.height, image.width, image.height, image.width))
    )
    width, height = image.size
    for box in ((0, 0, width, top), (width-right, 0, width, height),
                (0, height-bottom, width, height), (0, 0, left, height)):
        image.paste("white", box)
    if flip:
        image = image.transpose(Image.Transpose.ROTATE_180)
    if not angle:
        return image
    rotated = image.rotate(angle, Image.Resampling.BICUBIC, expand=fit, fillcolor="white")
    if not fit:
        return rotated
    rotated.thumbnail(image.size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", image.size, "white")
    canvas.paste(rotated, ((image.width-rotated.width)//2, (image.height-rotated.height)//2))
    return canvas


@contextmanager
def open_pdf(source):
    # Python handles paths containing non-ASCII characters on both platforms.
    with source.open("rb") as stream:
        with PDFIUM_LOCK:
            try:
                document = pdfium.PdfDocument(stream)
            except pdfium.PdfiumError as exc:
                if exc.err_code == pdfium_raw.FPDF_ERR_PASSWORD:
                    raise ValueError("パスワードで保護されたPDFです。保護を解除したPDFを開いてください。") from exc
                raise ValueError("PDFを読み取れませんでした。元のPDFが開けるか確認してください。") from exc
        try:
            with PDFIUM_LOCK:
                document.init_forms()
                count = len(document)
            if count == 0:
                raise ValueError("PDFにページがありません")
            yield document, count
        finally:
            with PDFIUM_LOCK:
                document.close()


def render_pdf_page(document, index, preview):
    try:
        with PDFIUM_LOCK, closing(document[index]) as page:
            # PDFium reports dimensions with the visible crop and page rotation applied.
            width, height = page.get_size()
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise ValueError("PDFのページ寸法を取得できませんでした")
            scale = 3200 / max(width, height) if preview else 300 / 72
            pixels = math.ceil(width * scale) * math.ceil(height * scale)
            if Image.MAX_IMAGE_PIXELS is not None and pixels > 2 * Image.MAX_IMAGE_PIXELS:
                raise ValueError("PDFのページが大きすぎます。縮小したPDFを開いてください。")
            with closing(page.render(scale=scale, draw_annots=True)) as bitmap:
                # Detach the image from the bitmap before releasing its buffer.
                image = bitmap.to_pil().convert("RGB")
        return image, (width, height)
    except pdfium.PdfiumError as exc:
        raise ValueError(f"PDFの{index + 1}ページ目を読み込めませんでした") from exc


def read_pages(source, document, count, preview=False):
    if document is None:
        yield rgb(source), None
    else:
        for index in range(count):
            yield render_pdf_page(document, index, preview)


def analyse_page(image):
    angle = detect(image)
    image.thumbnail((2000, 2000))
    preview = io.BytesIO()
    image.save(preview, "JPEG", quality=85)
    return {"angle": angle, "preview": base64.b64encode(preview.getvalue()).decode()}


def export_settings(data, count):
    settings = json.loads(data)
    if not isinstance(settings, list) or len(settings) != count:
        raise ValueError("ページ設定の数が一致しません")
    for page in settings:
        if not isinstance(page, list) or len(page) not in (3, 4):
            raise ValueError("ページ設定の形式が無効です")
        angle, flip, fit = page[:3]
        margins = page[3] if len(page) == 4 else [0, 0, 0, 0]
        if type(angle) not in (int, float) or not -15 <= angle <= 15:
            raise ValueError("角度は±15°以内で指定してください")
        if type(flip) is not bool or type(fit) is not bool:
            raise ValueError("回転・縮小の設定が無効です")
        if not isinstance(margins, list) or len(margins) != 4 or any(
            type(value) not in (int, float) or not 0 <= value <= 20 for value in margins
        ):
            raise ValueError("端の白塗りは各辺0〜20%で指定してください")
        yield angle, flip, fit, margins


def write_pdf(output, count, pages):
    offsets = [0]
    output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    def obj(payload):
        offsets.append(output.tell())
        output.write(f"{len(offsets)-1} 0 obj\n".encode() + payload + b"\nendobj\n")

    def stream(header, data):
        obj(header + f" /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream")

    obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3+3*i} 0 R" for i in range(count))
    obj(f"<< /Type /Pages /Count {count} /Kids [{kids}] >>".encode())
    for index, (image, (width, height)) in enumerate(pages):
        w, h = f"{width:.8f}", f"{height:.8f}"
        base = 3 + 3*index
        obj(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Resources << /XObject << /Scan {base+1} 0 R >> >> /Contents {base+2} 0 R >>".encode())
        jpeg = io.BytesIO()
        image.save(jpeg, "JPEG", quality=95, subsampling=0)
        stream(f"<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode".encode(), jpeg.getvalue())
        stream(b"<<", f"q {w} 0 0 {h} 0 0 cm /Scan Do Q".encode())
    start = output.tell()
    output.write(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())


def rgb(path):
    try:
        with Image.open(path) as image:
            if image.n_frames != 1:
                raise ValueError("複数フレームの画像には対応していません")
            rgba = image.convert("RGBA")
            result = Image.new("RGB", image.size, "white")
            result.paste(rgba, mask=rgba.getchannel("A"))
            return result
    except Image.DecompressionBombError as exc:
        raise ValueError("画像の解像度が大きすぎます。縮小した画像を開いてください。") from exc
    except UnidentifiedImageError as exc:
        raise ValueError("画像を読み取れませんでした。元の画像が開けるか確認してください。") from exc


class Server(BaseHTTPRequestHandler):
    def reply(self, data, kind, status=200):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply(Path(__file__).with_name("index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.headers.get("Host") != "127.0.0.1:8765" or self.headers.get("Origin") != "http://127.0.0.1:8765":
            self.send_error(403)
            return
        try:
            self.process()
        except ConnectionError:
            return
        except (ValueError, OSError, pdfium.PdfiumError) as exc:
            try:
                self.reply(json.dumps({"error": str(exc)}).encode(), "application/json", 400)
            except ConnectionError:
                return

    def process(self):
        size = int(self.headers.get("Content-Length", "0"))
        settings_length = int(self.headers.get("X-Settings-Length", "0"))
        if not 0 <= settings_length <= min(8 * 1024**2, size):
            raise ValueError("ページ設定のサイズが無効です")
        size -= settings_length
        if not 0 < size <= 512 * 1024**2:
            raise ValueError("入力サイズが上限の512 MBを超えています")
        settings_data = self.headers.get("X-Settings", "[]")
        if settings_length:
            settings_data = self.rfile.read(settings_length)
            if len(settings_data) != settings_length:
                raise ValueError("ページ設定の受信が途中で終了しました")
        with tempfile.TemporaryDirectory(prefix="score-") as temp:
            folder = Path(temp)
            source = folder / "input"
            with source.open("wb") as handle:
                remaining = size
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024**2))
                    if not chunk:
                        raise ValueError("受信が途中で終了しました")
                    handle.write(chunk)
                    remaining -= len(chunk)
            fmt = self.headers.get("X-Format", "").lower()
            pdf = fmt == ".pdf"
            if not pdf and fmt not in FORMATS:
                raise ValueError("未対応の形式です")
            with (open_pdf(source) if pdf else nullcontext((None, 1))) as (document, count):
                if self.path == "/analyse":
                    streaming = self.headers.get("Accept") == "application/x-ndjson"
                    pages = read_pages(source, document, count, preview=streaming)
                    if streaming:
                        self.stream_analysis(pages, count)
                    else:
                        previews = [analyse_page(image) for image, _ in pages]
                        self.reply(json.dumps(previews).encode(), "application/json")
                elif self.path == "/export":
                    pages = read_pages(source, document, count)
                    self.export(pages, count, settings_data, fmt, folder)
                else:
                    self.send_error(404)

    def export(self, pages, count, settings_data, fmt, folder):
        settings = list(export_settings(settings_data, count))
        images = (
            (corrected(image, *setting), size)
            for (image, size), setting in zip(pages, settings)
        )
        if fmt == ".pdf":
            with tempfile.TemporaryFile(dir=folder) as output:
                write_pdf(output, count, images)
                length = output.tell()
                output.seek(0)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                shutil.copyfileobj(output, self.wfile, length=256 * 1024)
        else:
            output = io.BytesIO()
            image, _ = next(images)
            options = {"quality": 95} if fmt in (".jpg", ".jpeg", ".webp") else {}
            image.save(output, FORMATS[fmt], **options)
            self.reply(output.getvalue(), "application/octet-stream")

    def stream_analysis(self, pages, count):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def send(record):
            self.wfile.write((json.dumps(record) + "\n").encode())
            self.wfile.flush()

        try:
            send({"total": count})
            for image, _ in pages:
                send(analyse_page(image))
        except ConnectionError:
            return
        except (ValueError, OSError) as exc:
            send({"error": str(exc)})


if __name__ == "__main__":
    print("http://127.0.0.1:8765", flush=True)
    with ThreadingHTTPServer(("127.0.0.1", 8765), Server) as server:
        server.serve_forever()
