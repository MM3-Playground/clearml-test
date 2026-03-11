from PIL import Image
import numpy as np

import onnxruntime as ort

import json
import io
import boto3
import os
import uuid

from botocore.config import Config

### NumPy DCT-II ###

def dct_1d_np(x: np.ndarray, norm: str) -> np.ndarray:
    """
    DCT-II over the last axis using an FFT trick.
    x: (..., N) float32/float64
    """
    x = np.asarray(x)
    x_shape = x.shape
    N = x_shape[-1]
    x2 = x.reshape(-1, N)

    v = np.concatenate([x2[:, ::2], np.flip(x2[:, 1::2], axis=1)], axis=1)
    Vc = np.fft.fft(v, axis=1)

    k = -np.arange(N, dtype=np.float32)[None, :] * np.pi / (2.0 * N)
    W_r = np.cos(k)
    W_i = np.sin(k)

    V = Vc[:, :N].real * W_r - Vc[:, :N].imag * W_i

    if norm == "ortho":
        V[:, 0] /= (np.sqrt(N) * 2.0)
        V[:, 1:] /= (np.sqrt(N / 2.0) * 2.0)

    V = 2.0 * V.reshape(*x_shape)
    return V

def dct_2d_np(x: np.ndarray, norm: str) -> np.ndarray:
    """
    DCT-II over the last two axes.
    x: (..., H, W)
    """
    X1 = dct_1d_np(x, norm=norm)
    X2 = dct_1d_np(np.swapaxes(X1, -1, -2), norm=norm)
    return np.swapaxes(X2, -1, -2)

def preprocess_rgb_to_dct_nchw(img: np.ndarray, log=True, factor=1.0) -> np.ndarray:
    """
    - Normalize mean=0,std=1 -> img / 255
    - HWC RGB -> CHW
    - Per-channel DCT2 (ortho)
    - log(abs(dct)+1e-12) if log
    Returns NCHW float32
    """
    img = img.astype(np.float32) / 255.0
    chw = np.transpose(img, (2, 0, 1)) # CHW

    # DCT per channel (ortho)
    dcts = []
    for c in range(chw.shape[0]):
        d = dct_2d_np(chw[c], norm="ortho")
        dcts.append(d)
    dd = np.stack(dcts, axis=0)  # CHW

    if log:
        dd = factor * np.log(np.abs(dd) + 1e-12)

    dd = dd.astype(np.float32)
    return np.expand_dims(dd, axis=0)  # NCHW

##############

ALLOWED_EXTENSIONS = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff"
}

AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")
UPLOAD_BUCKET = os.environ["UPLOAD_BUCKET"]

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual"}
    ),
)

# Load once per container
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

def json_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "content-type": "application/json",
            "access-control-allow-origin": "*"
        },
        "body": json.dumps(body),
    }

def run_inference_from_bytes(raw: bytes) -> dict:
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((512, 512), resample=Image.BILINEAR)
    img = np.array(img)

    img = preprocess_rgb_to_dct_nchw(img, log=True, factor=1.0)

    output = sess.run(None, {input_name: img})[0]

    label = "fake" if output > 0.5 else "real"

    return {
        "label": label,
    }

def handler(event, context):
    try:
        if "body" in event and event.get("body"):
            body = json.loads(event["body"])
        else:
            body = event

        action = body.get("action")

        if action == "presign":
            ext = body.get("ext", "jpg").lower()

            if ext not in ALLOWED_EXTENSIONS:
                return json_response(400, {"error": "Unsupported image type"})

            content_type = ALLOWED_EXTENSIONS[ext]
            key = f"uploads/{uuid.uuid4()}.{ext}"

            upload_url = s3.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": UPLOAD_BUCKET,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=300,
            )

            return json_response(200, {
                "upload_url": upload_url,
                "key": key,
                "content_type": content_type
            })

        if action == "infer":
            key = body["key"]

            obj = s3.get_object(Bucket=UPLOAD_BUCKET, Key=key)
            raw = obj["Body"].read()

            result = run_inference_from_bytes(raw)

            s3.delete_object(Bucket=UPLOAD_BUCKET, Key=key)

            return json_response(200, result)

        return json_response(400, {"error": "Unknown action"})

    except Exception as e:
        return json_response(500, {"error": str(e)})