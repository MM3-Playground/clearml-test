from PIL import Image
import numpy as np

import onnxruntime as ort

### NumPy DCT-II ###

def dct_1d_np(x: np.ndarray, norm: str) -> np.ndarray:
    """
    DCT-II over the last axis using an FFT trick. Matches your torch logic.
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
    Matches your Albumentations+ToTensorV2+DCT pipeline:
    - Resize to 512x512
    - Normalize mean=0,std=1 -> img / 255
    - HWC RGB -> CHW
    - Per-channel DCT2 (ortho)
    - log(abs(dct)+1e-12) if log
    Returns NCHW float32
    """
    img = img.astype(np.float32) / 255.0  # matches A.Normalize(mean=0,std=1)
    chw = np.transpose(img, (2, 0, 1))    # CHW

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

def main(image_path, model_path = "model.onnx"):
    # CPU provider
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    # load image
    img = Image.open(image_path).convert("RGB").resize((512, 512), resample=Image.BILINEAR)
    img = np.array(img)

    img = preprocess_rgb_to_dct_nchw(img, log=True, factor=1.0)

    # ort
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: img})[0]

    label = "fake" if output > 0.5 else "real"

    print({"label": label})

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
