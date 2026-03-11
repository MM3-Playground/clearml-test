import cv2
import numpy as np

import torch

import onnxruntime as ort

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

### FCRDCT ###

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)
    
    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

class DCT(ImageOnlyTransform):
    def __init__(self, convert = False, log = False, factor = 1, always_apply=False, p=1.0):
        super(DCT, self).__init__(always_apply, p)
        self.convert = convert
        self.log = log
        self.factor = factor

    def apply(self, img, **params):
        if (self.convert):
            img = img / 255.0

        dd = torch.stack([dct_2d(c, norm='ortho') for c in img.unbind()])

        if (self.log):
            dd = self.factor * torch.log(torch.abs(dd) + 1e-12)

        return dd

##############

def main(image_path, model_path = "model.onnx"):
    # CPU provider
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    # transforms
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=0.0, std=1.0), 
        ToTensorV2(),
        DCT(p = 1.0, log=True, factor=1)
    ])

    # load image
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    ori_size = img.shape

    # resize to fit model
    img = transform(image = img)['image'].unsqueeze(0)

    img = img.numpy()

    # ort
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: img})

    label = "fake" if output[0] > 0.5 else "real"

    print({"label": label})

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
