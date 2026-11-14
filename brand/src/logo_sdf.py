"""Bake the OQVAR mark into a signed-distance texture for the WebGL logo on the landing page.

usage: python logo_sdf.py <body.png> <eye.png> <out.png>
Inputs are 2048² renders of the mark (black shape on white) on a square canvas with 15% padding.
Output is 1024² RGB: R = body, G = eye. value = 0.5 + d / (2 * BAND), d in output texels, negative inside.
"""
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

N, BAND = 1024, 48  # BAND / N * 2 must equal the BAND constant in the shader in web/landing.js (0.09375)


def sdf(path: str) -> np.ndarray:
    inside = np.asarray(Image.open(path).convert("L")) < 128
    scale = inside.shape[0] / N
    d = (distance_transform_edt(~inside) - distance_transform_edt(inside)) / scale  # px -> output texels
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((N, N), Image.BILINEAR))
    return np.clip(0.5 + d / (2 * BAND), 0, 1)


body, eye = sdf(sys.argv[1]), sdf(sys.argv[2])
rgb = np.stack([body, eye, np.zeros_like(body)], axis=-1)
Image.fromarray(np.round(rgb * 255).astype(np.uint8), "RGB").save(sys.argv[3], optimize=True)
print("saved", sys.argv[3], rgb.shape, "inside texels:", int((body < .5).sum()), int((eye < .5).sum()))
