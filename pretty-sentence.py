"""
File: pretty-sentence.py
Author: Chuncheng Zhang
Date: 2023-09-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-09-11 ------------------------
# Requirements and constants
import cv2
import numpy as np
import pandas as pd

from rich import print, inspect
from pathlib import Path
from IPython.display import display
from PIL import ImageFont, ImageDraw, Image


# %% ---- 2023-09-11 ------------------------
# Function and class
sentence = open('text1.txt', 'rb').read().decode().replace('\n', '')
n = len(sentence)

fg = (0, 0, 0, 0)

# fontpath = Path("C:\Windows\Fonts", "STXINWEI.TTF")  # 华文新魏
# fontpath = Path("C:\Windows\Fonts", "STLITI.TTF")  # 隶书
# fontpath = Path("C:\Windows\Fonts", "STXINGKA.TTF")  # 华文行楷
fontpath = Path('./yunfengjinglongxingshu.ttf')

font = ImageFont.truetype(fontpath.as_posix(), 40)


def compute_char(char, font=font, fg=(0, 0, 0, 0)):
    bbox = font.getbbox(char)
    x, y, w, h = bbox
    mat = np.zeros((max(h-y, 20), max(w-x, 20), 3), dtype=np.uint8) + 255
    img = Image.fromarray(mat)
    draw = ImageDraw.Draw(img)
    draw.text((-x, -y), char, font=font, fill=fg)

    canny_bgr = cv2.Canny(np.array(img), 1, 1)

    ratio = np.sum(canny_bgr / 255) / np.prod(canny_bgr.shape)

    canny = np.concatenate([np.array(img), cv2.cvtColor(
        canny_bgr, cv2.COLOR_GRAY2BGR)], axis=1)

    return img, canny, ratio


# %%
lst = []
for char in sentence:
    img, canny, ratio = compute_char(char)
    display(Image.fromarray(canny))
    # print(char, ratio)
    lst.append(dict(
        char=char,
        ratio=ratio
    ))

df = pd.DataFrame(lst)
df['scale'] = df['ratio'] / df['ratio'].max()
df


def draw_text(df, force_size=None):
    mat = np.zeros((40 * 40, 40 * 10, 3), dtype=np.uint8) + 255
    image = Image.fromarray(mat)
    draw_to_image = ImageDraw.Draw(image)

    x = 0
    y = 0
    for i in df.index:

        if force_size is None:
            size = int(40 * df.loc[i, 'scale'])
        else:
            size = force_size

        char = df.loc[i, 'char']
        font = ImageFont.truetype(fontpath.as_posix(), size=size)

        if x + size > 400:
            x = 0
            y += 40

        draw_to_image.text((x, y + (40 - size)/2), char, font=font, fill=fg)

        x += size

    image = image.crop((0, 0, 400, y + 40))

    return image


# display(img_pil)

img_pil_rescale = draw_text(df)
display(img_pil_rescale)

img_pil_raw = draw_text(df, force_size=int(40 * df.loc[0, 'scale']))
display(img_pil_raw)

display(df)
# %% ---- 2023-09-11 ------------------------
# Play ground

# %% ---- 2023-09-11 ------------------------
# Pending

# %% ---- 2023-09-11 ------------------------
# Pending

# %%
