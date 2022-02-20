from PIL import Image, ImageOps
import numpy as np
import wscreenshot
import time
import cv2
import os


def record():
    dest_dir = 'screenshots5'
    win_text = '升级角色版'

    ws = None
    while True:
        try:
            ws = wscreenshot.Screenshot(win_text)
            break
        except Exception:
            time.sleep(0.5)
            continue

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    while True:
        img = ws.screenshot()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dest_dir, str(int(time.time() * 10)) + '.bmp'), img)
        time.sleep(0.5)



def get_template1():
    img = np.array(Image.open('screenshots2/16442133829.bmp'))
    height = 16
    width = 22
    x0 = 343
    y0 = 796
    stride = 24

    for i in range(39):
        x = x0 + int(i * stride)
        y = y0
        rect = img[y:y+height, x: x+width]
        char_img = Image.fromarray(rect)
        char_img.save(str(i) + '.bmp')


def get_template1a():
    img = np.array(Image.open('screenshots6/1645289775.2096496.bmp'))
    height = 8
    width = 22
    x0 = 199
    y0 = 792
    stride = 24

    for i in range(45):
        x = x0 + int(i * stride)
        y = y0
        rect = img[y:y+height, x: x+width]
        char_img = Image.fromarray(rect)
        char_img.save(str(i) + '.bmp')


def get_template2():
    img = np.array(Image.open('screenshots2/16442133829.bmp'))
    height = 18
    width = 22
    x0 = 343
    y0 = 813
    stride = 24

    for i in range(39):
        x = x0 + int(i * stride)
        y = y0
        rect = img[y:y+height, x: x+width]
        char_img = Image.fromarray(rect)
        char_img.save(str(i) + '.bmp')


def get_template3():
    img = np.array(Image.open('digits.bmp'))
    height = 11
    width = 6
    x0 = 3
    y0 = 2
    stride = 12

    for i in range(16):
        x = x0 + int(i * stride)
        y = y0
        rect = img[y:y+height, x: x+width]
        char_img = Image.fromarray(rect)
        char_img.save('s' + str(i) + '.bmp')


def get_template4():
    img = np.array(Image.open('screenshots3/16442123931.bmp'))
    height = 17
    width = 13
    x0 = 108
    y0 = 944
    stride = 21

    for i in range(2):
        x = x0 + int(i * stride)
        y = y0
        rect = img[y:y+height, x: x+width]
        char_img = Image.fromarray(rect)
        char_img.save(str(i) + '.bmp')


get_template1a()

