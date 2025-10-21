import logging
import os
import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image, ImageOps
from scipy.stats import norm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)#(000)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs
class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(112*0.5))#32
        return img

class TargetedStrongAugment(object):
    def __init__(self, n, window_size):
        assert n >= 1
        self.n = n
        self.window_size = window_size
    def local_pixel_shuffling(self, image):
        # 获取输入图像的高度和宽度
        h, w = image.size
        random_window_width = random.randint(self.window_size//2, self.window_size)#0
        # 生成窗口高度的随机范围 [20, window_size]
        random_window_height = random.randint(self.window_size//2, self.window_size)
        # 随机生成窗口的起始位置
        start_h = random.randint(0, h - random_window_height)
        start_w = random.randint(0, w - random_window_width)
        # 从原图像中裁剪出窗口
        window = image.crop((start_w, start_h, start_w + random_window_width, start_h + random_window_height))
        # 获取窗口内的像素值
        pixels = list(window.getdata())
        # 随机打乱窗口内的像素值
        random.shuffle(pixels)
        # 将打乱后的像素值重新展开为一维列表
        flat_pixels = [val for sublist in pixels for val in sublist]
        # 截取前 random_window_width * random_window_height 个像素值，确保不超过窗口大小
        flat_pixels = flat_pixels[:random_window_width * random_window_height]
        # 创建一个新的黑色图像，用于存放打乱后的像素值
        #new_image = Image.new(image.mode, (random_window_width, random_window_height), None)  # 0表示全黑
        new_image = Image.new(image.mode, (random_window_width, random_window_height))
        # 将打乱后的像素值放入新图像中
        new_image.putdata(flat_pixels)
        # 将新图像粘贴回原图像的相应位置
        image.paste(new_image, (start_w, start_h))
        # 返回添加噪声后的图像
        return image

    def in_painting(self,image):
        # 获取输入图像的高度和宽度
        h, w = image.size
        # 随机生成多个噪声窗口
        # 生成窗口宽度的随机范围 [20, window_size]
        random_window_width = random.randint(self.window_size // 2, self.window_size)
        # 生成窗口高度的随机范围 [20, window_size]
        random_window_height = random.randint(self.window_size // 2, self.window_size)
        # 随机生成窗口的起始位置
        start_h = random.randint(0, h - random_window_height)
        start_w = random.randint(0, w - random_window_width)
        # 从原图像中裁剪出窗口
        window = image.crop((start_w, start_h, start_w + random_window_width, start_h + random_window_height))
        # 生成与窗口大小相同的随机噪声
        noise = Image.fromarray(
            np.random.normal(0.05, 0.15, random_window_width * random_window_height).reshape(random_window_height,
                                                                                             random_window_width).astype(
                np.uint8))
        # 创建一alpha 蒙版，并将噪声粘贴到蒙版上
        alpha_mask = Image.new(image.mode, (random_window_width, random_window_height), None)
        alpha_mask.paste(noise, (0, 0))
        # 将生成的噪声窗口覆盖到原图像的相应位置
        image.paste(Image.new(image.mode, (random_window_width, random_window_height)), (start_w, start_h), mask=alpha_mask)
        # 返回添加噪声后的图像
        return image
    def CutoutAbs(self, img):
        # 获取输入图像的宽度和高度
        h, w = img.size
        # 随机生成窗口的起始位置
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
        # 使用 self.window_size 作为正方形窗口的边长 v
        v = self.window_size
        # 计算正方形窗口的左上角坐标
        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        # 计算正方形窗口的右下角坐标
        x1 = int(min(w, x0 + v))
        y1 = int(min(h, y0 + v))
        # 定义窗口的坐标元组
        xy = (x0, y0, x1, y1)
        # 定义填充颜色为灰色
        color = (0, 0, 0)  # 或者其他颜色 (0, 0, 0)
        # 创建图像的副本，并在副本上绘制矩形
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)

        # 返回处理后的图像
        return img

    def simulate_ultrasound_dropout(self,image):
        # 获得图像的宽度和高度
        w, h = image.size
        # 随机生成窗口的宽度和高度范围
        random_window_width = random.randint(self.window_size // 2, self.window_size)
        random_window_height = random.randint(self.window_size // 2, self.window_size)
        # 随机生成窗口的起始位置
        start_h = random.randint(0, h - random_window_height)
        start_w = random.randint(0, w - random_window_width)
        # 定义小窗口的边界
        end_h = start_h + random_window_height
        end_w = start_w + random_window_width
        # 计算小窗口内的像素数量
        window_pixels = random_window_width * random_window_height
        # 随机确定最大丢失像素数，最多不超过小窗口内的像素数量的50%
        max_lost_pixels = window_pixels // 2
        # 随机确定丢失的像素数，最多不超过最大丢失像素数
        num_lost_pixels = random.randint(0, max_lost_pixels)
        # 随机确定丢失的像素位置，限制在小窗口内
        lost_pixels = [(random.randint(start_w, end_w - 1), random.randint(start_h, end_h - 1)) for _ in
                       range(num_lost_pixels)]
        # 替换丢失的像素为纯黑色
        for x, y in lost_pixels:
            image.putpixel((x, y), (0, 0, 0))
        # 返回模拟丢失后的图像
        return image
    def adjust_brightness(self, image):
        brightness_transform = transforms.ColorJitter(brightness=0.5)
        image=brightness_transform(image)
        return image

    def Blur(self, image):
        GaussianBlur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
        image = GaussianBlur(image)
        return image


    def __call__(self, img):
        # 判断是否应用 CustomAugment 方法
        if random.random() < 0.5:  # 50% 的概率应用
            #augment_operations = [self.generate_random_spots, self.local_pixel_shuffling, self.adjust_brightness, self.Blur]
            #augment_operations = [self.in_painting, self.local_pixel_shuffling, self.adjust_brightness, self.Blur]
            augment_operations = [self.in_painting, self.local_pixel_shuffling, self.CutoutAbs,self.simulate_ultrasound_dropout,self.adjust_brightness, self.Blur]
            num_operations = random.randint(1, self.n)  # 随机选择 1 到 self.n 个方法
            ops = random.sample(augment_operations, k=num_operations)
            for op in ops:
                img = op(img)
        return img
