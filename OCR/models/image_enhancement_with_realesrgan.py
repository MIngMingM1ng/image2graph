import cv2
import numpy as np
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer


# 初始化SR模型
def init_sr_model():
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    upsampler = RealESRGANer(
        scale=4,
        model_path='./SR/realesr-general-x4v3.pth',
        model=model,
        tile=320,  # 处理大图像时分块
        tile_pad=2,
        pre_pad=0,
        gpu_id=0,
    )
    return upsampler


# 超分处理
def enhance_image(upsampler, img):
    sr_output, _ = upsampler.enhance(img, outscale=2)
    return sr_output