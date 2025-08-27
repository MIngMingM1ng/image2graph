import time
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
import cpi

##63.9MB 处理较慢  小模型10.2MB
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
upsampler = RealESRGANer(
        scale=4,
        #model_path='./RealESRGAN_x4plus_anime_6B.pth',
        model_path='./realesr-general-x4v3.pth',
        model=model,
        tile=320,  # 处理大图像时分块
        tile_pad=2,
        pre_pad=0,
        gpu_id=0,

)
start_at = time.perf_counter()
# 读取图像
#img = cv2.imread('./ocrtest/test5.png', cv2.IMREAD_UNCHANGED)
img=cv2.imread('./test.png', cv2.IMREAD_UNCHANGED)
h, w = img.shape[:2]

output, _ = upsampler.enhance(img,outscale=4)
# 保存输出
cv2.imwrite('./test_sr.png', output)
end_time = time.perf_counter() - start_at
print(f"Time taken: {end_time:.3f} seconds")