import time
import os
import cv2
from SR.SR_process import init_sr_model, enhance_image
from OCR.OCR_process import init_ocr_model, predict_ocr

# 配置路径
INPUT_DIR = "./process/images"  # 输入图片目录
SR_OUTPUT_DIR = "./process/SR"  # 超分结果输出目录
OCR_IMAGE_DIR = "./process/OCR/ocr_image"  # OCR图像输出目录
OCR_JSON_DIR = "./process/OCR/ocr_label"  # OCR JSON结果输出目录

# 创建输出目录（如果不存在）
os.makedirs(SR_OUTPUT_DIR, exist_ok=True)
os.makedirs(OCR_IMAGE_DIR, exist_ok=True)
os.makedirs(OCR_JSON_DIR, exist_ok=True)

# 支持的图片格式
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')


# 处理单张图片
def process_image(filename, upsampler, ocr):
    try:
        # 读取图像
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return

        # 超分处理
        start_time = time.perf_counter()
        sr_output = enhance_image(upsampler, img)

        # 保存超分结果
        sr_filename = os.path.splitext(filename)[0] + "_sr.png"
        sr_path = os.path.join(SR_OUTPUT_DIR, sr_filename)
        cv2.imwrite(sr_path, sr_output)

        # OCR识别
        result = predict_ocr(ocr, sr_path)

        # 保存OCR结果
        base_name = os.path.splitext(filename)[0]
        for i, res in enumerate(result):
            # 保存OCR图像（如果有多页/多结果，添加索引）
            if len(result) > 1:
                ocr_img_name = f"{base_name}_ocr_{i}.png"
                ocr_json_name = f"{base_name}_ocr_{i}.json"
            else:
                ocr_img_name = f"{base_name}_ocr.png"
                ocr_json_name = f"{base_name}_ocr.json"

            res.save_to_img(os.path.join(OCR_IMAGE_DIR, ocr_img_name))
            res.save_to_json(os.path.join(OCR_JSON_DIR, ocr_json_name))

        processing_time = time.perf_counter() - start_time
        print(f"处理完成: {filename}，耗时: {processing_time:.3f}秒")

    except Exception as e:
        print(f"处理 {filename} 时出错: {str(e)}")


# 批量处理目录中的所有图片
def batch_process():
    # 初始化模型
    print("初始化超分模型...")
    upsampler = init_sr_model()
    #在这里调用时，前面函数里的路径都要改成相对于main的路径

    print("初始化OCR模型...")
    ocr = init_ocr_model()

    # 获取目录中所有图片文件
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f))
           and f.lower().endswith(SUPPORTED_FORMATS)
    ]

    if not image_files:
        print(f"在目录 {INPUT_DIR} 中未找到任何图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件，开始处理...")

    # 逐个处理图片
    total_start_time = time.perf_counter()
    for i, filename in enumerate(image_files, 1):
        print(f"处理第 {i}/{len(image_files)} 个文件: {filename}")
        process_image(filename, upsampler, ocr)

    total_time = time.perf_counter() - total_start_time
    print(f"所有图片处理完成，总耗时: {total_time:.3f}秒")


if __name__ == "__main__":
    batch_process()
    print('end')
    print()
