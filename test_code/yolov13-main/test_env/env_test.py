# from ultralytics import YOLO

# model = YOLO('yolov13n.pt')  # Replace with the desired model scale
# print('end')

import os
from ultralytics import YOLO
from PIL import Image, ImageDraw

def predict_and_save_images(image_folder, bordered_output_folder, unbordered_output_folder, cropped_objects_folder, model_path):
    # 确保输出文件夹存在
    if not os.path.exists(bordered_output_folder):
        os.makedirs(bordered_output_folder)
    if not os.path.exists(unbordered_output_folder):
        os.makedirs(unbordered_output_folder)
    if not os.path.exists(cropped_objects_folder):
        os.makedirs(cropped_objects_folder)

    # 加载YOLO模型
    model = YOLO(model_path)

    # 定义支持的图片文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    # 遍历图片文件夹中的所有文件
    for filename in os.listdir(image_folder):
        # 获取文件扩展名并转换为小写
        file_ext = os.path.splitext(filename)[1].lower()
        
        # 检查文件是否为图片文件
        if file_ext in image_extensions:
            image_path = os.path.join(image_folder, filename)
            
            # 进行预测
            results = model(image_path)
            
            # 打开图片
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # 创建去边框的图片副本
            unbordered_img = img.copy()
            unbordered_draw = ImageDraw.Draw(unbordered_img)
            
            # 用于计数裁剪的对象
            object_count = 0
            
            # 绘制预测结果
            for result in results:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 在带边框的图片上绘制边界框
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    # 裁剪对象区域
                    cropped_object = img.crop((x1, y1, x2, y2))
                    
                    # 保存裁剪的对象图片
                    cropped_object_path = os.path.join(cropped_objects_folder, f"{os.path.splitext(filename)[0]}_object_{object_count}{file_ext}")
                    cropped_object.save(cropped_object_path)
                    print(f"Saved cropped object: {cropped_object_path}")
                    
                    # 增加对象计数
                    object_count += 1
                    
                    padding = 0
                    inner_x1 = max(x1 + padding, 0)
                    inner_y1 = max(y1 + padding, 0)
                    inner_x2 = min(x2 - padding, img.width - 1)
                    inner_y2 = min(y2 - padding, img.height - 1)
                    
                    # 确保 inner_x1 <= inner_x2 和 inner_y1 <= inner_y2
                    if inner_x1 < inner_x2 and inner_y1 < inner_y2:
                        # 在去边框的图片上用白色填充边界框内的区域（去掉边框外围一周）
                        unbordered_draw.rectangle([inner_x1, inner_y1, inner_x2, inner_y2], fill="white")

            # 保存带有预测结果的图片
            bordered_image_path = os.path.join(bordered_output_folder, filename)
            img.save(bordered_image_path)
            print(f"Saved bordered image: {bordered_image_path}")

            # 保存去掉边框的图片
            unbordered_image_path = os.path.join(unbordered_output_folder, filename)
            unbordered_img.save(unbordered_image_path)
            print(f"Saved unbordered image: {unbordered_image_path}")

# 使用示例
image_folder = './images'          # 替换为你的图片文件夹路径
bordered_output_folder = './predict'     # 替换为你想要保存带边框结果图片的目标文件夹路径
unbordered_output_folder = './预测结果_去边框'   # 替换为你想要保存去边框结果图片的目标文件夹路径
cropped_objects_folder = './预测结果_裁剪对象'   # 替换为你想要保存裁剪对象图片的目标文件夹路径
model_path = './weights/best.pt' # 替换为你的YOLO模型路径
predict_and_save_images(image_folder, bordered_output_folder, unbordered_output_folder, cropped_objects_folder, model_path)