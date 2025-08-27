from paddleocr import PaddleOCR
import  PPOCRLabel

#在paddleOCR中修改默认配置
ocr = PaddleOCR(paddlex_config="PaddleOCR.yaml",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,)

# 默认使用 PP-OCRv5_server_det 模型作为默认文本检测模型，如果微调的不是该模型，通过 text_detection_model_name 修改模型名称

result = ocr.predict("./test.png")
for res in result:
    res.print()
    res.save_to_img("./test_ocr")
    res.save_to_json("./SRoutput")