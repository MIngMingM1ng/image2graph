from paddleocr import PaddleOCR


# 初始化OCR模型
def init_ocr_model():
    return PaddleOCR(
        paddlex_config="./OCR/PaddleOCR.yaml",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


# OCR识别
def predict_ocr(ocr, img_path):
    result = ocr.predict(img_path)
    return result
