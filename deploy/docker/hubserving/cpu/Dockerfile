# Version: 2.4.2
FROM registry.baidubce.com/paddlepaddle/paddle:2.4.2

# PaddleOCR base on Python3.7
RUN pip3.7 install --upgrade pip -i https://mirror.baidu.com/pypi/simple

# RUN pip3.7 install paddlehub==2.3.1 --upgrade -i https://mirror.baidu.com/pypi/simple

RUN git clone https://gitee.com/caiiiac/PaddleOCR.git /PaddleOCR

WORKDIR /PaddleOCR

RUN pip3.7 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

RUN pip3.7 install ./deploy/docker/hubserving/wheel/paddlehub-2.3.2-py3-none-any.whl

RUN mkdir -p /PaddleOCR/inference/
# Download orc detect model(light version). if you want to change normal version, you can change ch_ppocr_mobile_v2.0_det_infer to ch_ppocr_server_v2.0_det_infer, also remember change det_model_dir in deploy/hubserving/ocr_system/params.py>）
ADD https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_PP-OCRv3_det_infer.tar -C /PaddleOCR/inference/

# Download direction classifier(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_cls_infer to ch_ppocr_mobile_v2.0_cls_infer, also remember change cls_model_dir in deploy/hubserving/ocr_system/params.py）
ADD https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer.tar -C /PaddleOCR/inference/

# Download orc recognition model(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_rec_infer to ch_ppocr_server_v2.0_rec_infer, also remember change rec_model_dir in deploy/hubserving/ocr_system/params.py）
ADD https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_PP-OCRv3_rec_infer.tar -C /PaddleOCR/inference/

EXPOSE 8866

CMD ["/bin/bash","-c","hub install deploy/hubserving/ocr_system/ && hub install deploy/hubserving/ocr_rec/ && hub serving start -c deploy/hubserving/ocr_system/config.json"]