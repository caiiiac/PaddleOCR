# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0, ".")
import copy
import gc

import time
import logging
from ppocr.utils.logging import get_logger
from paddlehub.module.module import moduleinfo, runnable, serving
import cv2
import numpy as np
import paddlehub as hub

from tools.infer.utility import base64_to_cv2
from tools.infer.predict_system import TextSystem
from tools.infer.utility import parse_args
from deploy.hubserving.ocr_system.params import read_params
logger = get_logger()

@moduleinfo(
    name="ocr_system",
    version="1.0.0",
    summary="ocr system service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/PP-OCR_system")
class OCRSystem(hub.Module):
    def _initialize(self, use_gpu=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        cfg = self.merge_configs()
        if not cfg.show_log:
            logger.setLevel(logging.INFO)

        cfg.use_gpu = use_gpu
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 8000
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        cfg.ir_optim = True
        cfg.enable_mkldnn = enable_mkldnn

        self.text_sys = TextSystem(cfg)

    def merge_configs(self, ):
        # deafult cfg
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        cfg = parse_args()

        update_cfg_map = vars(read_params())

        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        sys.argv = copy.deepcopy(backup_argv)
        return cfg

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                logger.info("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images

    def predict(self, images=[], paths=[], ids=[], locate=[]):
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The result of chinese texts and save path of images.
        """

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        all_results = []
        for index, img in enumerate(predicted_data):
            if img is None:
                logger.info("error in loading image")
                all_results.append([])
                continue
            starttime = time.time()
            length = len(locate)
            # locate 定位坐标(高起始,高结束,宽起始,宽结束) & 小于图片分辨率
            # tt = length == 4 and locate[0] < locate[1] and locate[2] < locate[3] and locate[1] <= img.shape[0] and locate[3] <= img.shape[1]
            # all_results.append({'locate': locate, 'shape': img.shape, 'tt': tt})
            if length == 4 and locate[0] < locate[1] and locate[2] < locate[3] and locate[1] <= img.shape[0] and locate[3] <= img.shape[1]:
                img = img[locate[0]:locate[1],locate[2]:locate[3]]
            # elif length > 0 and length < 4:
            #     raise TypeError("The locate data is inconsistent with expectations.")
            dt_boxes, rec_res, _ = self.text_sys(img)
            elapse = time.time() - starttime
            logger.info("Predict time: {}".format(elapse))

            dt_num = len(dt_boxes)
            rec_res_final = []

            for dno in range(dt_num):
                text, _ = rec_res[dno]
                # rec_res_final.append({
                #     'text': text,
                #     'confidence': float(score),
                #     'text_region': dt_boxes[dno].astype(np.int_).tolist()
                # })
                rec_res_final.append(text)
            all_results.append({
                'id': ids[index],
                'text':rec_res_final
            })

        # 手动释放内存
        del predicted_data
        gc.collect()
        return all_results

    @serving
    def serving_method(self, images, type, ids, locate=[], **kwargs):
        """
        Run as a service.
        """
        results = ""
        if len(images) != len(ids):
            raise TypeError("The input ids is inconsistent with expectations.")
        
        if(type == "image"):
            images_decode = [base64_to_cv2(image) for image in images]
            results = self.predict(images=images_decode, ids=ids, locate=locate, **kwargs)
        elif(type == "path"):
            results = self.predict(paths=images, ids=ids, locate=locate)
        else:
            raise TypeError("The input type is inconsistent with expectations.")
        return results
    


if __name__ == '__main__':
    ocr = OCRSystem()
    ocr._initialize()
    image_path = [
        './doc/imgs/11.jpg',
        './doc/imgs/12.jpg',
    ]
    res = ocr.predict(paths=image_path)
    print(res)
