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
# import psutil
import gc

import time
import logging
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read
from paddlehub.module.module import moduleinfo, runnable, serving
import cv2
import paddlehub as hub

from tools.infer.utility import base64_to_cv2
from tools.infer.predict_system import TextSystem
from tools.infer.utility import parse_args
from deploy.hubserving.ocr_system.params import read_params
logger = get_logger()

# import objgraph

@moduleinfo(
    name="ocr_rec",
    version="1.0.0",
    summary="ocr recognition service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class OCRRec(hub.Module):
    def _initialize(self, use_gpu=False, enable_mkldnn=True):
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

    def read_image(self, path):
        assert os.path.isfile(
                path), "The {} isn't a valid file.".format(path)
        img = cv2.imread(path)
        if img is None:
            logger.info("error in loading image:{}".format(path))
            return None
            
        return img

    def del_images(self, paths=[]):
        for image in paths:
            try:
                os.remove(image)
            except OSError as e:
                logger.info("删除文件 {} 时出错".format(image))
    
    def predict(self, image, locates=[]):
        """
        Get the chinese texts in the predicted images.
        Args:
            image (numpy.ndarray): images data, shape of each is [H, W, C]. If images not paths
        Returns:
            res (list): The result of chinese texts and save path of images.
        """

        assert image is not None, "There is not any image to be predicted. Please check the input data."

        all_results = []
        if len(locates) > 0:
            starttime = time.time()

            for locate in locates:
                length = len(locate)
                rec_res_final = []
                # locate 定位坐标(高起始,高结束,宽起始,宽结束) & 小于图片分辨率
                # tt = length == 4 and locate[0] < locate[1] and locate[2] < locate[3] and locate[1] <= img.shape[0] and locate[3] <= img.shape[1]
                # all_results.append({'locate': locate, 'shape': img.shape, 'tt': tt})
                if length == 4 and locate[0] < locate[1] and locate[2] < locate[3] and locate[1] <= image.shape[0] and locate[3] <= image.shape[1]:
                    small_img = image[locate[0]:locate[1],locate[2]:locate[3]]
                else:
                    rec_res_final.append("")
                    continue
                dt_boxes, rec_res, _ = self.text_sys(small_img)

                dt_num = len(dt_boxes)

                for dno in range(dt_num):
                    text, _ = rec_res[dno]
                    rec_res_final.append(text)

                all_results.append(rec_res_final)
                del small_img

            elapse = time.time() - starttime    
            logger.info("Predict time: {}".format(elapse))
        else:
            dt_boxes, rec_res, rec_time = self.text_sys(image)
            logger.info("Predict time: {}".format(rec_time['all']))

            dt_num = len(dt_boxes)

            for dno in range(dt_num):
                text, _ = rec_res[dno]
                all_results.append(text)

        # 手动释放内存
        del image, locates
        return all_results

    def predict_pdf(self, path):
        """
        Get the chinese texts in the predicted images.
        Args:
            path (str): The path of pdf.
        Returns:
            res (list): The result of chinese texts and save path of pdf.
        """

        all_results = []
        pdf, flag_gif, flag_pdf = check_and_read(path)
        if pdf is None or not flag_pdf:
            predicted_data = []
        else:
            page_num = len(pdf)
            predicted_data = pdf[:page_num]

        assert predicted_data != [], "There is not any file to be predicted. Please check the input data."
        starttime = time.time()

        for index, img in enumerate(predicted_data):
            rec_res_final = []
            if img is None:
                logger.info("error in loading image")
                rec_res_final.append([])
                continue
            
            dt_boxes, rec_res, _ = self.text_sys(img)
            dt_num = len(dt_boxes)
            
            for dno in range(dt_num):
                text, _ = rec_res[dno]
                rec_res_final.append(text)
        
            all_results.append(rec_res_final)

        elapse = time.time() - starttime
        logger.info("Predict time: {}".format(elapse))

        # 手动释放内存
        del predicted_data, pdf
        return all_results


    # 显示当前 python 程序占用的内存大小
    # def show_memory_info(self, hint):
    #     pid = os.getpid()
    #     p = psutil.Process(pid)

    #     info = p.memory_full_info()
    #     memory = info.uss / 1024. / 1024
    #     logger.debug('{} memory used: {} MB'.format(hint, memory))

    @serving
    def serving_method(self, image, type, locates=[], **kwargs):
        """
        Run as a service.
        """

        if(type == "image"):
            images_decode = base64_to_cv2(image)
            results = self.predict(image=images_decode, locates=locates)
        elif(type == "path"):
            images_decode = self.read_image(image)
            results = self.predict(image=images_decode, locates=locates)
        elif(type == "delfile"):
            images_decode = self.read_image(image)
            results = self.predict(image=images_decode, locates=locates)
            self.del_images([image])
        elif(type == "pdf"):
            results = self.predict_pdf(path=image)
        else:
            raise TypeError("The input type is inconsistent with expectations.")
       
        return results
    
if __name__ == '__main__':
    ocr = OCRRec()
    ocr._initialize()
    image_path = [
        './doc/imgs/11.jpg',
        './doc/imgs/12.jpg',
    ]
    res = ocr.predict(paths=image_path)
    print(res)
