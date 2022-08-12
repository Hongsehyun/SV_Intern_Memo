# SV_Intern_Memo

​	

​	

#### File Description

![image](https://user-images.githubusercontent.com/84533279/184269629-dc3f0d01-3eed-430d-afac-bea7275dd7c5.png)



​	

**1. Study**

Summary of Study Materials

  


​	

​	

**2. FasterRCNN_Tutorial**

Trial and Error of Studying Faster R-CNN,

this FOLDER includes Tutorial implementation of Faster R-CNN based on Tensorflow & Pytorch.

​	

​	

**3. FasterRCNN_Quantization_Framework**

This FOLDER contains several frameworks related to **quantization** & **faster Rcnn** & **Pytorch** etc...

Reference Link : https://github.com/chenyuntc/simple-faster-rcnn-pytorch

Reference Link : https://github.com/openvinotoolkit/openvino_notebooks

Reference Link : https://github.com/open-mmlab/mmdetection

Reference Link : https://github.com/jwyang/faster-rcnn.pytorch

Reference Link : https://github.com/dankernel/pytorch-static-quantization

​	

​	

​	

#### **(+) Frequently used commands**

- **Faster-rcnn.pytorch** Framework's Commands 

**Visdom HTTP**
http://localhost:8097/

​	

**Visdom Define**
python -m visdom.server -p 8098

​	

**Server Path**
cd media/hdd/sehyun/faster-Rcnn-Demo  

​	

**Training Command [Python Command Line Arguments]**
python train.py train --env='fasterrcnn' --plot-every=100

​	

**Training Command Option[1] :: Background에서 Training**
python trian.py &   - back 작업

​	

**Training Command Option[2] :: Training Log Recording**
python trian.py &> train.log

​	

**Training Command Option[3] :: Background Training + Log Recording**
python trian.py &> train.log &  - back 작업 + Log 기록 남기기

​	

**Background Training시, Training 중지하는 방법**
nvidia-smi
sudo kill -9 <Process iD>

​	

**Training Command Final[Python Command Line Arguments]**
python train.py train --env='fasterrcnn' --plot-every=100 &> train.log &

​	

**Test Code**
python train.py train --env='fasterrcnn-test' --plot-every=100

​	

​	

​	

* **MMdetection** Framework's Commands

**Inference**
python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --show-dir faster_rcnn_r50_fpn_1x_results

​	

**Train - Single GPU**
python tools/train.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --auto-scale-lr

​	

**Train - Multiple GPU & Log Train Back Env.**
bash ./tools/dist_train.sh &> MMtrain.log \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    8 \
    --auto-scale-lr

​	

**설치**
%pip install ipdb
%pip install scikit-image
%pip install visdom
%pip install torchnet
%pip install ipywidgets
pip install opencv-python
pip install Image
pip install --upgrade pip
pip install torchvision==0.7.0
ipykernel package
conda install -n base ipykernel --update-deps --force-reinstall

​	

​	

#### (+) Memo

**Deep-Learning Model의 경량 정도 및 빠르기 및 효율성 평가 지표** 
flops / map
