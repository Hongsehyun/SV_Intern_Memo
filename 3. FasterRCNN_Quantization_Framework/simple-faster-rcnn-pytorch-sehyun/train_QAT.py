from __future__ import  absolute_import

import os   
import random
import numpy as np
import cv2 as cv
from cv2 import * 
import torch

import ipdb                    # Debugging Tool
                               # https://stricky.tistory.com/93 
import matplotlib
from tqdm import tqdm          # 연속적인 작업을 수행할때 진행률 보다 깔끔하게 표시해주는 Library
                               # https://jangjy.tistory.com/342

# Pre-Defined Model and Function Call
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

import visdom
vis = visdom.Visdom()

# # fix for ulimit
# # https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))  

matplotlib.use('agg')



def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()   # 예측한 Bounding Box, 예측한 Class, 예측 점수 저장 LIST
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()     # 실제 Bounding Box, 실제 Class, 실제 점수 저장 LIST

    # for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(tqdm(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])

        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())

        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result



def DeleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return 'Remove All File'
    else:
        return 'Directory Not Found'



# define a floating point model where some layers could benefit from QAT
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x



def train(**kwargs):   # **kwargs = 딕셔너리 형태로 값을 저장. https://legitcode267.tistory.com/13

    # Run Visdom
    vis = visdom.Visdom()

    opt._parse(kwargs) #  _parse  = 추출하고자 하는 문자열의 패턴
    dataset = Dataset(opt)

        

    # create a model instance
    model_fp32 = M()

    # model must be set to train mode for QAT logic to work
    model_fp32.train()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference and
    # 'qnnpack' for mobile inference. Other quantization configurations such
    # as selecting symmetric or assymetric quantization and MinMax or L2Norm
    # calibration techniques can be specified here.
    model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # fuse the activations to preceding layers, where applicable
    # this needs to be done manually depending on the model architecture
    model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
        [['conv', 'bn', 'relu']])

    # Prepare the model for QAT. This inserts observers and fake_quants in
    # the model that will observe weight and activation tensors during calibration.
    model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)



    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers
                                  # num_workers=0)
                                 )
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset, \
                                       batch_size=1, \
                                       num_workers=opt.test_num_workers, \
                                       # num_workers=0, \
                                       shuffle=False
                                       # pin_memory=True
                                       )
                                       
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    print('trainer construct completed')

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    # Visdom Print[1] :: Print All Label Classes - 1회 실행!
    # Visdom Print[1] :: Print All Label Classes - 1회 실행!
    trainer.vis.text(dataset.db.label_names, win='labels', opts = dict(title='[1] PASCAL VOC Classes'))
    best_map = 0
    lr_ = opt.lr

    for epoch in range(opt.epoch):
        
        print('\n**********          Note!   Current Epoch is  {}          **********'.format(epoch))
        print('**********          Note!   Current Epoch is  {}          **********'.format(epoch))
        print('**********          Note!   Current Epoch is  {}          **********\n'.format(epoch))

        trainer.reset_meters()   # Reset All Loss Valeus

        # for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        for ii, (img, bbox_, label_, scale) in enumerate(tqdm(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            model_fp32_prepared.train_step(img, bbox, label, scale)


            # run the training loop (not shown)
            # training_loop(model_fp32_prepared)


            if (ii + 1) % opt.plot_every == 0:   # if plot-every option = 100, print every 100 cycles. 
                
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                # Visdom Print[2~6] :: 100번째 데이터 값을 읽어와서 5가지의 Loss 값을 Plotting  - 매 Iteration 마다 출력 (100개의 이미지 학습할 때마다 값 넘겨주기, 누적 Plot)
                # Visdom Print[2~6] :: 100번째 데이터 값을 읽어와서 5가지의 Loss 값을 Plotting  - 매 Iteration 마다 출력 (100개의 이미지 학습할 때마다 값 넘겨주기, 누적 Plot)
                trainer.vis.plot_many(trainer.get_meter_data()) 

                # plot groud truth bboxes
                # Visdom Print[7] :: 100번째 사진을 가져와서 Ground Truth Label의 Class와 Bounding Box 자체를 Showing! - 매 Iteration, 이미지 100개 마다 초기화 및 갱신
                # Visdom Print[7] :: 100번째 사진을 가져와서 Ground Truth Label의 Class와 Bounding Box 자체를 Showing! - 매 Iteration, 이미지 100개 마다 초기화 및 갱신
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('[7] Ground-Truth Image', gt_img)
                
                # plot predict bboxes
                # Visdom Print[8] :: 100번째 사진을 가져와서 predict 시키고, 예측 Class와 Score 및 Bbox를 Showing! - 매 Iteration, 이미지 100개 마다 초기화 및 갱신
                # Visdom Print[8] :: 100번째 사진을 가져와서 predict 시키고, 예측 Class와 Score 및 Bbox를 Showing! - 매 Iteration, 이미지 100개 마다 초기화 및 갱신
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('[8] Predicted Image', pred_img)  

                # rpn confusion matrix(meter)   
                # Visdom Print[9] :: 100번째 사진의 RPN, 제안된 Region에 실제로 물체가 있는지 없는지 2 by 2 Confusion Matrix Print - 매 Iteration, 이미지 100개 마다 초기화 및 갱신
                # Visdom Print[9] :: 100번째 사진의 RPN, 제안된 Region에 실제로 물체가 있는지 없는지 2 by 2 Confusion Matrix Print - 매 Iteration, 이미지 100개 마다 초기화 및 갱신
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm', opts = dict(title='[9] rpn_Confusion_Matrix'))          

                # roi confusion matrix
                # Visdom Print[10] :: 100번째 사진의 RoI Confusion Matrix - 매 Iteration, 이미지 100개 마다 초기화 및 갱신
                # Visdom Print[10] :: 100번째 사진의 RoI Confusion Matrix - 매 Iteration, 이미지 100개 마다 초기화 및 갱신
                trainer.vis.img('[10] roi_Confusion_Matrix', at.totensor(trainer.roi_cm.conf, False).float())
                

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

        # Visdom Print[11] :: 매 Epoch 마다 출력, 5000장의 이미지 학습 한 결과를 Test Data에 적용하여 mAP 계산 후 Plot
        # Visdom Print[11] :: 매 Epoch 마다 출력, 5000장의 이미지 학습 한 결과를 Test Data에 적용하여 mAP 계산 후 Plot
        trainer.vis.plot('[11] Test Dataset mAP', eval_result['map'])   # map = mAP (mean Average Precision)
        
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']


        # Visdom Print[12] :: 매 Epoch 마다 출력, 5000장의 이미지 학습 한 결과를 Test Data에 적용한 종합적인 성능을 출력
        # Visdom Print[12] :: 매 Epoch 마다 출력, 5000장의 이미지 학습 한 결과를 Test Data에 적용한 종합적인 성능을 출력
        log_info = 'epoch:{}, lr:{}, map:{}, loss:{}'.format(str(epoch),
                                                             str(lr_),
                                                             str(eval_result['map']),
                                                             str(trainer.get_meter_data()))
        trainer.vis.log(log_info)


        # Model Save Definition
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save_func(best_map=best_map, epoch=epoch)
            


        # Learning Rate Change :: 이미 충분히 학습하였다고 판단되는 Epoch인 '9'부터는 Learning Rate를 낮춰서 Global Minimum을 찾아보도록 한다.
        #                         처음부터 Learning Rate를 낮게 설정해버리면, Local Minimum에 빠지는 문제가 발생할 수 있다.
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay


        if epoch == 0: 
            current_path = os.getcwd()
            torch.save(trainer.state_dict(), 'tmp.pt')
            break



if __name__ == '__main__':
    import fire

    fire.Fire()