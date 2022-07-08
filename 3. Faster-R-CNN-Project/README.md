# Dataset

​	

**PASCAL VOC 2007 Dataset**

Total Images :: 9963
Train/Val Images :: 5011
Test Images :: 4952

Epoch :: 14
Learning Rate :: 0.001
Batch Size :: 1

![train_log_visualization](https://user-images.githubusercontent.com/84533279/177906225-a9c836ec-2142-48a5-bd71-c1ff08d533eb.JPG)


Visdom Window Explanation

----- 최초 1회 실행 -----
[1] Pascal VOC Classes :: PASCAL Dataset의 모든 Class 출력


----- 100개의 Image 학습할 때마다 실행(= 50 iteration per 1epoch)  -----
[2] rpn_loc_loss :: Smooth L1 Loss로 계산
[3] rpn_cls_loss :: CrossEntropy Loss로 계산
[4] roi_loc_loss :: Smooth L1 Loss로 계산
[5] roi_cls_loss :: CrossEntropy Loss로 계산
[6] total_loss ::[2], [3], [4], [5]의 합
      X축 :: One Points per 100 Images = 50 Points per 1 Epoch = 750 Points per 15 Epochs
      Y축 :: Loss Value

[7] Ground-Truth Image :: 원본 이미지 + Ground Truth Bounding Box + Class 

[8] Predicted Image :: 원본 이미지 + Predicted Bounding Box + Predicted Class + Predicted Score

[9] rpn_Confusion_Matrix :: 100번째 사진의 RPN, 제안된 Region에 실제로 물체가 있는지 없는지 2 by 2 Confusion Matrix Print
                                               대각 성분[TN, TP]값이 점점 많아짐 -> 정확도가 좋아짐(예측과 정답이 같은 경우가 점점 많아지는 것)

[10] roi_Confusion_Matrix :: 100번째 사진의 RoI Confusion Matrix
                                                 가로 X 세로 = 20 X 20 ( Class Label의 갯수 )


-----  매 epoch마다 실행  -----
[11] test_map :: @mAP -> Model의 전반적인 Accuray를 매 Epoch마다 Plot

[12] Log :: epoch , learning Rate, mAP, 5가지 Loss를 출력하여 매 Epoch마다 Model의 Training 정도를 확인
