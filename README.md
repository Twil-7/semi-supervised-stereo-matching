# Semi-supervised Stereo Matching Framework

## 1. Introduction
In recent years, deep stereo matching algorithms have made impressive achievements both in terms of disparity accuracy and computational efficiency. However, to make these algorithms truly practical in the real-world scenario, we are still faced with some obstacles. 

One major problem is the scarcity of ground truth disparities. Unlike image classification, object detection, semantic segmentation and other computer vision tasks, ground truth disparities of stereo matching are rather difficult to obtain, and even cannot be labeled manually. Another major problem is domain shift. Just like most learning-based methods, models trained from samples in the source domain suffer from substantial performance drop when directly performed in the target domain. 

In this paper, we propose a Semi-supervised Stereo Matching Framework (SSMF), i.e., a continuous self-training pipeline involving both teacher model and student model. The proposed framework combines the consistency regularization with the entropy minimization to effectively utilize the unlabeled data in large quantities. To the best of our knowledge, this is the first semi-supervised learning framework for stereo matching, which exhibits impressive performance on both accuracy and robustness.

<div align=center>
<img src="https://github.com/Twil-7/semi-supervised-stereo-matching/blob/main/pipeline.png" width="434" height="351" alt="pipeline"/><br/>
</div>


Comprehensive experimental results show that the proposed framework enables to largely improve the disparity accuracy and robustness. Moreover, it also demonstrates competitive performance in cross-domain scenarios. Among all published methods as of August 2023, it achieves 1st on KITTI 2012 benchmark and 4th on KITTI 2015 benchmark. The code is being sorted out and will be released soon.


## 2. Dataset
In this code, we use the Sceneflow dataset and Gwcnet model as the baseline of the semi supervised stereo matching framework.
SceneFlow Dataset: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
GwcNet network: https://github.com/xy-guo/GwcNet


## 3. Code
The whole code is divided into two parts: teacher model training and student model training. 

(1) teacher model training

Taking the sceneflow dataset and gwcnet model as examples, for the initial teacher model training, we include the following four steps:

Step 1: divide the training set and the test set. For the training set data, we divide it into labeled data and unlabeled data.
```
# Please remember to change the filepath = "/ssd2/xufudong_dataset/sceneflow/" to yours.
python annotation.py
```
Step 2: train the teacher model on the labeled training set.
```
python train.py
```

Step 3: test the performance of teacher model.
```
# Please remember to change the weight_path = "0098_3.59319_0.32577_0.96966_GwcNet_sceneflow.tar" to yours.
python test.py
python test_visualize.py
```
Step 4: use the trained teacher model to generate pseudo labels on the unlabeled training set.

```
# Please remember to check the validity of save_name.
pseudo1_make.py
pseudo1_txt.py
pseudo1_show.py
```

(2) student model training

Taking the sceneflow dataset and gwcnet model as examples, for the student model training, we include the following three steps:

Step 1: move the "model_data/" and "Pseudo_stage1/" files from the teacher_model file to the student_model file.

```
cp -r path1 path2
```

Step 2: train the student model on the mixed labeled and pseudo-labeled training set.

```
python train.py
```

Step 2: test the performance of student model.

```
# Please remember to change the weight_path = "0098_3.59319_0.32577_0.96966_GwcNet_sceneflow.tar" to yours.
python test.py
```
