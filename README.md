# semi-supervised-stereo-matching

In recent years, with the advent of deep learning, deep stereo matching algorithms have made impressive achievements both in terms of disparity accuracy and computational efficiency. However, to make these algorithms truly practical in the real-world scenario, we are still faced with some obstacles.

One major problem is the scarcity of ground truth disparities. Unlike image classification, object detection, semantic segmentation and other computer vision tasks, ground truth disparities of stereo matching are rather difficult to obtain, and even cannot be labeled manually. Another major problem is domain shift. Just like most learning-based methods, models trained from samples in the source domain suffer from substantial performance drop when directly performed in the target domain. 
