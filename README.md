# facial-keypoint-detection
Facial keypoint detection CNN - custom architecture using partial convolution padding.  This part of the Udacity computer vision nanodegree, and was quite a fun project!

Example result:


# Facial Keypoint Detection
This project will be all about defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to transform images of faces. The first step in any challenge like this will be to load and visualize the data you'll be working with.



Net(
  (conv1): PartialConv2d(1, 48, kernel_size=(4, 4), stride=(1, 1))
  (conv2): PartialConv2d(48, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv3): PartialConv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (conv4): PartialConv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  (conv5): PartialConv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
  (conv6): PartialConv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1))
  (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (bn1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn6): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=1024, out_features=700, bias=True)
  (fc2): Linear(in_features=700, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=136, bias=True)
  (drop2): Dropout(p=0.2)
)


