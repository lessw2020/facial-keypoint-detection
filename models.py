## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
#partial padding...
from partialconv2d import PartialConv2d





class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layers
        
        #self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 48, kernel_size = 7)
        self.conv1 = PartialConv2d(in_channels = 1, out_channels = 48, kernel_size = 4)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = PartialConv2d(in_channels = 48, out_channels = 64, kernel_size = 3)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = PartialConv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.conv4 = PartialConv2d(in_channels = 128, out_channels = 256, kernel_size = 3)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        self.conv5 = PartialConv2d(in_channels = 256, out_channels = 512, kernel_size = 3)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        self.conv6 = PartialConv2d(in_channels = 512, out_channels = 1024, kernel_size = 3)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        
         # Maxpooling Layer
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        #bn layer
        self.bn1 = nn.BatchNorm2d(num_features = 48)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features = 128)
        self.bn4 = nn.BatchNorm2d(num_features = 256)
        self.bn5 = nn.BatchNorm2d(num_features = 512)
        self.bn6 = nn.BatchNorm2d(num_features = 1024)
        

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 1024, out_features = 700) 
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(in_features = 700,    out_features = 512)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(in_features = 512,    out_features = 136) #  2 for each of the 68 keypoint (x, y) pairs
        torch.nn.init.kaiming_normal_(self.fc3.weight)

        # Dropouts
        self.drop2 = nn.Dropout(p = 0.2)
        
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
      
        #first pass...lets do step by step      
        x = self.conv1(x)
        x = self.bn1(x) # apply bn before relu...
        x = F.relu(x)
        x = self.pool(x)
       
        
 
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        #print("Final shape:  ",x.shape)
        
        
        x = self.drop2(x)  #drop only before fc layers            

        # Flatten
        x = x.view(x.size(0), -1)
                      
        #print("Check - Flatten size: ", x.shape)
        #end of convolution section...begin linear networks for final prep and predictions             
        
        # fc1
        x = F.relu(self.fc1(x))
        #fc2
        x = F.relu(self.fc2(x))
        #print("Second fc size: ", x.shape)

        # fc3
        x = self.fc3(x)
        #print("Final prediction ready, size: ", x.shape)

        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

#let's build a basic DenseNet (used medium tutorial)
#first the basic denseblock - concatentates outputs in each layer...

class DenseBlock(nn.Module):
    def forward(self,x):
        bn = self.bn(x)   #normalize
        
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        
        #start adding dense net connections via concat
        dense2 = self.relu(torch.cat([conv1,conv2],1))
        
        conv3 = self.relu(self.conv3(dense2))
        dense3 = self.relu(torch.cat([conv1,conv2, conv3],1))
        
        conv4 = self.relu(self.conv4(dense3))
        dense4 = self.relu(torch.cat([conv1,conv2,conv3,conv4],1))
        
        conv5 = self.relu(self.conv5(dense4))
        dense5 = self.relu(torch.cat([conv1,conv2, conv3, conv4, conv5],1))
        
        return dense5
         
        
        
    def __init__(self, in_channels):
        super(DenseBlock,self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features = in_channels)
        
        #conv layers
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size =3, stride = 1, padding =1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size =3, stride =1, padding =1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size =3, stride =1, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size =3, stride =1, padding=1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels =32, kernel_size =3, stride =1, padding=1)
        
class TransitionLayer(nn.Module):
    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out
    
    
    
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size =2, stride =2, padding=0)
        
class FaceDenseNet(nn.Module):
    def make_dense(self, block, in_channels):
        layers =[]
        layers.append(block(in_channels))
        return nn.Sequential(*layers)
    
    def make_transition(self, layer, in_channels, out_channels):
        modules =[]
        modules.append(layer(in_channels, out_channels))
        return nn.Sequential(*modules)
    
    def forward(self,x):
        print("dense block trace -----------",x.shape)
        
        out = self.relu(self.lowconv(x))
        print("after first relu: ",out.shape)
        out = self.denseblock1(out)
        out = self.transition1(out)
        
        out = self.denseblock2(out)
        out = self.transition2(out)
        
        out = self.denseblock3(out)
        out = self.transition3(out)
        print(" after final transition - ",out.shape)
        
        out = self.bn(out)
        print("after final bn...",out.size())
        out = out.view(out.size(0),-1 )
        print("after view: ",out.size())
        out = self.pre_classifier(out)
        print("after preclass: ",out.size())
        out = self.classifier(out)
        print("after classifier ",out.size())
        
        return out    
        
        
    def __init__(self, in_channels, final_output):
        super(FaceDenseNet, self).__init__()
        
        self.lowconv = nn.Conv2d(in_channels =in_channels, out_channels=64, kernel_size=7, padding=3, bias = False)
        
        self.relu = nn.ReLU()
        
        #make blocks
        self.denseblock1 = self.make_dense(DenseBlock,64)
        self.denseblock2 = self.make_dense(DenseBlock, 128)
        self.denseblock3 = self.make_dense(DenseBlock, 128)
        
        #make transitions
        self.transition1 = self.make_transition(TransitionLayer, in_channels=160, out_channels = 128)
        self.transition2 = self.make_transition(TransitionLayer, in_channels=160, out_channels = 128)
        self.transition3 = self.make_transition(TransitionLayer, in_channels=160, out_channels = 64)
        
        #classifier
        self.bn = nn.BatchNorm2d(num_features=64)
        self.pre_classifier = nn.Linear(50176, 512)
        self.classifier = nn.Linear(512,final_output)
        
       
        
            
         

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    