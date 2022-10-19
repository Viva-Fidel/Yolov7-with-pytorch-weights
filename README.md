# Yolov7 with pytorch weights

## Introduction
This is a modified code of [Yolov7](https://github.com/WongKinYiu/yolov7). that gives you the ability to use your own weights with your video or with any other source without need to use of argparse. 

## Installing
>!git clone https://github.com/Viva-Fidel/Yolov7-with-pytorch-weights.git

Additionally install CUDA if you want to use GPU for processing


## Requirements

opencv-python~=4.6.0.66  
numpy~=1.23.4  
torch~=1.12.1  
torchvision~=0.13.1  
requests~=2.28. 

>!pip install -r requirements.txt

## Usage

<b>self.weights</b> - provide a path to your .pt weights file  
<b>self.source</b> - provide a path to your video file  
![image](https://user-images.githubusercontent.com/98227548/196693746-45be9f93-ae59-4c3e-8faa-1d7a9fdcba5f.png)
