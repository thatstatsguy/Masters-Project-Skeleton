# Masters-Project-Skeleton
**Overview**<br>
This repository contains the majority of the skeleton code used in my masters project. A deep learning approach was used in tackling the given task on some dummy data. The MobileNetV1, MobileNetV2 and VGG-16 models were attempted to solve the given task. In addition, the standard Adam and SGD optimisers were used in implementing the models. The VGG-16 model with the adam optimiser achieved an out of sample accuracy of 95% on unseen data.

In order to (try) train with constant memory, the ImageDataGenerator iterator was used to feed in the image data dynamically in order to not have to load all the images into memory. Additionally, the IBM Large Model Support package was installed to implement a call-back feature which can assist in preventing out of memory errors when working with large data on the graphics cards.
The solution is in the form of a python script. To run this script (and train efficiently) you will need to install tensorflow-gpu and keras.
<br><br>**Interesting Links/Further Work**<br>
Due to external time constraints, I could not implement some further additions to the project. However, some further work that can be done includes: 
<br> 
- **Starting Parameter Optimisation**
<br>The performance of VGG16 model is highly sensitive to the initialized weights of the added fully connected layers. Perhaps a method can be determined to find a more concrete way to ensure a “good” starting weights. 

- **Histogram equalisation** 
<br>(Link: https://towardsdatascience.com/image-augmentation-for-deep-learning-using-keras-and-histogram-equalization-9329f6ae5085)

- **Progressive resizing of input images**
<br>(Link: https://towardsdatascience.com/boost-your-cnn-image-classifier-performance-with-progressive-resizing-in-keras-a7d96da06e20)

**To Run Script**
- Clone copy of repository.
- Unzip animal data folder. This contained a sorted copy of the cat and dog data which is separated into two folders.
- Install all dependencies listed on the import statement.
- If you have a sufficient system and don't need large model support, delete import statement on line 12 as well as the callbacks in line 138 and 144.
- Install tensorflow-gpu with all the necessary extra software to use the gpu.
- Install Keras

**System Details**
- AMD Ryzen Threadripper 1950X
- 128GB
- 64-bit Windows 10 Professional
- Nvidia GeForce GTX 1080Ti
-500GB SSD
- Python 3.6.5 (v3.6.5:f59c0932b4, Mar 28 2018, 17:00:18)
- Tensorflow-gpu version 1.12
- Keras version 2.2.4
- JetBrains Pycharm IDE
