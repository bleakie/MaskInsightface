# FaceQnet
FaceQnet: Quality Assessment for Face Recognition based on Deep Learning

This repository contains the DNN FaceQnet presented in the paper: <a href="https://arxiv.org/abs/1904.01740" rel="nofollow">"FaceQnet: Quality Assessment for Face Recognition based on Deep Learning"</a>.

FaceQnet is a No-Reference, end-to-end Quality Assessment (QA) system for face recognition based on deep learning. 
The system consists of a Convolutional Neural Network that is able to predict the suitability of a specific input image for face recognition purposes. 
The training of FaceQnet is done using the VGGFace2 database.

-- Configuring environment in Windows:

1) Installing Conda: https://conda.io/projects/conda/en/latest/user-guide/install/windows.html

  Update Conda in the default environment:

    conda update conda
    conda upgrade --all

  Create a new environment:

    conda create -n [env-name]

  Activate the environment:

    conda activate [env-name]

2) Installing dependencies in your environment:

  Install Tensorflow and all its dependencies: 
    
    pip install tensorflow
    
  Install Keras:
  
    pip install keras
    
  Install OpenCV:

    conda install -c conda-forge opencv
  
 3) If you want to use a CUDA compatible GPU for faster predictions:
  
   You will need CUDA and the Nvidia drivers installed in your computer: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/
  
   Then, install the GPU version of Tensorflow:
    
    pip install tensorflow-gpu
  
-- Using FaceQnet for predicting scores:
  2) Due to the size of the video example, please download the FaceQnet pretrained model <a href="https://github.com/uam-biometrics/FaceQnet/releases/download/v1.1/FaceQnet.h5" rel="nofollow">here</a> (.h5 file) and place it in the /src folder.
