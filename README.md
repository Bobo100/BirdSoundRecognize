# BirdSoundRecognize
# Introduce

Here is the training stage for bird sound recognition

I have prepared two versions which are resnet and efficientnet versions

The resnet version refers to the code provided by the author [Stefan Kahl](https://github.com/kahst/BirdCLEF-Baseline) on github to modify it to the tensorflow version

And efficientnet is a reference [SOUND-BASED BIRD CLASSIFICATION. How group of Polish women used deep… | by Magdalena Kortas | Towards Data Science](https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b)

Both work in the context of win10

python version is 3.7.10

| Equipment             |           |
|-----------------------|-----------|
| System                | Window 10 |
| CPU                   | intel i7-6700   |
| GPU                   | GTX 1070  |
| CUDA                  | 8.0       |
| cuDNN                 | 7.5       |
| RAM                   | 24GB      |
| Development Platform  | Anaconda  |
| Programming Language  | Python    |



If there is a situation where it cannot be implemented

Welcome to leave a message, I will try my best to answer

The flow chart of the system is as follows:
![alt text](image/model%20flow%20chart.png "dataset_path")

Model used in smartphone / APP link [Link](https://github.com/Bobo100/BirdSoundRecognizeAPP)

# Dataset

I use crawler to grab all the bird sound files of [xeno-canto](https://xeno-canto.org/)

# ResNet

## Process

Please prepare the materials you want to train first and organize them

The folder structure is as follows:

```livescript
\DATASET-DOWNLOAD\DATASET_INPUT_FOLDER_NAME
|
+---species1
|       species1_1.wav
|       species1_2.wav     
+---species2
|       species2_1.wav
|       species2_2.wav
```


Then extract features (the settings here use the settings of [Stefan Kahl](https://github.com/kahst/BirdCLEF-Baseline))

First open config.py to change the path of data set input and output

![alt text](image/config%20folder%20path.PNG "dataset_path")

Then you can execute spec.py to extract features

Note: The file for extracting features must be in wav format

If it is mp3 format, it needs to be converted

Please go to the dataset dataset-download and execute mp3_to_wav.py

Remember to modify the path of the mp3 folder

After extracting features

Then can be trained

There are two approaches here, the difference is the speed of training and accuracy

The first method is to use the image file just output directly as input to train

The second method is to make the image file just output into tfrecord for training

However, the floating values of my method 2 training are large and I haven't figured out why


## Method 1
Execute mytrain_resnet18_ver1.py

will start training

## Method 2

not update

# After

Then I'll put the model on my phone and make instant predictions

APP will be introduced in another project