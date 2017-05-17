# Behavioral Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./images_report/model.png "Model Visualization"
[middle_driving]: ./data/IMG/center_2017_04_20_16_50_45_786.jpg "Grayscaling"
[sin1]: ./data/IMG/center_2017_05_12_12_03_10_856.jpg "Recovery Image"
[sin2]: ./data/IMG/center_2017_05_12_12_03_11_139.jpg "Recovery Image"
[sin3]: ./data/IMG/center_2017_05_12_12_03_11_285.jpg "Recovery Image"
[sin4]: ./data/IMG/center_2017_05_12_12_03_11_576.jpg "Normal Image"
[sin5]: ./data/IMG/center_2017_05_12_12_03_11_721.jpg "Flipped Image"
[dataset]: ./images_report/figure_dataset.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consist in an architecture based in the NVIDIA architecture. Although there is some differences. First of all, NVIDIA architecture use convilutional layers with stride but in my case I use a maxpooling layer 2x2 after the convolutional layer (code line 86, 90 & 94).  
The model contains two preprocessing layers in the beginning, the first one crop the image: 
- 60 pixels on the top, so I can delete unuseful data from the environment.
- 20 pixels on the bottom, there is a part of the image that is unuseful because it is the front of the car.

#### 2. Attempts to reduce overfitting in the model

There is not any attempt to reduce overfitting in the model because the use of dropout layers was not satisfactory improving the result.

There was different test with different data so it would be possible to test the robustness of the architecture with different data.

The car is able to take several laps in the circuit in the two directions without going out of the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Basically there is data of center driving and a lap doing sinusuidal driving around all the track. This means that I was driving inside of the track all the time but not keeping straigh the steering wheel. This helped the car to take of different aproaches the road with a beter generalization. As a downside, in the curves the car does not go in the center of the road but is still inside the track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

To find an optimum solution, I started with only a layer depth architecture which it did not work properly as one could expect. Then I added several fully connected layers to see how the architecture worked driving around the circuit. This fact only helped because it softened the output of the steering angle, although it did not help to help the car to stay between the lines.  

I added several convolutional layers to test if the driving improved but i still had the same problem as before, the output was smooth but somehow the drving of the net seemed with a random output. At this point I realized I had a problem somewhere because I was having a low validation loss but in driving in the circuit wasn't correct. I could see there was a problem, the images which drive.py file receives are RGB but the ones that the model was training for were BGR.  

Once the image adquisition problem was solver the car seems to drive properly, although in some corners went out of the road. To solve the problem I tried to adquire more data in those corners, but it did not help. The car still went out of the road. Then I decided to increase the depth of the architecture because it looked like the model was not able to detect enough features in the road.  
So I deciced to use the NVIDIA architecture. Once I implemented it, the was an improvement when the car turn but still there was a corner that could not approach properly. Taking more data in the same turn did not help.  
Because the GPU in my computer is not fast enough I was recording the data in the fastest mode of the simulator and I realized that the quality of the simulator affected the quality of the images recorded. I decided to record the training data in good quality on the simulator and run the autonomous mode in good quality too. This solved the problem instantly.  
The NVIDIA architecture was able to do the full length of the track in both ways without going out of track.

#### 2. Final Model Architecture

Here in the figure you can find the final architecture used to train the model.

![Representation of the model][model]

#### 3. Creation of the Training Set & Training Process

To achieve a good driver behavour I decided to make the car drive first a full lap around the circuit in the middle of the road all the time. 
![alt text][middle_driving]  

This made the car make almost a full lap without any problem, although I wanted to have a more robust driving during all the lap and a achieve a better generalization of the model so I decided to take another full lap but in this case driving from side to side during all the circuit, and then in the opposite direction. Of this way I was able to have a better generalization of the driving because the car was able to recover from every part of the inner part of the track without any problem. As a downside, in the turns the car doesn't always go in the middle of the road.

![alt text][sin1]  
![alt text][sin2]  
![alt text][sin3]  
![alt text][sin4]  
![alt text][sin5]  

Here you can see the dataset of the steering angles.

![alt text][dataset]

To augment data on training runtime, the generator was flipping the images so the model will not overgeneralize turning in certain side than another. Because as you can see in the dataset, there is more turning angles to the left than the right.

Once I had the dataset ready I splitted it in a training dataset and a validation dataset which was a 20% of the full dataset.

I used this validation test to be sure on training that the model was correct.

I need to use a generator due to hardware limitations, the dataset was suffled everytime data was catched in runtime.  

The batch size used was 2 because of memory limitations so 2 epochs were enough to train the model. I tried to train with more epochs which did not improve the performance.  

Due to the small number of epochs of training I wasn't able to use dropout as a method to avoid overfitting.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
