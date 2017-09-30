#**Traffic Sign Recognition**

## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/wandonye/carnd_P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are distributed across different labels

![bar chart of class distribution][./visualization/bar_chart_distribution.png]
Randomly sampled images shown below:
![10 random training images][./visualization/random_training_img_original.png]
![5 random training images per label][./visualization/5_img_each_class.png]
![20 random training images per label][./visualization/20_img_each_class.png]

###Design and Test a Model Architecture

####1. Preprocess the image data.
What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I created a grayscale version for each image. In the later modeling step, I tried both the grayscale and the color version.

Then for both grayscale and color images, I decided to apply histogram equalization because by exploring the dataset I found there are many over-exposed or under-exposed images. Such images make it difficult to train the weights. For example, features in the under-exposed images might be ignored because the weights are not big enough to pickup the features. The effect of histogram equalization is similar to normalization of a feature in other machine learning method. Note that for color images, the histogram equalization was done by first converting RGB to YUV and only balancing the Y component (lighting).

Then I augmented the data by rotation and perspective transformations. I restricted rotation to +/- 15 degrees since a car camera seldomly see traffic signs in an angle outside of that range. The generation of perspective transformations is also restricted in the sense that the resulting image should not be too small, and the camera angle shouldn't be too off-center.

Here are some examples of preprocessed images
![preprocessed][./visualization/preprocessed.png]

To balance the training data, I kept all the original training images, and generated new images using rotation and perspective transformation, so that each sign has about 8000 images. This balancing method worked better in validation accuracy than the other method where I random sampled 1000 or all (whichever is smaller) samples from each class, and generated 6 images using rotation and perspective transformation from each image in these samples.

The balanced data now has distribution:

![preprocessed_distribution][./visualization/balanced.png]


####2. The best result was achieved using a VGG model. The design is given below:

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 30x30x8 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x12x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| Input = 5x5x16 Output = 400.      									|
| Fully connected		| Input = 400 Output = 120    									|
| RELU					|												|
| Dropout		|         									|
| Fully connected		| Input = 120 Output = 84 								|
| RELU					|												|
| Dropout		|         									|
| Fully connected		| Input = 84 Output = 34 	  						|
|						|												|
|						|												|



####3. Model training settings and hyperparameters.

To train the model, I used AdamOptimizer with learning rate = 0.001, BATCH_SIZE = 256. I usually start training with 40 epochs, and continue training if needed.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model (VGG model) results were:
* training set accuracy of 95.2%
* validation set accuracy of 96.2%
* test set accuracy of 92.7%

Before reaching the final model, I tried the following:
* The first architecture I tried was LeNet as suggested in the description of the project. Without balancing the data, I was able to achieve 100% training accuracy and 95% validation accuracy.
* While the number looks good, when I apply the model to web images, 0 out 7 images were correctly predicted. So obviously there is overfitting. Another hint of overfitting is that validation accuracy stuck at 95% while training accuracy went to 100%. So I augmented the training data using rotation and perspective transformations, and added dropout.
* With more data that are balanced, and dropout layers, validation accuracy plateaued with 93.2% and training accuracy kept improve and passed 95%. It's possible that after the augmentation, I may need more filters to pickup the features.

* Since we have more data with augmentation, I decided to increase parameters in the model. In particular, I increased number of filters in conv1 layer of LeNet (from 6 to 16).
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

* Finally I borrowed the idea from VGG: use consecutive small (3x3) filters to replace large filters (5x5). Two pairs of (3x3) conv layers gave a receptive field of 10x10. This is reasonable for a 32x32 image with basically one object occupying most portion of the image.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. I downloaded 7 German traffic signs, 6 of which are signs existing in the 43 labels of the training data. The last one is double curve, which is not included in the training data. I added this one purely out of curiosity.

![images from internet][web_imgages/websigns.png]

The first image might be difficult to classify because its difference from `Turn right ahead` is minor. Speed limit (30km/h) is sometimes difficult to tell from 50km/h. The `stop` sign and `Turn left ahead` sign presented extra difficulty because of the perspective view.

####2. Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Go straight or right      		| Go straight or right  									|
| Turn left ahead    			| Turn left ahead										|
| General caution					| General caution											|
| Stop					| Stop									|
| Speed limit (30km/h)		| Speed limit (30km/h)		 				|
| No entry			| No entry     							|
| Double Curve		| Children crossing    			|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.7%. The model was able to deal with different perspective angles because of the augmentation I did in the preprocessing step.

####3. The top 5 softmax probabilities for each image along with the sign type of each probability.

Image 0: As expected, other than `Go straight or right`, the next guess of this image is `Turn right ahead`. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.809361      		| Go straight or right 									|
| 0.190637    		| Turn right ahead									|
| 1.70663e-06			| Ahead only									|
| 2.45226e-07     	| Keep right			 				|
| 1.27938e-08	    | End of no passing by vehicles over 3.5 metric tons    							|

Image 1: Model is very confident in predicting `Turn left ahead`
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.998798    		| Turn left ahead							|
| 0.000646399    	| Keep right 				|
| 0.000517454 		| Ahead only				|
| 1.95769e-05	| Go straight or left								|
| 1.75404e-05    | Go straight or right		|

Image 2: Model is very confident in predicting `General caution`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.996853   		| General caution		|
| 0.00138585   	| Road narrows on the right	|
| 0.00111512		| Pedestrians |
| 0.00057348 	| Traffic signals	|
| 7.23495e-05  | Right-of-way at the next intersection |

Image 3: Model is extremely confident in predicting `Stop` sign. This is because the shape of the symbol on this sign is quite unique. The model had no difficulty with the viewing angle of this image.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.996853   		| Stop	|
| 1.24687e-15  	| Speed limit (30km/h) 	|
| 4.76721e-18	| Road work |
| 4.07421e-19	| Road narrows on the right	|
| 2.47972e-20  | Speed limit (20km/h)  |

Image 4: Model is very confident in predicting `Speed limit (30km/h)`, but also considered `Speed limit (50km/h)`
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.925796 		| Speed limit (30km/h)|
| 0.0731715  	| Speed limit (50km/h) 	|
| 0.000847794	| Speed limit (70km/h) |
| 9.14498e-05	| Speed limit (100km/h)	|
| 5.41302e-05 | Speed limit (20km/h)  |

Image 5: Model is extremely confident in predicting `No entry` sign.
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0		| No entry |
| 6.38649e-18  	| No passing for vehicles over 3.5 metric tons 	|
| 2.81191e-18	| Stop |
| 1.06696e-19	| Yield	|
| 3.82698e-29 | No passing |

Image 6: This is the one outside of the training labels. Top prediction is Children crossing.
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.984269	| Children crossing |
| 0.0113422 	| Beware of ice/snow 	|
| 0.00183505	| Road narrows on the right |
| 0.00163407	| Dangerous curve to the right	|
| 000479989 | Pedestrians |

For comparison:
![compare][web_imgages/comparison.png]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
