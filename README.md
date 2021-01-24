# Lyft-Motion-Prediction-for-Autonomous-Vehicles-final
Autonomous vehicles (AVs) are expected to dramatically redefine the future of transportation.
However, there are still significant engineering challenges to be solved before one can fully realize the benefits of self-driving cars. 
One such challenge is building models that reliably predict the movement of traffic agents around the AV, such as cars, cyclists, and pedestrians.  The ridesharing company Lyft started Level 5 to take on the self-driving challenge and build a full self-driving system (they’re hiring!). Their previous competition tasked participants with identifying 3D objects, an important step prior to detecting their movement. Now, they’re challenging you to predict the motion of these traffic agents.  In this competition, you’ll apply your data science skills to build motion prediction models for self-driving vehicles. You'll have access to the largest Prediction Dataset ever released to train and test your models. Your knowledge of machine learning will then be required to predict how cars, cyclists,and pedestrians move in the AV's environment.  Lyft’s mission is to improve people’s lives with the world’s best transportation. They believe in a future where self-driving cars make transportation safer, environment-friendly and more accessible for everyone. Their goal is to accelerate development across the industry by sharing data with researchers. As a result of your participation, you can have a hand in propelling the industry forward and helping people around the world benefit from self-driving cars sooner.  

## Table of Contents
* [Data Overview](#data_overview)
* [Predicting](#predicting)
* [Evaluation](#evaluation)
* [Learning section](#learning_section)
* [Activation function](#activation_function)
* [Loss function](#loss_function)
* [Methods](#methods)
* [Modeling](#modeling)
* [Further Discussion](#further_discussion)

## Data Overview <a name="data_overview"></a>
The Lyft Motion Prediction for Autonomous Vehicles competition is fairly unique, data-wise. In it, a very large amount of data is provided, which can be used in many different ways. Reading the data is also complex - please refer to Lyft's L5Kit module and sample notebooks to properly load the data and use it for training. Further Kaggle-specific sample notebooks will follow shortly.
  ### Files
    aerial_map - an aerial map used when rasterisation is performed with mode "py_satellite"
    semantic_map - a high definition semantic map used when rasterisation is performed with mode "py_semantic"
    sample.zarr - a small sample set, designed for exploration
    train.zarr - the training set, in .zarr format
    validate.zarr - a validation set (roughly the size of train)
    test.csv - the test set, in .zarr format
    mask.npz - a boolean mask for the test set. All and only the agents included in the mask should be submitted
    *sample_submission.csv - two sample submissions, one in multi-mode format, the other in single-mode
    train_full.csv - the complete training set, 10x bigger than train.zarr. Contained in a separate dataset, hosted on Kaggle here and also on Lyft (registration required).

Note also that this competition requires that submissions be made from kernels, and that internet must be turned off in your submission kernels. For your convenience, Lyft's l5kit module is provided via a utility script called kaggle_l5kit. Just attach it to your kernel, and the latest version of l5kit and all dependencies will be available.

## Predicting <a name="predicting"></a>
We are predicting the motion of the objects in a given scene. For test, we will have 99 frames of objects moving around will be asked to predict their location in the next 50.


## Evaluation <a name="evaluation"></a>
The goal of this competition is to predict the trajectories of other traffic participants. You can employ uni-modal models yielding a single prediction per sample, or multi-modal ones generating multiple hypotheses (up to 3) - further described by a confidence vector.

Due to the high amount of multi-modality and ambiguity in traffic scenes, the used evaluation metric to score this competition is tailored to account for multiple predictions.

Note: The following is a brief excerpt of our metrics page in the L5Kit repository

We calculate the negative log-likelihood of the ground truth data given the multi-modal predictions. Let us take a closer look at this. Assume, ground truth positions of a sample trajectory are


![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/gif.gif)

and we predict K hypotheses, represented by means

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/x1.gif)

In addition, we predict confidences c of these K hypotheses. We assume the ground truth positions to be modeled by a mixture of multi-dimensional independent Normal distributions over time, yielding the likelihood


![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/x2.gif)


which results in the loss

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/x3.gif)


## Learning section<a name="learning_section"></a>
	
**1. BGD batch gradient descent**

**2. SGD tochastic gradient descent**

**3. MBGD minibatch gradient descent**

**4. Learning rate** 

	Learning rate should be properly chosen, as a small learning rate would lead to a slow convergency and 
	a fast learning rate would make the loss function fluctuate
	
**5. Momentum**
	
	Momentum is similar to the concept of Momentum in physics, which means that the gradients of previous 
	times are also involved. In order to represent momentum, a new variable V (Velocity) is introduced. V is the 
	accumulation of previous gradients, but there is a certain attenuation in each turn.

	When the front and back gradient direction is consistent, learning can be accelerated

	When the front and back gradient direction is not consistent, the shock can be suppressed

	Hyperparameter setting value:

	The general value of γ is about 0.9.

**6. Nesterov accelerated gradient(NAG)**

	This is a variation on Momentum, and the idea is, you estimate the parameters, and then you use 
	the estimated parameters to calculate the error

	NAG can make RNN perform better on many tasks.

	So far, we can adjust the speed according to the gradient of the loss function and accelerate the SGD 
	when the gradient is updated.

	We also want to be able to update different parameters to varying degrees depending on their importance.

**7. Saddle point**
	
	The curve, surface, or hypersurface of the saddle point neighborhood of a smooth function is located on 
	different sides of the tangent to this point.

	For example, this two-dimensional graph, like a saddle: it bends up in the X-axis, it bends down in the Y-axis, 
	and the saddle point is 0,0.

	
![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/鞍点.png)

**8. RMSPropAdaGrad**

	This algorithm can do a larger update parameters of low frequency, less done on the high frequency of updates, 
	and therefore, its performance for sparse data is very good, especially for improving the robustness of SGD, such as 
	identification of Youtube video inside of the cat, training GloVe word embeddings, as they all need more updates 
	on the characteristics of low frequency.

	Adagrad has the advantage of reducing manual adjustment of learning rate

	Super parameter setting value:

	Usually η is 0.01.

	Disadvantages:

	The disadvantage is that the denominator keeps accumulating, so the learning rate shrinks and eventually becomes very small.
	
![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/图片1.png)

**9. RMSProp**

	RMSProp decays r by a certain percentage each turn by introducing a decaying factor, similar to what happens in Momentum.

	Hyperparameter setting value:

	It is recommended to set γγ as 0.9 and αα as 0.001.

	Advantages:

	Compared with Adagrad, this method is a good solution to the problem of premature ending in deep learning

	It is suitable for non-stationary targets and has good effect on RNN

	Disadvantages:

	A new superparameter, attenuation coefficient, is introduced

	It still depends on the global learning rate

**10. Adam(Adaptive Moment Estimation)**

	This algorithm is another way to calculate the adaptive learning rate for each parameter. At present in the field 
	of DL, Adam is the most common optimizer. The main advantage is that after bias correction, the learning rate 
	of each iteration has a certain range, which makes the parameters relatively stable.

	Hyperparameter setting value:

	Suggest beta 1 = 0.9, beta 2 = 0.999, ϵ = 10 e - 8

	The practice shows that ADAM is more effective than other adaptive learning methods.

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/图片2.png)
![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/图片3.png)

**In both cases, Adagrad, Adadelta, RMSProp found the right direction almost quickly and converged quite quickly, while the other methods were either slow or took a lot of detours to find it.**
**It can be seen from the figure that the adaptive learning rate method, namely ADAGRAD, ADADELTA, RMSProp, ADAM, is more suitable and has better convergence in this situation.**



## Activation function <a name="activation_function"></a>
**Simply put, an activation function is a function that is added into an artificial neural network in order to help the network learn complex patterns in the data.**
  ### Sigmoid
Sigmoid function, also known as S-type function, can map the entire real interval to (0,1) interval, so it is often used to calculate probability. It is also an activation function that is often used in traditional neural networks.

Advantages of the Sigmoid activation function: the mapping interval of the output (0,1) is monotonically continuous, which is very suitable for the output layer and is easy to differentiate.

Disadvantages of the Sigmoid activation function: it has soft saturation, that is, as the input x tends to infinity, its derivative tends to zero, which can easily cause the gradient to disappear.

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/图片4.png)

  ### Tanh
Tanh is a hyperbolic tangent function that maps the entire real interval to (-1,1), and tanh has soft saturation. Its output is centered on 0. Tanh converges faster than sigmoid, and tanh also has the problem of gradient disappearance due to the existence of soft saturation.

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/图片5.png)

  ### Relu
The output of the relu function is always 0 when x<0. Since the derivative of ReLU function is 1 when x>0, ReLU function can keep the gradient attenuating continuously when x>0. So Relu function could alleviate the problem of gradient disappearance, accelerate the convergence rate, and make the neural network have sparse expression ability, which are also the reasons why ReLU activation function can be used in deep neural network.

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/图片6.png)

  ### Softmax
The Softmax activation function is a normalized exponential function. The output of a neuron depends not only on its own input value, but also on the sum of the inputs of all other neurons that exist in the layer.

## Loss function <a name="loss_function"></a>
**Loss function is a very important content in machine learning, which measures the difference between the output value of the model and the target value. In other words, it is an important indicator to evaluate the effect of the model. The smaller the loss function is, the better the robustness of the model is.When training the model in TensorFlow, the loss function tells TensorFlow whether the predicted result is better or worse than the target result. In many cases, sample data and target data for model training are given, and the loss function is to compare the difference between the predicted value and the given target value.**


**The loss function of regression model:**

	L1 is positive then loss function (absolute value loss function)

	The L1 is positive then loss function is the absolute value of the difference between the predicted value and the target value

	L2 is positive then loss function (i.e. Euler loss function)

	The L2 is positive then loss function is the sum of squares of the difference between the predicted value and the target value

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/L1L2.png)

  ### Hinge
Hinge losses are often used in dichloric problems to evaluate vector machine algorithms, but sometimes neural network algorithms as well

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/hinge.png)

  ### Cross entropy
Cross entropy comes from information theory and is a widely used loss function in classification problems. The cross entropy describes the distance between two probability distributions. The closer the two probability distributions are, the smaller their cross entropy will be. The cross-entropy loss function is mainly applied to dichotomy problems, and its predicted value is a probability value in the range of [0,1].

![alt text](https://github.com/SLN0513/Lyft-Motion-Prediction-for-Autonomous-Vehicles-final/blob/main/crossentropy.png)


## Method <a name="Method"></a>
	If the data is sparse, the adaptive methods should be adopted, i.e., ADAGRAD, ADADELTA, RMSProp, ADAM.
	
	RMSProp, Adadelta, and Adam have similar effects in many cases.
	
	Adam added biase-correction and momentum on the basis of RMSprop.
	
	As the gradient becomes sparse, ADAM performs better than RMSProp.
	
	Overall, Adam is the best choice.

	A lot of papers use SGD, or without momentum. SGD can reach a minimum value, but it takes 
	longer time than other algorithms and can get stuck at saddle points. If faster convergence is needed, or 
	deeper and more complex neural networks are trained, an adaptive algorithm is needed.
## Modeling <a name="modeling"></a>

# 1 .pytorch_neg_multi_log_likelihood_batch   ...  
# 2. pytorch_neg_multi_log_likelihood_single   ...
# 3. define baseline model  ...
# 4.class LyftMultiModel(nn.Module):  ...
# 5.visualize_trajectory  ...


## Further discussion <a name="further_discussion"></a>

![alt text](https://github.com/xuyuan1/Lyft-Motion-Prediction-for-Autonomous-Vehicles/blob/main/rank.png)
