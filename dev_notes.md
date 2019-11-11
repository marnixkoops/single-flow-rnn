# `TensorFlow 2.0 Embedding GRU-RNN`

## DEV NOTES

### TO IMPLEMENT

* [ ] Training from previous weights? https://www.tensorflow.org/guide/keras/save_and_serialize
* [x] Build experimental framework with tracking
* [x] Set up GCP VM to train on GPU with TensorFlow 1
* [x] Set up GCP VM to train on GPU with TensorFlow 2
* [x] Filter invalid sequences
* [x] Dynamic window length for products per sequence for train-val pairs
* [x] Dynamic look back window (history) to base predictions on
* [x] Double dataset option
* [x] Train with batches
* [x] Train with validation set
* [x] Train with early stopping
* [x] Train with Categorical Cross-Entropy loss for encoded product ID's directly without OHE

### TO EXPLORE

##### INPUT DATA

* Remove sessions of only duplicate items
* Extend used input data
  * [ ] Customer demographics/information
  * [ ] Extra Session data
    * Purchase central in sequence?
    * Add to cart
    * Wishlist
    * Time spent on page?
    * Read reviews?
  * [ ] Seasonality patterns? (Gaming/tv's at night, whitegoods during day or morning more popular?)

##### DATA PROCESSING

* [ ] Hashing products instead of embedding
* [ ] Data type to reduce memory footprint (float16 instead of float32?)
  * Post-Training Quantization?
* [ ] Size of window (sequence length) when generating input-validation pad_sequences
* [ ] Size of prediction look back window, dynamic or cut-off?
* [ ] Minimum items required per sequence (>2?)
* [ ] Train with cross-validation to determine epochs, or use as ensemble to majority vote

##### MODEL ARCHITECTURE

* [ ] Train many to many (output 5?)
  * Complications: Many short sequences, zero masking output sequence for training?
* [ ] Attention layer (requires Encoder-Decoder architecture?)
* [ ] Bi-Directional RNN


##### MODEL PARAMETERS

* [ ] Stateful network
* [x] Embedding dimensions
* [x] Hidden units in GRU cell


##### TRAINING

* [ ] Input dimension = VOCAB_SIZE + 1 due to masking zero's?
* [ ] Learning rate (not that important when using adaptive optimizers like Adam or Nadam)
* [ ] Batch size (large batch sizes slow down training)
* [ ] Dropout (Or recurrent dropout), (having training dropout seems to improve performance for now)
* [ ] Test and tune optimizers (Adam and Nadam seem most promising)
  * Adam = RMSprop + Momentum. Combines the good properties of AdaDelta and RMSProp and hence tends to do better for most of the problems
* [ ] Gradient clipping
* [ ] Float16 instead of Float32?

--

## THEORY NOTES

`Why Recommendations?`

https://venturebeat.com/2017/06/14/airbnb-vp-talks-about-ais-profound-impact-on-profits/

`Why NNs for Recommendations?`

* One of the most attractive properties of neural architectures is that they are (1) end-to-end differentiable and (2) provide suitable inductive biases catered to the input data type. As such, if there is an inherent structure that the model can exploit, then deep neural networks ought to be useful. For instance, CNNs and RNNs have long exploited the intrinsic structure in vision (and/or human language). Similarly, the sequential structure of session or click-logs are highly suitable for the inductive biases provided by recurrent/convolutional models.
* Moreover, deep neural networks are also composite in the sense that multiple neural building blocks can be composed into a single (gigantic) differentiable function and trained end-to-end. key advantage here is when dealing with content-based recommendation. is is inevitable when modeling users/items on the web, where multi-modal data is commonplace. For instance, when dealing with textual data (reviews [202], tweets [44] etc.), image data (social posts, product images), CNNs/RNNs become indispensable neural building blocks. Here, the traditional alternative (designing modality-specific features etc.) becomes significantly less attractive and consequently, the recommender system cannot take advantage of joint (end-to-end) representation learning.
* All in all, the capabilities of deep learning in this aspect can be regarded as paradigm-shifting and the ability to represent images, text and interactions in a unified joint framework [197] is not possible without these recent advances.

`GRU:`

* Gated Recurrent Unit. GRU (Cho et al.) alternative memory cell design to LSTM. Often reaches equal or better performance while using fewer parameters and being faster to compute. Related to LSTM having 3 gates (input, remember, forget) while GRU combines  remember and forget into a single update gate resulting in only 2 gates.

`Gradient Descent:`

* Gradient descent is used to reduce the cost function of the ML problem iteratively. We want to find the minimum (lowest) point. This corresponds to lowest loss which should reflect our real life metric of interest. Loss function != real life metric. For example, MAP can not be used as loss function because it is not differentiable. The loss function needs to be differentiable because it is highly complex and non convers. Hence, we need to optimize it by taking steps in the right direction. To find which direction we need to step into the gradient is used. The gradient of a function indicate direction of increase. Hence, we update parameters in the negative gradient direction to minimize the loss.
* The loss function tells the optimizer if it is moving in the right direction (lower) There are a couple different flavors of optimizers:
  * Momentum: Like a ball rolling down a hill, it will gain momentum as it rolls down, accelerating. For updating the weights it takes the gradient of the current step as well as the gradient of the previous time steps. This corresponds to taking larger steps when the gradient is high (decreasing fast) and hopefully resulting in faster convergence.
  * Nesterov Accelerated Gradient (NAG): Similar to momentum is like a ball rolling down the hill but a bit smarter. It knows when to slow down before the gradient increases again (overshooting the minimum). This is done be computing the gradient not with respect to the current timestep but with respect to the future step. This means we look ahead before making the step.
  * Tuning the learning rate in Momentum and NAG is expensive. Adagrad can be used as adaptive learning rate method. It adapts the learning rate to the parameters, larger updates for infrequent and smaller updates for freqeuent. It is well suited when we have sparse data as in large scale neural networks. For example, GloVe word embedding uses Adagrad where infrequent words required a greater update and frequent words require smaller updates. This might make sense for our product case as some products occur much more frequently compared to others. Also, Adagrad eliminates the need to manually tune the learning rate.
  * Adadelta, RMSProp and Adam, try to resolve Adagrads isue of radically diminishing learning rates. (This sounds similar to vanishing gradient problem in RNN's?)
  * Adadelta restricts the window of past accumulated gradient to a fixed size.
  * RMSProp (Root Mean Square Propogation) uses a moving average of the squared gradient to combat diminishing learning rates. In RMSProp learning rate gets adjusted automatically and it chooses a different learning rate for each parameter.
  * Adam (Adaptive Moment Estimation) calculates the individual adaptive learning rate for each parameter from estimates of first and second moments of the gradients. It is a combination of Adagrad and RMSProp. Adam implements the exponential moving average of the gradients to scale the learning rate instead of a simple average as in Adagrad. It keeps an exponentially decaying average of past gradients. Adam is computationally efficient and has very little memory requirement. Adam optimizer is currently the most popular choice of optimizer.
  * Nadam (Nesterov-accelerated Adaptive Moment Estimation) combines NAG and Adam. Is emplyed for noisy gradients or for gradients with high curvatures. The learning process is accelerated by summing up the exponential decay of the moving averages for the previous and current gradient

`Dropout in RNN`:

* Each hidden unit in a neural network trained with dropout must learn to work with a randomly chosen sample of other units. This should make each hidden unit more robust and drive it towards creating useful features on its own without relying on other hidden units to correct its mistakes.
* Regular dropout is applied on input/output samples in a normal network
* In a RNN we can also have recurrent dropout, in this case the dropout is applied on the recurrent connections in the network.

`Attention`:

* Give machine ability to focus more on certain parts, just like human attention.
* Attention is one component of a network’s architecture, and is in charge of managing and quantifying the interdependence:
 * Between the input and output elements (General Attention)
 * Within the input elements (Self-Attention)
