# RNN DEV NOTES

http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

### TO IMPLEMENT

* Change to
* Padding + masking zeros
* Specify sequence_length for training in dynamic_rnn


### FURTHER OPTIMIZATION

* Optimizer (RMSProp alternatives)
* Gradient clipping (currently 1.0, was 4.0 in original implementation)
* Size of window when generating input-validation pad_sequences
* Minimum items required per sequence (>2?)
