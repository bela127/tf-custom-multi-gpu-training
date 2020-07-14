# tf-custom-multi-gpu-training
Enables custom multi gpu training with a predefined trainings loop which is highly customizable through callbacks.
The standart Keras training loop is good for training simple models, but with complex models a custom training loop must be used to utilize all functionality. This leads to a lot of boilerplate code, especialy if checkpointing and multi GPU training should be supported.
This mini framework enabels full custom training loops with checkpointing and multi GPU training without writing the same boilerplate code again and again.

## How To
All parts of the training process can be modified through callback functions, standard callback functions are already implemented, so you only have to write the parts where your doing custom stuff.

