# tf-custom-multi-gpu-training
Enables custom multi gpu training with a predefined trainings loop which is highly customizable through callbacks.
The standart Keras training loop is good for training simple models, but with complex models a custom training loop must be used to utilize all functionality. This leads to a lot of boilerplate code, especialy if checkpointing and multi GPU training should be supported.
This mini framework enabels full custom training loops with checkpointing and multi GPU training without writing the same boilerplate code again and again.

## How To
All parts of the training process can be modified through callback functions, standard callback functions are already implemented, so you only have to write the parts where your doing custom stuff.

Following callbacks are predefined:

### every_steps
A **every_steps[N]** callback is called every N steps.
Can be used for checkpointing, logging, summary, ...

It has the signature:
``` python
every_steps(dev, step, update_gradients, loss, output, batch, train_model, train_loss, lr, optimizer,  display_gradients, gradients, ckpt, manager)
```

Two every_steps callbacks are predefined:

**callbacks.every_steps[1] = print_step**
Is called every step and prints the current step to console

**callbacks.every_steps[10] = print_loss**
Is called every 10 steps and prints the loss to console and logs it with a scalar-summary

### at_step
A at_step[N] callback is called onece after N steps, then never again.
Can be used for network initialisation and freezing.

It has the signature:
``` python
at_step(dev, step, update_gradients, loss, output, batch, train_model, train_loss, lr, optimizer,  display_gradients, gradients, ckpt, manager)
```

No at_step callback is predefined.

### create_dataset
Must be defined, creates the dataset for training.

It has the signature:
``` python
dataset = create_dataset(per_replica_batch_size)
```

Should create a dataset with the acording per_replica_batch_size. If the make_batches callback is used to create custom batches it can return a dataset without batch-dimention.

### make_batches
Creates a dataset with custom batching routine.
One element of the returned dataset is used as a batch.

It has the signature:
``` python
dataset = make_batches(dataset, per_replica_batch_size)
```

The standart callback for this just returns the unchanged dataset.

``` python
def make_batches(dataset, per_replica_batch_size):
    return dataset
```
### create_model
Must be defined, creates the model for training.

It has the signature:
``` python
train_model = create_model()
```

It should return a tensorflow op (train_model) with a input signature:
``` python
output = train_model(inputs)
```

### create_loss
Must be defined, creates the loss for training.

It has the signature:
``` python
train_loss = create_loss()
```

It should return a tensorflow op (train_loss) with a input signature:
``` python
loss = train_loss(inputs)
```

### create_opt
Creates the optimizer to use in the training process.

It has the signature:
``` python
optimizer = create_opt(lr)
```

The standart callback creates a Nadam optimizer.

``` python
def create_opt(lr):
    optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, epsilon=0.001)
    return optimizer
```

### create_ckpt
The create_ckpt callback specifies which parameters should be included in a checkpoint and how checkpoints should be saved.

It has the signature:
``` python
ckpt, manager = create_ckpt(ckpt_path, step, lr, optimizer, train_model, train_loss)
```

The standart callback for this saves every parameter in a chekpoint, and keeps 50 checkpoints.

``` python
def create_ckpt(ckpt_path, step, lr, optimizer, train_model, train_loss):
    nets = {"train_model":train_model,
            "train_loss":train_loss,
           }
    ckpt = tf.train.Checkpoint(step = step, lr = lr, optimizer = optimizer, **nets)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=50)
    return ckpt, manager
```

### input_pre
Defines which part of the batch should be used as model input.

It has the signature:
``` python
inputs = input_pre(batch, lr)
```

The train_model is than called with:
``` python
output = train_model(inputs)
```

The standart callback expects a batch with two elements, the first element is used as model input.

``` python
def input_pre(batch, lr):
    inputs, gt = batch
    return inputs
```

### loss_pre
Defines which part of the batch and model output should be used as loss input.

It has the signature:
``` python
loss_input = loss_pre(output, batch, train_model, lr)
```

The train_loss is than called with:
``` python
loss = train_loss(loss_input)
```

The standart callback expects a batch with two elements and a singel model output, the second element in the batch is used as GT. The model output is used as prediction. Both get packed and pased to the train_loss as singel argument.

``` python
def loss_pre(output, batch, train_model, lr):
    inputs, gt = batch
    return output, gt
```

### grad_pre
Defines the loss terms for gradient calculation and the trainable variables.

It has the signature:
``` python
loss_per_batch, extra_loss, trainable_vars = grad_pre(loss, output, batch, train_model, train_loss, lr)
```

**loss_per_batch** is the sum of loss over a whole batch.
**extra_loss** is the batch independent loss, like regularisation terms.
**trainable_vars** is a list of all trainable variables.

The standart callback takes the loss as given by train_loss and collects regularisation terms with the train_model.losses api.
As trainable variables all variables from the train_model are used over the train_model.trainable_variables api.

``` python
def grad_pre(loss, output, batch, train_model, train_loss, lr):
    loss_per_batch = loss
    extra_loss_list = train_model.losses
    extra_loss = tf.reduce_sum(extra_loss_list)
    trainable_vars = train_model.trainable_variables
    return loss_per_batch, extra_loss, trainable_vars
```

### update_pre
Before gradient update the gradients can pe cliped or normed.

It has the signature:
``` python
update_gradients, display_gradients = update_pre(gradients, loss, output, batch, train_model, train_loss, lr, optimizer)
```

The update_gradients are then used to update the variables with the optimizer:
``` python
to_optimize = zip(update_gradients, trainable_vars)
optimizer.apply_gradients(to_optimize)
```

The standart callback replaces nan gradients with 0 gradients and uses clip_by_global_norm to avoid to big updates.

``` python
def update_pre(gradients, loss, output, batch, train_model, train_loss, lr, optimizer):
    non_nan_gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]
    capped_gradients, norm = tf.clip_by_global_norm(non_nan_gradients, 10.)
    update_gradients = capped_gradients
    display_gradients = non_nan_gradients
    return update_gradients, display_gradients
```
