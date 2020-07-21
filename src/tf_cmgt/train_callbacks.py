import tensorflow as tf
from addict import Dict

def standart_callbacks():
    callbacks = Dict()
    callbacks.every_steps[1] = print_step
    callbacks.every_steps[10] = print_loss
    callbacks.at_step = {}
    
    callbacks.make_batches = make_batches
    
    callbacks.create_dataset = None
    
    callbacks.create_ckpt = create_ckpt
    
    callbacks.create_model = None
    
    callbacks.create_loss = None
    
    callbacks.create_opt = create_opt
    
    callbacks.input_pre = input_pre
    callbacks.loss_pre = loss_pre
    callbacks.grad_pre = grad_pre
    callbacks.update_pre = update_pre
    return callbacks

def print_step(dev, step, update_gradients, loss, output, batch, train_model, train_loss, lr, optimizer,  display_gradients, gradients, ckpt, manager):
    tf.print("Step:", step)
    
def print_loss(dev, step, update_gradients, loss, output, batch, train_model, train_loss, lr, optimizer,  display_gradients, gradients, ckpt, manager):
    tf.print("Loss:", loss)
    tf.summary.scalar(f"loss", loss)

def make_batches(dataset, per_replica_batch_size):
    return dataset

def create_opt(lr):
    optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, epsilon=0.001)
    return optimizer

def create_ckpt(ckpt_path, step, lr, optimizer, train_model, train_loss):
    nets = {"train_model":train_model,
            "train_loss":train_loss,
           }
    ckpt = tf.train.Checkpoint(step = step, lr = lr, optimizer = optimizer, **nets)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=50)
    return ckpt, manager

def input_pre(batch, lr):
    inputs, gt = batch
    return inputs

def loss_pre(output, batch, train_model, lr):
    inputs, gt = batch
    return output, gt

def grad_pre(loss, output, batch, train_model, train_loss, lr):
    loss_per_batch = loss
    extra_loss_list = train_model.losses
    extra_loss = tf.reduce_sum(extra_loss_list)
    trainable_vars = train_model.trainable_variables
    return loss_per_batch, extra_loss, trainable_vars

def update_pre(gradients, loss, output, batch, train_model, train_loss, lr, optimizer):
    non_nan_gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]
    capped_gradients, norm = tf.clip_by_global_norm(non_nan_gradients, 10.)
    update_gradients = capped_gradients
    display_gradients = non_nan_gradients
    return update_gradients, display_gradients
    