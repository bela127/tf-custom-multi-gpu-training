import tensorflow as tf
from addict import Dict

def standart_callbacks():
    callbacks = Dict()
    callbacks.every_steps[1] = print_step
    callbacks.every_steps[10] = print_loss
    callbacks.at_step = {}
    
    callbacks.make_batches = make_batches
    
    callbacks.create_dataset = None
        
    callbacks.create_model = None
    
    callbacks.create_loss = None
    
    load_ckpt = None
        
    callbacks.input_pre = input_pre
    callbacks.loss_pre = loss_pre
    callbacks.eval_pre = eval_pre
    return callbacks

def print_step(dev, step, loss, output, batch, eval_model, eval_loss):
    tf.print("Step:", step)
    
def print_loss(dev, step, loss, output, batch, eval_model, eval_loss):
    tf.print("Loss:", loss)
    tf.summary.scalar(f"loss", loss)

def make_batches(dataset, per_replica_batch_size):
    return dataset

def input_pre(batch):
    inputs, gt = batch
    return inputs

def loss_pre(output, batch, train_model):
    inputs, gt = batch
    return output, gt

def eval_pre(loss, output, batch, eval_model, eval_loss):
    loss_per_batch = loss
    extra_loss_list = eval_model.losses
    extra_loss = tf.reduce_sum(extra_loss_list)
    return loss_per_batch, extra_loss
    