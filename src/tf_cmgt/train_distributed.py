import os.path as path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tensorflow as tf
import numpy as np
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

def main():
    pass

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
    

def train(steps, dist_strat, batch_size = 8, learning_rate = 0.01, ckpt_path = "./ckpt", summary_path = "./log_dir", callbacks = standart_callbacks()):
    logger.info("### TRAINING ###")
    writer_train = tf.summary.create_file_writer(path.join(summary_path,"./train"))
    
    if dist_strat is None:
        #singel_dev_train.train(steps = steps, get_train_model = get_train_model, dataset = dataset, batch_size = batch_size, learning_rate = learning_rate, step_callbacks = step_callbacks)
        logger.error("changed interface so None is not supported use a starategie")
        return
    
    logger.info("-> creating dataset")
    if callbacks.create_dataset:
        per_replica_batch_size = batch_size // dist_strat.num_replicas_in_sync
        dataset = callbacks.create_dataset(per_replica_batch_size)
    else:
        logger.error("No train dataset, set create_train_dataset in the callbacks")
        return
    
    dataset = dataset.repeat(-1)
    
    def make_dist_dataset(dataset, steps = None):
        def dataset_fn(input_context):
            per_replica_batch_size = input_context.get_per_replica_batch_size(batch_size)
            d = callbacks.make_batches(dataset, per_replica_batch_size)
            if steps is None:
                return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id).take(1)
            else:
                return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id).take(steps)
        return dataset_fn
    
    logger.info("-> creating batches")
    if callbacks.make_batches:
        dist_dataset = dist_strat.experimental_distribute_datasets_from_function(make_dist_dataset(dataset, steps))
    else:
        logger.error("No batching, set make_batches in the callbacks or use standart_callbacks")
        return
    
    step = tf.Variable(0, trainable=False, dtype = tf.int32)
    lr = tf.Variable(learning_rate, trainable=False, dtype = tf.float32)
    
    with dist_strat.scope():
        
        logger.info("-> creating optimizer")
        if callbacks.create_opt:
            optimizer = callbacks.create_opt(lr)
        else:
            logger.error("No optimizer, set create_opt in the callbacks or use standart_callbacks")
            return
        
        logger.info("-> creating training model")
        if callbacks.create_model:
            train_model = callbacks.create_model()
        else:
            logger.error("No train_model, set create_model in the callbacks")
            return

        logger.info("-> creating loss function")
        if callbacks.create_loss:
            train_loss = callbacks.create_loss()
        else:
            logger.error("No train_loss, set create_loss in the callbacks")
            return
        
        logger.info("-> init checkpointing")
        if callbacks.create_ckpt:
            ckpt, manager = callbacks.create_ckpt(path.join(ckpt_path,"./train"), step, lr, optimizer, train_model, train_loss)
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                logger.info("--> Restored from {}".format(manager.latest_checkpoint))
            else:
                logger.warn("--> Initializing from scratch.")
        else:
            logger.warn("No ckpt and manager, set create_ckpt in the callbacks or use standart_callbacks")
            ckpt = manager = None
            logger.warn("----#### !! continueing without checkpointing !! ####----")

    @tf.function(experimental_relax_shapes=True)
    def singel_device_train_step(batch):
        
        logger.info("--> model input preparation")
        if callbacks.input_pre:
            inputs = callbacks.input_pre(batch, lr)
        else:
            logger.error("No inputs, set input_pre in the callbacks or use standart_callbacks")
            return
        
        logger.info("--> model building")
        output = train_model(inputs)
        
        logger.info("--> loss input preparation")
        if callbacks.loss_pre:
            loss_input = callbacks.loss_pre(output, batch, train_model, lr)
        else:
            logger.error("No loss_input, set loss_pre in the callbacks or use standart_callbacks")
            return
        
        logger.info("--> loss building")
        loss = train_loss(loss_input)
        
        logger.info("--> gradient calculation preparation")
        if callbacks.grad_pre:
            loss_per_batch, extra_loss, trainable_vars = callbacks.grad_pre(loss, output, batch, train_model, train_loss, lr)
        else:
            logger.error("No loss and trainable_vars, set grad_pre in the callbacks or use standart_callbacks")
            return

        loss_per_input = loss_per_batch / tf.cast(batch_size, dtype=loss_per_batch.dtype)
        
        agg_loss = loss_per_input + extra_loss

        dev = tf.distribute.get_replica_context().devices
        
        logger.info(f"--> tracing gradients on {dev}")
        gradients = optimizer.get_gradients(agg_loss, trainable_vars)

        logger.info("--> gradient cleaning and update preparation")
        if callbacks.update_pre:
            update_gradients, display_gradients = callbacks.update_pre(gradients, loss, output, batch, train_model, train_loss, lr, optimizer)
        else:
            logger.error("No gradients, set update_pre in the callbacks or use standart_callbacks")
            return
        
        to_optimize = zip(update_gradients, trainable_vars)
        
        logger.info("--> update with gradients")
        optimizer.apply_gradients(to_optimize)
        
        logger.info("--> callback functions")
        for callback_step, step_callback in callbacks.every_steps.items():
            if callback_step > 0 and step % callback_step == 0:
                step_callback(dev, step, update_gradients, loss, output, batch, train_model, train_loss, lr, optimizer,  display_gradients, gradients, ckpt, manager)
        for callback_step, step_callback in callbacks.at_step.items():
            if step == callback_step:
                step_callback(dev, step, update_gradients, loss, output, batch, train_model, train_loss, lr, optimizer,  display_gradients, gradients, ckpt, manager)
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(batch): 
        dist_strat.run(singel_device_train_step, args=(batch,))
    
    @tf.function
    def train_loop():
        logger.info("-> training loop")
        with dist_strat.scope():
            for batch in dist_dataset:
                step.assign_add(1)
                with writer_train.as_default():
                    tf.summary.experimental.set_step(tf.cast(step,tf.int64))
                    train_step(batch)
                    writer_train.flush()
        
    train_loop()
