import os.path as path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tensorflow as tf

from tf_cmgt.eval_callbacks import standart_callbacks

def eval(steps, dist_strat = None, batch_size = 8, ckpt_path = "./ckpt", summary_path = "./log_dir", callbacks = standart_callbacks(), log_suffix = "./eval", train_log_suffix = "./train"):
    logger.info("### EVALUATION ###")
    writer_eval = tf.summary.create_file_writer(path.join(summary_path,log_suffix))
    
    if dist_strat is None:
        logger.warning("no dist_strat set using standart distribution strategy")
        logger.warning("----#### !! using MirroredStrategy with HierarchicalCopyAllReduce !! ####----")
        dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    
    logger.info("-> creating dataset")
    if callbacks.create_dataset:
        per_replica_batch_size = batch_size // dist_strat.num_replicas_in_sync
        dataset = callbacks.create_dataset(per_replica_batch_size)
    else:
        logger.error("No eval dataset, set create_dataset in the callbacks")
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
    
    with dist_strat.scope():
        
        logger.info("-> creating eval model")
        if callbacks.create_model:
            eval_model = callbacks.create_model()
        else:
            logger.error("No eval_model, set create_model in the callbacks")
            return

        logger.info("-> creating loss function")
        if callbacks.create_loss:
            eval_loss = callbacks.create_loss()
        else:
            logger.error("No eval_loss, set create_loss in the callbacks")
            return
        
        logger.info("-> init checkpointing")
        if callbacks.load_ckpt:
            ckpt, manager = callbacks.load_ckpt(path.join(ckpt_path,train_log_suffix), step, eval_model, eval_loss)
            ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                logger.info("--> Restored from {}".format(manager.latest_checkpoint))
            else:
                logger.error("No latest checkpoint, checkpoint folder has no checkpoints")
                logger.error("can not evaluate a model without loading weights from a checkpoint")
                return
        else:
            logger.error("No ckpt and manager, set create_ckpt in the callbacks or use standart_callbacks")
            logger.error("can not evaluate a model without loading weights from a checkpoint")
            return
        
    @tf.function(experimental_relax_shapes=True)
    def singel_device_eval_step(batch):
        
        logger.info("--> model input preparation")
        if callbacks.input_pre:
            inputs = callbacks.input_pre(batch)
        else:
            logger.error("No inputs, set input_pre in the callbacks or use standart_callbacks")
            return
        
        logger.info("--> model building")
        output = eval_model(inputs)
        
        logger.info("--> loss input preparation")
        if callbacks.loss_pre:
            loss_input = callbacks.loss_pre(output, batch, eval_model)
        else:
            logger.error("No loss_input, set loss_pre in the callbacks or use standart_callbacks")
            return
        
        logger.info("--> loss building")
        loss = eval_loss(loss_input)
        
        logger.info("--> evaluation calculation preparation")
        if callbacks.eval_pre:
            loss_per_batch, extra_loss = callbacks.eval_pre(loss, output, batch, eval_model, eval_loss)
        else:
            logger.error("No eval loss, set eval_pre in the callbacks or use standart_callbacks")
            return

        loss_per_input = loss_per_batch / tf.cast(batch_size, dtype=loss_per_batch.dtype)
        
        agg_loss = loss_per_input + extra_loss

        dev = tf.distribute.get_replica_context().devices
        
        
        logger.info("--> callback functions")
        for callback_step, step_callback in callbacks.every_steps.items():
            if callback_step > 0 and step % callback_step == 0:
                step_callback(dev, step, loss, output, batch, eval_model, eval_loss)
        for callback_step, step_callback in callbacks.at_step.items():
            if step == callback_step:
                step_callback(dev, step, loss, output, batch, eval_model, eval_loss)
    
    @tf.function(experimental_relax_shapes=True)
    def eval_step(batch): 
        dist_strat.run(singel_device_eval_step, args=(batch,))
    
    @tf.function
    def eval_loop():
        logger.info("-> eval loop")
        with dist_strat.scope():
            for batch in dist_dataset:
                step.assign_add(1)
                with writer_eval.as_default():
                    tf.summary.experimental.set_step(tf.cast(step,tf.int64))
                    eval_step(batch)
                    writer_eval.flush()
        
    eval_loop()
