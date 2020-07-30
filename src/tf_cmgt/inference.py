import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tensorflow as tf

from tf_cmgt.infer_callbacks import standart_callbacks

class InferenceModel(tf.keras.Model):
    
    def __init__(self, ckpt_path, callbacks = standart_callbacks()):
        super().__init__()
        self.callbacks = callbacks
        self.ckpt_path = ckpt_path
        
        logger.info("-> creating eval model")
        if callbacks.create_model:
            self.model = callbacks.create_model()
        else:
            logger.error("No eval_model, set create_model in the callbacks")
            return
        
        self.load_weights()
        self.warmup()
    
    def warmup(self):
        logger.info("-> warmup model")
        
        if self.callbacks.warmup:
            batch = self.callbacks.warmup()
            
            logger.info("--> model input preparation")
            if self.callbacks.input_pre:
                inputs = self.callbacks.input_pre(batch)
            else:
                logger.error("No inputs, set input_pre in the callbacks or use standart_callbacks")
                return
            
            y = self.model(inputs)
            
            logger.debug("model outputs:",y)
            logger.info("-> model warmup finish")
        else:
            logger.warn("no warmup possible, set warmup callback in the callbacks")
            logger.warn("----#### !! without warmup the first inference will take a long time !! ####----")

    def load_weights(self):
        logger.info("-> init checkpointing")
        if self.callbacks.load_ckpt:
            ckpt, manager = self.callbacks.load_ckpt(self.ckpt_path, self.model)
            ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                logger.info("--> Restored from {}".format(manager.latest_checkpoint))
            else:
                logger.error("No latest checkpoint, checkpoint folder has no checkpoints")
                logger.error("can not do inference with a model without loading weights from a checkpoint")
                return
        else:
            logger.error("No ckpt and manager, set create_ckpt in the callbacks or use standart_callbacks")
            logger.error("can not do inference with a model without loading weights from a checkpoint")
            return
        
    def call(self, inputs, training=False):
        batch = inputs
        logger.info("--> model input preparation")
        if self.callbacks.input_pre:
            inputs = self.callbacks.input_pre(batch)
        else:
            logger.error("No inputs, set input_pre in the callbacks or use standart_callbacks")
            return
        outputs = self.model(inputs, training=training)
        return outputs