import tensorflow as tf
from addict import Dict

def standart_callbacks():
    callbacks = Dict()
                        
    callbacks.load_ckpt = None
    
    callbacks.warmup = None
        
    callbacks.input_pre = input_pre
    return callbacks

def input_pre(batch):
    inputs = batch
    return inputs
    