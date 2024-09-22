import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.test.is_built_with_cuda())
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Configure TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)