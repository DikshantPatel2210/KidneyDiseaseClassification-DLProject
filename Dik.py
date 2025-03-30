import tensorflow as tf

# Check if TensorFlow is using GPU
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU")
else:
    print("TensorFlow is not using the GPU")
