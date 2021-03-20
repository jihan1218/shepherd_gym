"""
The PyTorch2Keras converter interface
"""
from tensorflow import keras
from onnx2keras import onnx_to_keras
import torch
import onnx
from onnx import optimizer
import io
import logging
from shepherd_gym.models.imitation_utils import DemoDataset, RandomSampler, \
        Policy, Trainer

from torch.autograd import Variable

def pytorch_to_keras(
    model, args, input_shapes=None,
    change_ordering=False, verbose=False, name_policy=None,
    use_optimizer=False, do_constant_folding=False
):
    """
    By given PyTorch model convert layers with ONNX.
    Args:
        model: pytorch model
        args: pytorch model arguments
        input_shapes: keras input shapes (using for each InputLayer)
        change_ordering: change CHW to HWC
        verbose: verbose output
        name_policy: use short names, use random-suffix or keep original names for keras layers
    Returns:
        model: created keras model.
    """
    logger = logging.getLogger('pytorch2keras')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger.info('Converter is called.')

    if name_policy:
        logger.warning('Name policy isn\'t supported now.')

    if input_shapes:
        logger.warning('Custom shapes isn\'t supported now.')

    if input_shapes and not isinstance(input_shapes, list):
        input_shapes = [input_shapes]

    if not isinstance(args, list):
        args = [args]

    args = tuple(args)

    dummy_output = model(*args)

    if isinstance(dummy_output, torch.autograd.Variable):
        dummy_output = [dummy_output]

    input_names = ['input_{0}'.format(i) for i in range(len(args))]
    output_names = ['output_{0}'.format(i) for i in range(len(dummy_output))]

    logger.debug('Input_names:')
    logger.debug(input_names)

    logger.debug('Output_names:')
    logger.debug(output_names)

    stream = io.BytesIO()
    torch.onnx.export(model, args, stream, do_constant_folding=do_constant_folding, verbose=verbose, input_names=input_names, output_names=output_names)

    stream.seek(0)
    onnx_model = onnx.load(stream)
    if use_optimizer:
        if use_optimizer is True:
            optimizer2run = optimizer.get_available_passes()
        else:
            use_optimizer = set(use_optimizer)
            optimizer2run = [x for x in optimizer.get_available_passes() if x in use_optimizer]
        logger.info("Running optimizer:\n%s", "\n".join(optimizer2run))
        onnx_model = optimizer.optimize(onnx_model, optimizer2run)

    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names,
                            input_shapes=input_shapes, name_policy=name_policy,
                            verbose=verbose, change_ordering=change_ordering)

    return k_model


p = Policy()
m_dict = torch.load('model.pt', map_location=torch.device('cpu'))
# print('m_dict: ', m_dict)
p.load_state_dict(m_dict)
print('p: ', p)

# input_np  = [137.5930,  -7.4188, 139.3250, -49.9179,  -2.5092,   9.0143,  97.5497, 2.9870,  42.5344, 141.0627]
input_np  = [[137.5930,  -7.4188, 139.3250, -49.9179,  -2.5092,   9.0143,  97.5497, 2.9870,  42.5344, 141.0627]]

input_var = Variable(torch.FloatTensor(input_np))
# tm = pytorch_to_keras(p, input_var, [(1, 10)])
tm = pytorch_to_keras(p, input_var)
print(tm.summary())

#### IMPORTANT ####
# There are two model formats used by the python tensorflow API
#
#
#  (1) SavedModel: This is the default format in which TensorFlow models are saved.
#
#  More info:
# "New in TensoFlow 2.4 The argument save_traces has been added to model.save, which allows you to toggle SavedModel function tracing. Functions are saved to allow the Keras to re-load custom objects without the original class definitons, so when save_traces=False, all custom objects must have defined get_config/from_config methods. When loading, the custom objects must be passed to the custom_objects argument. save_traces=False reduces the disk space used by the SavedModel and saving time."
#
#  The SavedModel format is documented here -
#  https://www.tensorflow.org/guide/saved_model
#  
#
#  (2) Keras model: Keras models are generally saved as an HDF5 file.
#
#  More info: 
#  "Keras also supports saving a single HDF5 file containing the model's architecture, weights values, and compile() information. It is a light-weight alternative to SavedModel."
#
#  More information about saving Keras models can be found here -
#  https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state


save_dir = 'tf_model'
# SavedModel format:
tm.save(save_dir) # creates a folder named tf_model, containing the following: the model architecture, and training configuration (including the optimizer, losses, and metrics) are stored in saved_model.pb. The weights are saved in the variables/ directory

# Keras model format:
tm.save('tf_model_keras.h5', save_format='h5') # saves directly to file



# Now you can load either of these model formats with the python tensorflow API

# SavedModel
reconstructed_model_saved_model = keras.models.load_model('tf_model')

# Keras model
reconstructed_model_keras = keras.models.load_model('tf_model_keras.h5')


# Read more about loading saved models here:
# https://www.tensorflow.org/guide/keras/save_and_serialize


#### these lines are for keras to tfjs conversion, we're not concrened with this atm.
# import tensorflowjs as tfjs
# tfjs.converters.save_keras_model(tm, 'tfjs_target')