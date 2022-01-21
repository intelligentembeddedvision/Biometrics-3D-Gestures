import tensorflow as tf

# Create pbtxt file out of frozen graph
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants
from tensorflow.python.tools import optimize_for_inference_lib


def graphdef_to_pbtxt(filename):
    with open(filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with open('frozen_models/frozen_graph.pbtxt', 'w') as fp:
        fp.write(str(graph_def))


graphdef_to_pbtxt('frozen_models/frozen_graph.pb')
