import numpy as numpy
from skimage import io, transform
import sys

from caffe2.proto import caffe2_pb2
from pycaffe2 import core, workspace

net = caffe2_pb2.NetDef()
net.ParseFromString(open('multibox_net.pb').read())
#with open('multibox_net.pb', 'w') as fid:
#    fid.write(net.SerializeToString())

import glob
parts = glob.glob("*tensors.pb.part*")
parts.sort()
tensors = caffe2_pb2.TensorProtos()
tensors.ParseFromString(''.join(open(f).read() for f in parts))

DEVICE_OPTS = caffe2_pb2.DeviceOption()
DEVICE_OPTS.device_type = caffe2_pb2.CUDA
DEVICE_OPTS.cuda_gpu_id = 0

workspace.SwitchWorkspace('default')
for param in tensors.protos:
	workspace.FeedBlob(param.name, param, DEVICE_OPTS)

workspace.CreateBlob('input')
net.device_option.CopyFrom(DEVICE_OPTS)
workspace.CreateNet(net)

LOCATION_PRIOR = np.loadtxt('ipriors800.txt')

def RunOnImage(image, location_prior):
    img = io.imread(image)
    img_ = transform.resize(img, (224, 224))
    img_ = img_.reshape((1, 224, 224, 3)).astype(np.float32) - 0.5
    workspace.FeedBlob("input", img_, DEVICE_OPTION)
    workspace.RunNet("multibox")
    location = workspace.FetchBlob("imagenet_location_projection").flatten(),
    # Recover the original locations
    location = location * location_prior[:,0] + location_prior[:,1]
    location = location.reshape((800, 4))
    confidence = workspace.FetchBlob("imagenet_confidence_projection").flatten()
    return location, confidence

location, confidence = RunOnImage('not-penguin.jpg', LOCATION_PRIOR)
import pickle
with open('location.pkl', 'wb') as f:
	pickle.dump(location, f)
with open('confidence.pkl', 'wb') as f:
	pickle.dump(confidence, f)
