# Overall structure referenced form:
# https://stackoverflow.com/questions/43172922/tensorflow-restore-model-and-predict


# Model restored using ckpt files

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from scipy import misc

import os
import os.path
import numpy as np
#import opencv as cv2


# model dir
model_dir = '/data/log_dir/180520_29363_resnet152_r'
checkpoint_path = model_dir + '/model.ckpt-40151'
# image_dir
image_path = '/home/ubuntu/results/prediction/test_images/'
# output dir
output_path ='/home/ubuntu/test_results'

# List all tensor
#print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True, all_tensor_names=False)


config = tf.ConfigProto(allow_soft_placement = True)

tf.reset_default_graph()

with tf.Session(config= config) as sess:
    model_saver = tf.train.import_meta_graph(model_dir + '/model.ckpt-40151.meta')
    model_saver.restore(sess, model_dir + '/model.ckpt-40151')

    print("Model restored with success")

# Retrieve list of operations in the graph
#    for i in sess.graph.get_operations():
#        print(i.name)


#    images = sess.graph.get_operation_by_name("images")

    images = tf.placeholder(tf.float32,shape=[32, 224, 224, 3])
    classes = sess.graph.get_operation_by_name("ArgMax")
    probabilities = sess.graph.get_operation_by_name("resnet_model/final_dense")
    print("output print_tensors_in_checkpoint_file restored")


    temp = []
    for file in os.listdir(image_path):
        if file.endswith(".jpg"):
            img = misc.imread(image_path + file)
            img = img.astype('Float32')
            img = np.resize(img, (224, 224, 3))
            temp.append(img)

    predictions = sess.run([classes, probabilities], feed_dict={images: temp})
    print(predictions)
   

#     cv2.imwrite(output_path, prediction * 255)


#    x = tf.get_collection("placeholder")[0]  
#    output = tf.get_collection('softmax_tensor:0') 

#    print(x)
#    print(output)

#    print("Model restored")
#    print("Initialized")


#    inputData = []
    # import data


#    inputData.append(process_data(patient, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT))


    # run prediction on input Data
#    prediction = sess.run(output, feed_dict= {x: inputData})
#    print(prediciton)
