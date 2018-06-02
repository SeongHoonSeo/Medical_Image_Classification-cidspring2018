# Overall structure referenced form:
# https://stackoverflow.com/questions/43172922/tensorflow-restore-model-and-predict


# Model restored using ckpt files

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# model dir
model_dir = '/data/log_dir/180520_29363_resnet152_r'
checkpoint_path = model_dir + '/model.ckpt-40151'

# List all tensor
#print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True, all_tensor_names=False)


config = tf.ConfigProto(allow_soft_placement = True)

#tf.reset_default_graph()

with tf.Session(config= config) as sess:
    model_saver = tf.train.import_meta_graph(model_dir + '/model.ckpt-40151.meta')
    model_saver.restore(sess, model_dir + '/model.ckpt-40151')

    for i in sess.graph.get_operations():
        print(i.name)

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
