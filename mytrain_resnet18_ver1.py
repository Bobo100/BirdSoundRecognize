# encoding:utf-8

# python3執行 3.7.10
# 完成版 => input shape更具狀況更改
from cProfile import label
from typing_extensions import runtime
from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from keras.models import Model, load_model
from keras import regularizers
from keras.utils.vis_utils import plot_model

from keras import backend as K

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

import math

import pickle

from matplotlib import pyplot as plt

import model_details

from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

model_details.use_gpu()

####################### BUILDING MODEL ########################
def residual_block_test(net_in, filters, kernel_size, stride=1, num_groups=1, preactivated=True):
    net_pre = BatchNormalization()(net_in)
    net_pre = Activation('relu')(net_pre)

    # Preactivated shortcut?
    if preactivated:
        net_sc = net_pre
    else:
        net_sc = net_in

    MAX_POOLING = False
    if MAX_POOLING:
        s = 1
    else:
        s = stride

    # First Convolution (alwys has preactivated input)   
    net = Conv2D(filters, 
                kernel_size,
                padding='same',
                strides=s,
                groups=num_groups,
                kernel_initializer=keras.initializers.HeNormal(),
                activation='relu'
                )(net_pre)
    net = BatchNormalization()(net)

    # # Optional pooling layer MAX_POOLING = false 所以不執行
    # if cfg.MAX_POOLING and stride > 1:
    #     net = l.MaxPool2DLayer(net, pool_size=stride)

    # # 也都不執行 DROPOUT_TYPE = random, DROUPOUT = 0.0
    # # Dropout Layer (we support different types of dropout)
    # if cfg.DROPOUT_TYPE == 'channels' and cfg.DROPOUT > 0.0:
    #     net = l.dropout_channels(net, p=cfg.DROPOUT)
    # elif cfg.DROPOUT_TYPE == 'location' and cfg.DROPOUT > 0.0:
    #     net = l.dropout_location(net, p=cfg.DROPOUT)
    # elif cfg.DROPOUT > 0.0:
    #     net = l.DropoutLayer(net, p=cfg.DROPOUT)

    # Second Convolution
    net = Conv2D(filters, 
                kernel_size,
                padding='same',
                strides=(1,1),
                groups=num_groups,
                kernel_initializer=keras.initializers.HeNormal(),
                activation=None
                )(net)    

    input_shape = K.int_shape(net)
    residual_shape = K.int_shape(net_sc)
    equal_channels = input_shape[3] == residual_shape[3]

    if not equal_channels:
        shortcut = Conv2D(filters, 
                kernel_size=(1,1),
                padding='same',
                strides=s,
                kernel_initializer=keras.initializers.HeNormal(),
                activation=None,
                bias_constraint=None
                )(net_pre)
    else:
        shortcut = net_sc

    out = add([net, shortcut])
    
    return out

def resnet_18(nclass):
    """
    build resnet-18 model using keras with TensorFlow backend.
    :param input_shape: input shape of network, default as (224,224,3)
    :param nclass: numbers of class(output shape of network), default as 1000
    :return: resnet-18 model
    """
    # 版本2

    tf.random.set_seed(1337) #不知道有何用 剛關閉 沒測試過(意思是沒訓練過)

    # tf.random.set_seed(getRandomState())

    # 前兩個高H 與 寬W 第三個 grayscale 1 ...RGB則是3
    # input_ = Input(shape=(128, 1, 1))
    input_ = Input(shape=(128, 256 , 1))

    FILTERS = [16, 32, 64, 128]
    RESNET_K = 2 
    KERNEL_SIZES = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    NUM_OF_GROUPS = [1, 1, 1, 1, 1, 1]
    # First Convolution
    net = Conv2D(FILTERS[0], 
                kernel_size=KERNEL_SIZES[0],
                padding='same',
                kernel_initializer=keras.initializers.HeNormal()
                )(input_)

    # Residual Stacks
    # for i in range(0,5):
    for i in range(0,4):
        net = residual_block_test(net, filters=FILTERS[i] * RESNET_K, kernel_size=KERNEL_SIZES[i], stride=2, num_groups=NUM_OF_GROUPS[i])
        for _ in range(1, 2):
            net = residual_block_test(net, filters=FILTERS[i] * RESNET_K, kernel_size=KERNEL_SIZES[i], num_groups=NUM_OF_GROUPS[i], preactivated=False)

    # Post Activation
    net = BatchNormalization()(net)
    net = Activation('relu')(net)

    # Pooling
    net = GlobalAvgPool2D()(net)

    # Classification Layer
    net = Dense(units=nclass,
                kernel_initializer=keras.initializers.HeNormal(),
                activation='linear'
                )(net)

    net = Activation('softmax')(net)

    model = Model(inputs=input_, outputs=net)

    return model

def cal_steps(num_images, batch_size):
   # calculates steps for generator
   steps = num_images // batch_size

   # adds 1 to the generator steps if the steps multiplied by
   # the batch size is less than the total training samples
   return steps + 1 if (steps * batch_size) < num_images else steps

def get_labels_from_tfdataset(tfdataset, batched=False):
    labels = list(map(lambda x: x[1], tfdataset)) # Get labels 
    if not batched:
        return tf.concat(labels, axis=0) # concat the list of batched labels
    return labels

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    i = 0
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[i] = score if score > 1.0 else 1.0
        i+=1
    
    return class_weight

if __name__ == '__main__':
    batch_size = 32
    img_height = 128
    img_width = 256
    
    base_path = os.path.dirname(os.path.abspath(__file__))  #D:\Desktop\BirdCLEF-Baseline-master\My-Deep-Learning\dataset-download
    TRAINSET_PATH = os.path.join("\\\?\\" + base_path, "dataset-download", "resnet_dataset")
    DATASET_PATH = os.path.join(TRAINSET_PATH, 'spec_Taiwan bird_combine2')
    data_dir = DATASET_PATH
    
    # 創立的資料夾名稱 create folder name
    create_folder_name = 'resnet_0813_Taiwan_combine2_batch32_Nadam'
    # 創立的資料夾路徑 
    create_folder_path = os.path.join(base_path, "my-model-save", "resnet", create_folder_name)
    if not os.path.isdir(create_folder_path):
        os.mkdir(create_folder_path)
    
    # 訓練dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                color_mode="grayscale",
                subset="training",
                shuffle=True,
                seed=1337,
                image_size=(img_height, img_width),
                batch_size=batch_size)

    class_names = train_ds.class_names
      
    # 驗證dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            color_mode="grayscale",
            subset="validation",
            shuffle=True,
            seed=1337,
            image_size=(img_height, img_width),
            batch_size=batch_size)
    
    # 建立model
    model = resnet_18(len(class_names))
    # model = resnet_18(610)
    # plot_model(model, 'ResNet-18.png', show_shapes=True)
    # plot_model(model, 'ResNet-18.pdf', show_shapes=True)
    # plot_model(model, 'ResNet-18-no_shapes.pdf')
    model.summary()

    model.compile(
        # optimizer='adam',
        optimizer='Nadam',
        # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # train 訓練
    checkpoint_path = create_folder_path + "/checkpoint/weights.{epoch:02d}.ckpt"
    
    #################### Create a callback that saves the model's weights
    cp_callback, ReduceLR = model_details.callbacks(checkpoint_path)
    
    #################### class_weight https://stackoverflow.com/questions/67181641/specifying-class-or-sample-weights-in-keras-for-one-hot-encoded-labels-in-a-tf-d
    pkl_folder_path = os.path.join(create_folder_path, "resnet_label_files_count.pkl")
    if os.path.exists(pkl_folder_path):
        print("--------------load class_weight--------------")                   
        with open(pkl_folder_path, 'rb') as f:
            class_num_training_samples = pickle.load(f)
    else:
        print("--------------no load class_weight--------------")                   
        class_num_training_samples = model_details.class_weights(train_ds, pkl_folder_path)
        
    ##########
    class_weights = create_class_weight(class_num_training_samples)
    print(f'class_weights {class_weights}')
        
    #################### steps_per_epoch
    train_samples = 0
    for key, value in class_num_training_samples.items() :
        # print(value)
        train_samples+=value
    print("--------------train samples--------------") 
    print(f"train samples: {train_samples} batch_size: {batch_size}")
    #3876306
    # steps_per_epoch = cal_steps(num_images=len(train_samples), batch_size=batch_size)
    steps_per_epoch = model_details.cal_steps(num_images=train_samples, batch_size=batch_size)
    print("--------------steps_per_epoch--------------") 
    print(f"steps_per_epoch {steps_per_epoch}")
    
    # 考慮dataset的表現 (要有緩衝)
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache()
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    epochs = 1
    run_time = 1
    for run_time in range(1, 100):
        print(f"--------------run epoch {run_time}--------------")        
        ################################################################################
        checkpoint_history_path = create_folder_path + "/checkpoint_history/"
        lastcheckpoint = model_details.checklastcheckpoint(checkpoint_history_path)
        ## lastcheckpoint 會等於最後一個資料的名稱 意思是 最後一次的epoch
        
        ################################################################################
        # 讀取最後一次的訓練model
        print("--------------load checkpoint(model)--------------")        
        load_folder_path = os.path.join(base_path, "my-model-save", "resnet", create_folder_name)
        if lastcheckpoint < 10:
            checkpoint_path = load_folder_path + "/checkpoint_history/weights.0" + str(lastcheckpoint) + ".ckpt/"
        else:
            checkpoint_path = load_folder_path + "/checkpoint_history/weights." + str(lastcheckpoint) + ".ckpt/"   
        print(checkpoint_path)
        if os.path.exists(checkpoint_path):
            print("--------------load weights--------------")                   
            # model.load_weights(checkpoint_path)
            model = tf.keras.models.load_model(checkpoint_path)
            print(model)
        else:
            print("--------------no weights--------------")
        
        print('\n')
        
        ################################################################################
        print("--------------start train--------------") 
        model.fit(train_ds, 
                epochs=epochs,
                steps_per_epoch = steps_per_epoch,
                class_weight = class_weights,
                validation_data=val_ds,
                callbacks=[cp_callback, ReduceLR])
        
        #################### Copy and move 
        print("--------------start copy checkpoint--------------")        
        history_lastcheckpoint = model_details.copycheckpoint(create_folder_path)
        
        ############# save history accuracy
        print("--------------save history_evaluate--------------") 
        history_path = os.path.join(create_folder_path, "evaluate")
        model_details.save_history(model, history_path) # lasthistory沒功能~~
        
        ############# copy history accuracy then delete 
        print("--------------start copy history_evaluate--------------")
        model_details.copyhistory(create_folder_path) # lasthistory沒功能~~
        
        
        ############# save_csv
        print("--------------save csv--------------")
        # (csv_folder_path, time, step, train_loss, train_acc):
        csv_folder_path = os.path.join(create_folder_path, "csv")
        time = "%s"%datetime.now() #獲得現在時間

        model_details.save_csv(csv_folder_path, time, lastcheckpoint+1, model.history.history['accuracy'], model.history.history['val_accuracy'],  model.history.history['loss'], model.history.history['val_loss'])
        

        ############# read history accuracy
        # read plk 
        # a_file = open(os.path.join(path, "history.pkl"), "rb")
        # testOPen = pickle.load(a_file)
        # print(f"testOPen {testOPen}")

        ############# plot
        # print("--------------save plot--------------") 
        # plot_path = os.path.join(create_folder_path, "plot")
        # model_details.plot_model_accuracy_loss(model, plot_path)

        ############# save model
        print("--------------save model--------------") 
        model_details.save_model(model, class_names, create_folder_path, model_name="resnet")
        
        print("done")
        
    # # Test model
    # img = keras.preprocessing.image.load_img(
    #     "test_file/Acadian Flycatcher.png", target_size=(img_height, img_width)
    #     )
    # img_array = keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    # predictions = model.predict(img_array)
    # score = predictions[0]
    # print(score)

            
    # print(
    #     "This image is %.2f percent cat and %.2f percent dog."
    #     % (100 * (1 - score), 100 * score)
    # )
# -----
