# encoding:utf-8

from ast import Param
import os
import pickle
from tabnanny import check
from matplotlib import pyplot as plt
import tensorflow as tf
from distutils.dir_util import copy_tree
import shutil
import numpy as np
import collections
from tqdm import tqdm
import math

#########################################################################################################
def use_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #   tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #   print("set_memory_growth")
    # except:
    #   # Invalid device or cannot modify virtual devices once initialized.
    #   pass
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

#########################################################################################################          
def preprocess_image(path, image_height, image_width, channel):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=channel)
    image = tf.image.resize(image, [image_height, image_width])
    
    # 数据增强
#     x=tf.image.random_brightness(x, 1)#亮度调整
#     x = tf.image.random_flip_up_down(x) #上下颠倒
#     x= tf.image.random_flip_left_right(x) # 左右镜像
#     x = tf.image.random_crop(x, [image_size, image_size, 3]) # 随机裁剪
    
    image /= 255.0  # normalize to [0,1] range
#    image= normalize(image) # 标准化
    return image


def load_and_preprocess_from_path_label(path, label):
    return preprocess_image(path, image_height=128, image_width=256, channel=1), label

#########################################################################################################        
## 找所有checkpoint資料夾裡面的最後一個checkpoint
def checklastcheckpoint(checkpoint_history_folder_path):
    i = 1
    checkpoint_path = checkpoint_history_folder_path + "weights.01.ckpt/"
    while(os.path.exists(checkpoint_path)):
        i+=1
        if i < 10:
            checkpoint_path = checkpoint_history_folder_path + "weights.0" + str(i) + ".ckpt/"
        else:
            checkpoint_path = checkpoint_history_folder_path + "weights." + str(i) + ".ckpt/"

    
    lastcheckpoint = i - 1
    return lastcheckpoint


#########################################################################################################
### 把最新的checkpoint複製到歷史區後 刪除最新的
def copycheckpoint(create_folder_path):
    
    checkpoint_history_folder_path = create_folder_path + "/checkpoint_history/"
    if not os.path.isdir(checkpoint_history_folder_path):
        os.mkdir(checkpoint_history_folder_path)
        
    i = 1
    new_path_count = 1
    checkpoint_path = create_folder_path + "/checkpoint/weights.01.ckpt/"    
    new_path = checkpoint_history_folder_path + "weights.01.ckpt/"
    
    ######## 檢查是否有檔案 沒有的話 就不用複製
    while(os.path.exists(checkpoint_path)):
        #################### 檢查 歷史區 是否有該epoch 一直 直到沒有
        while(os.path.exists(new_path)):
            if new_path_count < 10:
                new_path = checkpoint_history_folder_path + "weights.0" + str(new_path_count) + ".ckpt"
            else:
                new_path = checkpoint_history_folder_path + "weights." + str(new_path_count) + ".ckpt"
            new_path_count+=1

        print(f"現在要複製的 {checkpoint_path}")
        copy_tree(checkpoint_path, new_path)
        
        ## 然後刪除 以免誤會
        shutil.rmtree(checkpoint_path)
        
        ### 繼續下一個檔案複製
        i+=1
        if i < 10:
            checkpoint_path = create_folder_path + "/checkpoint/weights.0" + str(i) + ".ckpt"
        else:
            checkpoint_path = create_folder_path + "/checkpoint/weights." + str(i) + ".ckpt"
    
    return new_path_count
           
######################################################################################################### 
def callbacks(create_folder_path):
    checkpoint_path = create_folder_path
    ModelCheck = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False, # True
                                                 verbose=1,
                                                 save_best_only=False,
                                                 save_freq='epoch')

    ReduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=3e-4)
    return ModelCheck, ReduceLR
#########################################################################################################
def cal_steps(num_images, batch_size):
   # calculates steps for generator
   steps = num_images // batch_size
   # adds 1 to the generator steps if the steps multiplied by
   # the batch size is less than the total training samples
   return steps + 1 if (steps * batch_size) < num_images else steps

#########################################################################################################
#################### class_weight https://stackoverflow.com/questions/67181641/specifying-class-or-sample-weights-in-keras-for-one-hot-encoded-labels-in-a-tf-d
def class_weights(train_ds, pkl_folder_path):
    
    class_num_training_samples = {}
    print("start compute samples")
    for f in train_ds.file_paths:
        class_name = f.split('\\')[len(f.split('\\'))-2]
        # print(class_name)
        if class_name in class_num_training_samples:
            class_num_training_samples[class_name] += 1
        else:
            class_num_training_samples[class_name] = 1          
    #     print(f"{f} done")
    # print(class_num_training_samples)                
    ###################### save to dictionary to pkl
    print("save dict to pkl")
    a_file = open(pkl_folder_path, "wb")
    pickle.dump(class_num_training_samples, a_file)
    a_file.close()
    
    return class_num_training_samples

def class_weights_v2(numpy_labels, pkl_folder_path):
    
    class_num_training_samples = {}
    print("start compute samples")
    for f in range(len(numpy_labels)):
        class_name = numpy_labels[f]
        # print(class_name)
        if class_name in class_num_training_samples:
            class_num_training_samples[class_name] += 1
        else:
            class_num_training_samples[class_name] = 1          
    #     print(f"{f} done")
    # print(class_num_training_samples)                
    ###################### save to dictionary to pkl
    print("save dict to pkl")
    a_file = open(pkl_folder_path, "wb")
    pickle.dump(class_num_training_samples, a_file)
    a_file.close()
    
    return class_num_training_samples

# not done not complete
def class_weights2(train_ds, pkl_folder_path, label_to_index):
    # print("start compute samples")
    # for y, k in train_ds:
    #     print(k)
        
    # classes = np.concatenate([y for x, y in train_ds], axis=0)
    # unique = np.unique(classes, return_counts=True)
    # class_num_training_samples_index = dict(zip(unique[0], unique[1]))
    # print(class_num_training_samples_index)
    
    # class_num_training_samples = dict(label_to_index)
    # index = 0
    # for key, value in class_num_training_samples.items():
    #     # print(key)
    #     class_num_training_samples[key] = class_num_training_samples_index[index]
    #     index+=1
    
    print("start compute samples")
    class_num_training_samples_testt = {}
    for image, label in tqdm(train_ds):
        # print(label)
        for j in label:
            # print(j)
            if str(j.numpy()) in class_num_training_samples_testt:
                class_num_training_samples_testt[str(j.numpy())] += 1
            else:
                class_num_training_samples_testt[str(j.numpy())] = 1 
                
    print(class_num_training_samples_testt)
    
    # print("start compute samples")
    # class_num_training_samples_testt = {}    
    # images, labels = tuple(zip(*train_ds))    
    # for i in labels:
    #     # print(i.numpy())
    #     for j in i.numpy():
    #         if str(j) in class_num_training_samples_testt:
    #             class_num_training_samples_testt[str(j)] += 1
    #         else:
    #             class_num_training_samples_testt[str(j)] = 1 
    
    # print("start compute samples")
    # class_num_training_samples_testt = {}
    # for images, labels in train_ds.unbatch():
    #     # ds_labels.append(labels) # or labels.numpy().argmax() for int labels
    #     if labels.numpy() in class_num_training_samples_testt:
    #         class_num_training_samples_testt[labels.numpy()] += 1
    #     else:
    #         class_num_training_samples_testt[labels.numpy()] = 1
    
    od = collections.OrderedDict(sorted(class_num_training_samples_testt.items()))
    
    class_num_training_samples = dict(label_to_index)
    
    print("現在要開始進行改value動作")
    print(f"{od}")    
    print(f'{class_num_training_samples}')
    
        
    index = 0
    for key, value in class_num_training_samples.items():
        # print(key)
        class_num_training_samples[key] = od[str(index)]
        index+=1
                
    # ###################### save to dictionary to pkl
    print("save dict to pkl")
    a_file = open(pkl_folder_path, "wb")
    pickle.dump(class_num_training_samples, a_file)
    a_file.close()
    
    return class_num_training_samples

#########################################################################################################
############# create_class_weight
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


#########################################################################################################      
############# copy history accuracy     
def copyhistory(create_folder_path):
 
    history_path = create_folder_path + "/evaluate/history.01.pkl"
    
    backuppath = create_folder_path + "/history_evaluate/"
    if not os.path.isdir(backuppath):
        os.mkdir(backuppath)
    
    new_path = backuppath + "history.01.pkl"
    
    i = 1
    j = 1

    # 檢查有沒有最新的 history pkl檔案 沒有的話就不用複製了
    while(os.path.exists(history_path)):        
        #################### 檢查新目的是否有該epoch 一直直到沒有
        while(os.path.exists(new_path)):
            if j < 10:
                new_path = backuppath + "history.0" + str(j) + ".pkl"
            else:
                new_path = backuppath + "history." + str(j) + ".pkl"
            j+=1

        print(f"現在要複製的 {history_path}")
        # print(new_path)
        shutil.copyfile(history_path, new_path)
        
        ## 然後刪除 以免誤會
        os.remove(history_path)	
        
        i+=1
        if i < 10:
            history_path = create_folder_path + "/evaluate/history.0" + str(i) + ".pkl"
        else:
            history_path = create_folder_path + "/evaluate/history." + str(i) + ".pkl"
            
#########################################################################################################
############# save history accuracy
def save_history(model, history_path):
    if not os.path.isdir(history_path):
        os.mkdir(history_path)
        
    file_path = os.path.join(history_path, "history.01.pkl")
    with open(file_path, 'wb') as file_pi:
        pickle.dump(model.history.history, file_pi)
        
#########################################################################################################
############# plot
def plot_model_accuracy_loss(model, plot_path):
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plot_path, "accuracy.png"))
    # plt.show()
    
    plt.clf()
    
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss']) 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plot_path, "loss.png"))
    # plt.show()
    
#########################################################################################################   
# Save the model
def save_model(model, class_names, model_path, model_name):
    
    # model.save(model_path + '/' + model_name + '_model.h5')    
    # model.save_weights(model_path + '/' + model_name + '_weights.h5')
    
    ############# Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    tflite_model = converter.convert()
    # Save the model.
    with open(model_path + '/' + model_name + '_model.tflite', 'wb') as f:
        f.write(tflite_model)
    # Create Label  https://yanwei-liu.medium.com/tensorflow-lite%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-c95e12f97b9a
    with open(model_path + '/' + model_name + '_labels.txt', 'w') as f:
        f.write('\n'.join(class_names))
    # model.save('mytrain/my_model')
    # model.save(model_path)    
    
def save_model_v2(model, model_path, model_name):
    
    # model.save(model_path + '/' + model_name + '_model.h5')    
    # model.save_weights(model_path + '/' + model_name + '_weights.h5')
    
    ############# Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    tflite_model = converter.convert()
    # Save the model.
    with open(model_path + '/' + model_name + '_model.tflite', 'wb') as f:
        f.write(tflite_model)
    # Create Label  https://yanwei-liu.medium.com/tensorflow-lite%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-c95e12f97b9a

import random
import csv
def first_write_csv(csv_file_path):
    if not os.path.exists(csv_file_path):        
        with open(csv_file_path, 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow(['time','step','train accuracy','validation accuracy', 'train loss','validation loss'])

def save_csv(csv_folder_path, time, step, train_acc, val_acc, train_loss, val_loss):
    if not os.path.isdir(csv_folder_path):
        os.mkdir(csv_folder_path)
    csv_file_path = os.path.join(csv_folder_path, "train_acc.csv")
    
    first_write_csv(csv_file_path)
    
    with open(csv_file_path, 'a', newline='', encoding='UTF-8') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料s
        writer.writerow([time,step,train_acc, val_acc, train_loss, val_loss])


# #初始化train数据
# t_loss = 0.4
# t_acc = 0.3
# for i in range(20):#假设迭代20次
#     time = "%s"%datetime.now()#获取当前时间
#     step = "Step[%d]"%i
#     t_loss = t_loss - random.uniform(0.01,0.017)
#     train_loss = "%f"%t_loss
#     t_acc = t_acc + random.uniform(0.025,0.035)
#     train_acc = "%g"%t_acc


