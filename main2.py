import CallBackList
import densenet
import generator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import clean_data
import tensorflow as tf
import time
import os
from keras.backend.tensorflow_backend import set_session

np.random.seed(2019)
def main():
    data_path = r'G:\xindian\train_1\dataSet/'

    user_path = r'F:\天池\心电\code/'

    model_name = '1D_denseres2'

    PARA = {
        'batch_size':18,
        'dim':5000,
        'CSV':False,
        'CSV_path':user_path + '{}training_log.csv'.format(model_name),
        'tensorboard':False,
        'tensorboard_path':user_path + '{}'.format(model_name),
        'model':True,
        'model_path':user_path +model_name+'.h5',
        'monitor':'accuracy',
        'LR':True,
        'EarlyStop':False,
        'CheckPiont':True,
        'LossHistory':True,
        'continue':False
    }

    train_df = clean_data.load_and_clean_label(data_path + r'hf_round2_train.txt')

    with open(data_path+'hf_round2_arrythmia.txt', 'r',encoding='utf-8') as file:
        arrythmia = file.readlines()
    
    arrythmia,number2class = dict([[x, i] for i,x in enumerate([x.split()[0] for x in arrythmia])]),dict([[i, x] for i,x in enumerate([x.split()[0] for x in arrythmia])])
    # np.savetxt('arrythmia_txt.txt',X=arrythmia)
    print(arrythmia,'\n',number2class)

    def encode(label, dic={}):
        y = np.zeros((len(label), len(dic.keys())))
        list_keys = list(dic.keys())
        for i, y1 in enumerate(label):
            for j in y1:
                index = list_keys.index(j)
                if y[i, index] == 0:
                    y[i, index] = y[i, index] + 1
        return y

    def decode(label,dic=number2class,threshold=0.5):
        xi =[]
        for index,i in enumerate(label):
            if i > threshold:
                xi.append(dic[index])
        return xi

    Y_train = encode(train_df['arrythmia'],arrythmia)
    print(decode([1]*34))

    X_train, X_valid, Y_train1, Y_valid = train_test_split(train_df['id'], Y_train, test_size=0.05)
    wight_train = np.sum(Y_train1, axis=0)
    all_train = np.sum(wight_train)

    print(wight_train)
    print(X_train.shape,X_valid.shape)

    print('############训练##############')

    #生成器

    train_gen = generator.SpeechGen(
                                    X = X_train,
                                    Y = Y_train1,
                                    mul_label = arrythmia,
                                    data_path=data_path+ r'hf_round2_train/',
                                    batch_size=PARA['batch_size'],
                                    dim=PARA['dim'],
                                    down_dim=3000,
                                    shuffle=True,
                                    )

    valid_gen = generator.SpeechGen(X=X_valid,
                                    Y=Y_valid,
                                    mul_label=arrythmia,
                                    data_path=data_path+ r'hf_round2_train/',
                                    batch_size=PARA['batch_size'],
                                    dim=PARA['dim'],
                                    down_dim=3000,
                                    shuffle=True)
    #

    #
    Xi,Yi = train_gen.__getitem__(0)
    print('Xi.shape:',Xi.shape,'Yi.shape:',Yi.shape)

    Xi,Yi = valid_gen.__getitem__(0)
    print('Xi.shape:',Xi.shape,'Yi.shape:',Yi.shape)

    # import densenet1D
    # import densenet_mulscale
    import denseres1D
    model = denseres1D.DenseNet(blocks=[3, 4, 6, 6],
                 include_top=True,
                # input_shape=(Xi.shape[1],Xi.shape[2],Xi.shape[3]),
                  input_shape=(Xi.shape[1],Xi.shape[2]),
                 pooling=None,
                classes=len(arrythmia.keys()))

    from keras import backend as K
    import metric
    from keras.losses import binary_crossentropy

    weight_tensor = tf.constant(1 - wight_train / all_train, dtype=tf.float32)

    # weight_tensor = tf.constant(1/np.log(wight_train),dtype=tf.float32)

    def binary_crossentropy(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weight_tensor) #+ 1 - metric.f1(y_true, y_pred)

    PARA['continue'] = True
    # model.load_weights(user_path + r'My_advanced_k20_blur_SPP_2.h5', by_name=True)
    if PARA['continue'] == False:
        import denseres1D
        model = denseres1D.DenseNet(blocks=[3, 4, 6, 6],
                                    include_top=True,
                                    # input_shape=(Xi.shape[1],Xi.shape[2],Xi.shape[3]),
                                    input_shape=(Xi.shape[1], Xi.shape[2]),
                                    pooling=None,
                                    classes=len(arrythmia.keys()))
        model.load_weights(r'F:\天池\心电\code\1D_denseres.h5',by_name=True)
        model.compile(optimizer='adam', loss=binary_crossentropy, metrics=[metric.f1])
        # yi_t, y_pre, y_true = [], [] , []
        # for i in range(len(X_valid)):
        #     X,y = valid_gen.__getitem__(i)
        #     yi = model.predict(X)
        #     yi = yi.reshape([-1])
        #     yi_t.append(decode(yi, dic=arrythmia, threshold=0.5))
        #     y_true.append(decode(y.reshape([-1]), dic=arrythmia, threshold=0.5))
        print('over')
        val = model.evaluate_generator(valid_gen)
        print(val)

    else:
        # print('load_My_advanced_k20_blur_SPP')
        # import densenet
        # model0 = densenet.DenseNet(blocks=[6, 6, 6, 6],
        #                             include_top=True,
        #                             # input_shape=(Xi.shape[1],Xi.shape[2],Xi.shape[3]),
        #                             input_shape=(1000, Xi.shape[2]),
        #                             pooling=None,
        #                             classes=34)
        # model0.load_weights(r"F:\天池\心电\code\My_advanced_k20_blur_SPP.h5", by_name=True)
        #
        # print('My_advanced_k20_blur_SPP_3')
        # import densenet1D
        # model = densenet1D.DenseNet(blocks=[6, 6, 6, 6],
        #                             include_top=True,
        #                             # input_shape=(Xi.shape[1],Xi.shape[2],Xi.shape[3]),
        #                             input_shape=(Xi.shape[1], Xi.shape[2]),
        #                             pooling=None,
        #                             classes=34)
        # model.load_weights(r"F:\天池\心电\code\My_advanced_k20_blur_SPP_3.h5", by_name=True) #0.9092

        # print('My_advanced_k20_blur_SPP_2')
        # import densenet_1
        # model = densenet_1.DenseNet(blocks=[6, 6, 6, 6],
        #                             include_top=True,
        #                             # input_shape=(Xi.shape[1],Xi.shape[2],Xi.shape[3]),
        #                             input_shape=(Xi.shape[1], Xi.shape[2]),
        #                             pooling=None,
        #                             classes=len(arrythmia.keys()))
        # model.load_weights(r"F:\天池\心电\code\My_advanced_k20_blur_SPP_2.h5", by_name=True) #0.9114

        # print('1D_My_advanced_k20_blur_SPP_mulscale_1')
        # import densenet_mulscale
        # model = densenet_mulscale.DenseNet(blocks=[3, 4, 6, 6],
        #                                    include_top=True,
        #                                    # input_shape=(Xi.shape[1],Xi.shape[2],Xi.shape[3]),
        #                                    input_shape=(Xi.shape[1], Xi.shape[2]),
        #                                    pooling=None,
        #                                    classes=len(arrythmia.keys()))
        # model.load_weights(r"F:\天池\心电\code\1D_My_advanced_k20_blur_SPP_mulscale_1.h5", by_name=True) #0.9059
        # import keras_applications.resnet
        # print('1D_My_advanced_k20_blur_SPP_mulscale1')
        # import densenet_mulscale1
        # model = densenet_mulscale1.DenseNet(blocks=[3, 4, 6, 6],
        #                                    include_top=True,
        #                                    # input_shape=(Xi.shape[1],Xi.shape[2],Xi.shape[3]),
        #                                    input_shape=(Xi.shape[1], Xi.shape[2]),
        #                                    pooling=None,
        #                                    classes=len(arrythmia.keys()))
        # model.load_weights(r"F:\天池\心电\code\1D_My_advanced_k20_blur_SPP_mulscale1.h5", by_name=True)

        print('1D_focal_denseres')

        def multi_category_focal_loss1(alpha, gamma=2.0):
            """
            focal loss for multi category of multi label problem
            适用于多分类或多标签问题的focal loss
            alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
            当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
            Usage:
             model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
            """
            epsilon = 1.e-7
            alpha = tf.constant(alpha, dtype=tf.float32)
            # alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
            # alpha = tf.constant_initializer(alpha)
            gamma = float(gamma)

            def multi_category_focal_loss1_fixed(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
                ce = -tf.log(y_t)
                weight = tf.pow(tf.subtract(1., y_t), gamma)
                fl = tf.multiply(weight, ce)* alpha
                loss = tf.reduce_mean(fl)
                return loss

            return multi_category_focal_loss1_fixed

        import densenet_2
        model = densenet_2.DenseNet(blocks=[3, 4, 6, 6],
                                           include_top=True,
                                           # input_shape=(Xi.shape[1],Xi.shape[2],Xi.shape[3]),
                                           input_shape=(Xi.shape[1], Xi.shape[2]),
                                           pooling=None,
                                           classes=len(arrythmia.keys())) #0.908

        model.compile(optimizer='adam', loss=binary_crossentropy, metrics=[metric.f1])
        # model.compile(optimizer='adam', loss=multi_category_focal_loss1(alpha =1- wight_train / all_train, gamma=0.5), metrics=[metric.f1])
        model.summary()
        callbacklist = CallBackList.CallBackList(PARA)
        results = model.fit_generator(train_gen, validation_data = valid_gen, epochs= 30,
                                      use_multiprocessing=False, workers=1,verbose=1,
                                      steps_per_epoch= np.ceil(float(len(Y_train1)/PARA['batch_size'])),
                                      callbacks=callbacklist)


    # test_gen = generator.SpeechGen(X=X_test,
    #                                flag=True,
    #                                mul_label=arrythmia,
    #                                data_path=r'G:\xindian\train_1\dataSet\testA/',
    #                                batch_size=PARA['batch_size'],
    #                                dim= 1,
    #                                shuffle=False)
    #
    #
    # Xi= test_gen.__getitem__(0)
    # print('Xi.shape:',Xi.shape)
    #
    # h5_name = r'F:\天池\心电\code\logs\weights_h5\My_advanced.h5'
    # model.load_weights(h5_name)
    # model.summary()
    #
    # threshold = 0.5
    #
    # yi_t,y_pre = [],{}
    # from tqdm import tqdm
    # for i in tqdm(range(len(X_test))):
    #     X = test_gen.__getitem__(i)
    #     yi = model.predict(X)
    #     yi = np.where(np.greater(yi, threshold), 1, 0)
    #     yi_t.append( mlb.inverse_transform(yi))
    # import time
    # with open(r'G:\xindian\train_1\dataSet/hf_round1_subB_noDup_rename.txt','r',encoding='utf-8') as f1:
    #     with open(base_path + 'logs/' + '%s_result.txt'%(time.strftime("%Y%m%d%H%M")),'w',encoding='utf-8') as f2:
    #         for index,line1 in enumerate(f1):
    #             f2.write(line1.split('\n')[0])
    #             for i in yi_t[index][0]:
    #                 f2.write('\t'+ i)
    #             f2.write('\n')
    # f1.close()
    # f2.close()

if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("command", metavar="<command>", help="main")
    # args = parser.parse_args()
    # if (args.command == "main"):
    main()