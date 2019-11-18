import numpy as np
import pandas as pd
import keras
import warnings
from scipy import signal

warnings.filterwarnings("ignore")

def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[:, ::-1]

def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig

def strange_data(sig, train=False):
    # # 数据增强
    if train:
        # if np.random.randn() > 0.5: sig = scaling(sig)
       sig = verflip(sig)
        # if np.random.randn() > 0.5: sig = shift(sig)
    return sig

# def window_stastic(data, peakind = None, win= 60):
#     '''
#     制作滑动窗口进行统计，目的：希望能定位到异常位置, 采用重采样降维到1000 500Hz-100Hz
#     :param data: (n,n_sample)
#     :param winN:
#     :param overlap:
#     :return:  (n,num,100)
#     '''
#     peaks_win = np.zeros([20,8,win+win])
#     for index2, peak in enumerate(peakind):
#         if index2 >=1 and index2 < peakind.shape[0]-1:
#             peaks_win[index2,:,:] = data[:,int(peakind[index2]-win):int(peakind[index2]+win)]
#     return peaks_win

def get_rawdata(file,resample_dim):
    df = pd.read_csv(file, sep=' ').values
    df = signal.resample(df,resample_dim)
    # df['III'] = df['II'] - df['I']
    # df['aVR'] = -(df['I'] + df['II']) / 2
    # df['aVL'] = df['I'] - df['II'] / 2
    # df['aVF'] = df['II'] - df['I'] / 2
    return df

class SpeechGen(keras.utils.Sequence):
    """
    'Generates data for Keras'

    list_IDs - list of files that this generator should load
    labels - dictionary of corresponding (integer) category to each file in list_IDs

    Expects list_IDs and labels to be of the same length
    """

    def __init__(self, X,data_path, mul_label, Y = None,batch_size=64, dim = 5000,down_dim=1000, shuffle=True,flag = False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = Y
        self.X = list(X)
        self.shape = X.shape
        self.mul_label = mul_label
        self.mul_dim = len(mul_label.keys())
        self.resample_dim = down_dim
        self.shuffle = shuffle
        self.data_path = data_path
        self.on_epoch_end()
        self.flag = 0  #由于标签
        if flag == True:
            self.flag = 1

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Generate data
        if self.flag == 0:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

            # Find list of IDs
            X_list = [self.X[k] for k in indexes]
            y_list = [self.labels[k] for k in indexes]
            X, y = self.__data_generation(X_list,y_list)
            return X, y
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            # Find list of IDs
            X_list = [self.X[k] for k in indexes]
            X = self.__data_generation1(X_list)
            return X


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # def Wavelet(self,sig):
    #     data = []
    #     for i in sig:
    #         w = pywt.Wavelet('db8')
    #         maxlev = pywt.dwt_max_level(len(i), w.dec_len)
    #         threshold = 0.4
    #         coeffs = pywt.wavedec(i, 'db8', level=maxlev)
    #         for ii in range(1, len(coeffs)):
    #             coeffs[ii] = pywt.threshold(coeffs[ii], threshold * max(coeffs[ii]))
    #         datarec = pywt.waverec(coeffs, 'db8')
    #         data.append(datarec)
    #     return np.array(data)


    def __data_generation(self, x,y):
        #'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        # X = np.empty((self.batch_size,8 ,self.resample_dim))
        X = np.empty((self.batch_size, self.resample_dim,8 ))
        Y = np.empty((self.batch_size,self.mul_dim))
        # X = np.zeros([self.batch_size,20,8,100])

        # Generate data
        for i, curX in enumerate(x):
            file_path = self.data_path + str(curX) + '.txt'
            x_df = get_rawdata(file_path,self.resample_dim)
            X[i] = x_df
            Y[i] = y[i]

        return X, Y

    def __data_generation1(self, x):
       # 'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size,8 ,self.resample_dim))
        # X = np.zeros([self.batch_size,20,8,100])
        X = np.empty((self.batch_size, self.resample_dim, 8))
        # Generate data
        for i, curX in enumerate(x):
            file_path = self.data_path + str(curX) + '.txt'
            x_df = get_rawdata(file_path,self.resample_dim)
            x_df = strange_data(x_df,train=True)
            X[i] = x_df
        return X

    # def __data_gen_windows(self,x,y):
    #     #'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #
    #     Y = np.empty((self.batch_size,self.mul_dim))
    #     X = np.zeros([self.batch_size,20,8,80])
    #
    #     # Generate data
    #     for i, curX in enumerate(x):
    #         file_path = self.data_path + str(curX) + '.txt'
    #         x_df = self.get_rawdata(file_path).values
    #         x_df = signal.resample(x_df, self.resample_dim)
    #         x_df = x_df.transpose()
    #         peakind = find_peaks(x_df[1, :], height=40, distance=40)
    #         peak_win = window_stastic(x_df, peakind=peakind[0], win=40)
    #         X[i] = peak_win
    #
    #         plt.figure(1, figsize=(6, 12))
    #
    #         for i in range(8):
    #             plt.subplot(8, 1, i + 1)
    #             plt.plot(np.arange(x_df.shape[-1]), x_df[i, :], 'b')
    #             plt.scatter(peakind[0], peakind[1]['peak_heights'], cmap='r')
    #             plt.title( str(curX), fontsize=10, color='#F08080')
    #
    #         plt.figure(2, figsize=(6, 12))
    #         for i in range(20):
    #             plt.subplot(4, 5, i + 1)
    #             plt.plot(np.arange(80), peak_win[i, 4, :], 'b')
    #             plt.title('FFT of Mixed wave', fontsize=10, color='#F08080')
    #
    #         plt.show()
    #
    #
    #         Y[i] = y[i]
    #
    #     return X, Y
    #
    # def __data_gen_windows1(self, x):
    #    # 'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.zeros([self.batch_size,20,8,80])
    #     # Generate data
    #     for i, curX in enumerate(x):
    #         file_path = self.data_path + str(curX) + '.txt'
    #         x_df = self.get_rawdata(file_path).values
    #         x_df = signal.resample(x_df, self.resample_dim).transpose()
    #         peakind = find_peaks(x_df[1, :], height=40, distance=40)
    #         peak_win = window_stastic(x_df, peakind=peakind[0], win=40)
    #         X[i] = peak_win
    #
    #     return X

