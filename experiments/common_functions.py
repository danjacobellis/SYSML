import os
import pathlib
import numpy as np
import scipy as sp
import tensorflow as tf
import larq as lq
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

def slog(x):
    return np.log(1+np.abs(x))*np.sign(x)

def ulog(x):
    return np.log(1+np.abs(x))

def STFT(x):
    X = np.fft.rfft(np.reshape(x,(64,254)),axis=1)
    Xc = np.real(X)
    Xs = np.imag(X)
    X = np.reshape(np.hstack((Xc,Xs)),(128,128))
    return X

def ISTFT(X):
    Xc = X[0::2,:]
    Xs = X[1::2,:]
    X = Xc + 1j*Xs
    return np.reshape(np.fft.irfft(X,axis=1),-1)

def play_audio(x):
    return display.Audio(np.reshape(x,-1), rate=16000)

def im2col(x):
    shape = np.shape(x)
    y = np.zeros((shape[0]//2,shape[1]//2,4))
    for i1 in range(shape[0]//2):
        for i2 in range(shape[1]//2):
            block = x[2*i1:2*i1+2,2*i2:2*i2+2]
            y[i1,i2,:] = np.reshape(block,-1)
    return y

def col2im(x):
    shape = np.shape(x)
    y = np.zeros((2*shape[0],2*shape[1]))
    for i1 in range(shape[0]):
        for i2 in range(shape[1]):
            block = np.reshape(x[i1,i2,:],(2,2))
            y[2*i1:2*i1+2,2*i2:2*i2+2] = block
    return y

def VQ_encode(x,codebook):
    v = np.reshape(im2col(STFT(x)),(64*64,4))
    v_hat, _ = sp.cluster.vq.vq(v,codebook)
    # return v_hat.astype('int8')
    return binarize(v_hat.astype('int8'))

def VQ_encode_STFT(X,codebook):
    v = np.reshape(im2col(X),(64*64,4))
    v_hat, _ = sp.cluster.vq.vq(v,codebook)
    # return v_hat.astype('int8')
    return binarize(v_hat.astype('int8'))

def VQ_decode(b,codebook):
    x = serialize(b)
    return ISTFT(col2im(np.reshape(codebook[x],(64,64,4))))

def VQ_decode_STFT(b,codebook):
    x = serialize(b)
    return col2im(np.reshape(codebook[x],(64,64,4)))

def binarize(v_hat):
    v_hat = np.reshape(v_hat,(64,64))
    v_hat = np.expand_dims(v_hat,2)
    return np.concatenate((v_hat&1,v_hat>>1&1,v_hat>>2&1,v_hat>>3&1),axis=2)

def serialize(b):
    return np.reshape(b[:,:,0] + (b[:,:,1]<<1) + (b[:,:,2]<<2) + (b[:,:,3]<<3),-1)

def load_mini_speech_commands():
    DATASET_PATH = 'data/mini_speech_commands'

    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
      tf.keras.utils.get_file(
          'mini_speech_commands.zip',
          origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
          extract=True,
          cache_dir='.', cache_subdir='data')

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=None,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16256,
        subset='both')
    
    X = []
    y = []
    Xv = []
    yv = []
    for xi,yi in train_ds:
        X.append(np.expand_dims(STFT(xi),0))
        y.append(yi)
    for xi,yi in val_ds:
        Xv.append(np.expand_dims(STFT(xi),0))
        yv.append(yi)
    X = np.vstack(X)
    y = np.vstack(y)
    Xv = np.vstack(Xv)
    yv = np.vstack(yv)
    
    return X, y, Xv, yv

