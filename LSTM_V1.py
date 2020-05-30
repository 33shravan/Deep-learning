

import numpy as np
import keras
import scipy.io
from keras.layers import *
from numpy import linalg as LA
import sklearn
from keras.layers.merge import concatenate
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from keras.utils import np_utils, generic_utils
from keras.models import Sequential

from sklearn.externals import joblib
from sklearn import preprocessing
import spacy
from keras.utils import multi_gpu_model
from keras.models import Model
# from spacy.en import English





import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

def image_vector(img_id, map, features):
	no_of_samples = len(img_id)
	no_dimensions = features.shape[0]
	img_vector = np.zeros((no_of_samples, no_dimensions))
	for j in range(len(img_id)):
		img_vector[j,:] = features[:,map[img_id[j]]]

	return img_vector


def onehot_answers(answers, encoder):
	y = encoder.transform(answers)
	no_classes = encoder.classes_.shape[0]
	Y = np_utils.to_categorical(y, no_classes)
	return Y



mlp_hidden_units=1000
mlp_hidden_layers=2
lstm_hidden_units=2
dropout=0.50
activation_mlp='sigmoid'
num_epochs=100
batch_size=128


# import numpy as np
# def loadGloveModel(gloveFile):
#     print("Loading Glove Model")
#     f = open(gloveFile,'r')
#     model = {}
#     o=0
#     for line in f:
#         splitLine = line.split()
#         word = splitLine[0]

#         print(word)
# #         print(splitLine[1])
#         o=o+1
# #         if(o):
# #             break
#         embedding = np.array([float(val) for val in splitLine[1:]])
#         model[word] = embedding
#         print(o)
#     print("Done.",len(model)," words loaded!")
#     return model
# model=loadGloveModel('/home/jeevankr/data/glove.840B.300d.txt')


# In[ ]:


# word_vec_dim= 300
# img_dim = 4096
# max_len = 8

# no_classes = 1000
# import pandas as pd
# import csv

# words = pd.read_table('/home/jeevankr/data/glove.840B.300d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


# In[ ]:


# def vec(w):
#   return words.loc[w].as_matrix()


# In[ ]:


# vec('?')


# In[ ]:


def get_questions_tensor_timeseries(questions, nlp, timesteps):
    no_of_samples = len(questions)
    word_vec_dim = 300
    questions_tensor = np.zeros((no_of_samples, timesteps, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i])
#         print(i)
        for j in range(len(tokens)):
            try:
                temp=tokens[j].vector
#                 temp = model[str(tokens[j])]

            except KeyError:
                temp=tokens[j].vector
            else:
                a=1
            questions_tensor[i,j,:] = temp

    return questions_tensor


# In[ ]:


nlp=spacy.load('en_vectors_web_lg')


# In[ ]:


# tokens = nlp('q73649bvqwyieu')
# tokens[0].vector
# j=1
# if j<100:
#     try:
#         temp = model[str(tokens[0])]
#     except KeyError:
#         temp=tokens[0].vector
#     else:
#         print("Hi")
# print(temp)

questions_train = open('/home/jeevankr/data/ques_train_1000.txt', 'r').read().splitlines()
answers_train = open('/home/jeevankr/data/ann_train_1000.txt', 'r').read().splitlines()
images_train = open('/home/jeevankr/data/images_train_1000.txt', 'r').read().splitlines()

questions_val=open('/home/jeevankr/data/ques_val_1000.txt', 'r').read().splitlines()
answers_val = open('/home/jeevankr/data/ann_val_1000.txt', 'r').read().splitlines()
images_val = open('/home/jeevankr/data/images_val_1000.txt', 'r').read().splitlines()

answers = open('/home/jeevankr/data/answer_encoder_10001.txt', 'r').read().splitlines()




len(questions_train)
len(answers)



# answers = open('/home/jeevankr/data/answer_encoder.txt', 'r').read().splitlines()
len(answers)



maxAnswers =1000
labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answers)
no_classes = len(list(labelencoder.classes_))




features_struct = scipy.io.loadmat('/home/jeevankr/data/vgg_feats.mat')
features = features_struct['feats']
# print ('loaded vgg features')





# for i in range(123287):
#     print(i)
#     norm=LA.norm(features[:,i])
#     features[:,i]=features[:,i]/norm*1.0
# #     temp=features[:,i].reshape(4096,1)
# #     sklearn.preprocessing.normalize(features[:,i].reshape(4096,1), norm='l2', axis=0, copy=True, return_norm=False)






# answers_train[80]
img_dim = 4096
image_model = Sequential()
image_model.add(Reshape((img_dim,), input_shape=(img_dim,)))
image_model.add(Dense(1024,activation='sigmoid'))
image_model.output
# image_model.add(Activation('tanh'))




# language_model = Sequential()




input_ = Input(shape = (25,300))

lstm ,a,b= LSTM(units = 512,return_sequences=True, return_state=True)(input_)
lstm2,a2,b2= LSTM(units = 512,return_sequences=False, return_state=True)(lstm)
gh=Concatenate()([a,b,a2,b2])
gh=Dense(1024)(gh)
gh=Activation('sigmoid')(gh)
language_model=Model(input_,gh)
gh

# Concatenate()([.output,language_model.output])
# dense = Dense(10, activation='softmax')(lstm)

# modell = Model(inputs = input_, outputs = dense)




# a = Input(shape=(25, 300))
# aa=LSTM(input_shape=(25, 300),units=512, return_sequences=True, return_state=True)
# # state_h, state_c=LSTM(input_shape=(25, 300),units=512,return_sequences=True,return_state=True)
# language_model2 = Model(aa)
# language_model.add(LSTM(input_shape=(25, 300),units=512,return_sequences=False,return_state=True))
# language_model.add(LSTM(input_shape=(25, 300),units=512,return_sequences=True))
# for i in range(lstm_hidden_units-2):
#     language_model.add(LSTM(units=512,return_sequences=True))
# language_model.add(LSTM(units=512,return_sequences=False))




# model = Sequential()
# model.add(merge.concatenate([language_model.output, image_model.output]))




mergedOut = Multiply()([image_model.output,language_model.output])



for i in range(mlp_hidden_layers):
    mergedOut=Dense(mlp_hidden_units)(mergedOut)
    mergedOut=Activation(activation_mlp)(mergedOut)
    mergedOut=Dropout(dropout)(mergedOut)
mergedOut=Dense(no_classes)(mergedOut)
mergedOut=Activation('softmax')(mergedOut)




from keras.models import Model




newModel = Model([image_model.input,language_model.input], mergedOut)




es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=8)





newModel.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])





img_ids = open('/home/jeevankr/data/coco_vgg_IDMap.txt').read().splitlines()
map = {}
for i in img_ids:
	id_split = i.split()
	map[id_split[0]] = int(id_split[1])





# nlp=spacy.load('en_vectors_web_lg')

word_vec_dim = 300
# questions_train





timesteps = len(nlp(questions_train[-2]))
timesteps





timesteps
eer=questions_train[2:4]
tokens=nlp(eer[0])
str(tokens[0])




keras.utils.print_summary(newModel, line_length=None, positions=None, print_fn=None)
# 4194304





# # filename = '\\glove.6B.300d.txt'

# glove_vocab = []
# glove_embed=[]
# embedding_dict = {}

# file = open('/home/jeevankr/data/glove.840B.300d.txt','r',encoding='UTF-8')
# i=0
# for line in file.readlines():
#     print(i)
#     i=i+1
#     row = line.strip().split(' ')
#     vocab_word = row[0]
#     glove_vocab.append(vocab_word)
#     embed_vector = [float(i) for i in row[1:]] # convert to list of float
#     embedding_dict[vocab_word]=embed_vector
#     glove_embed.append(embed_vector)




# from gensim.scripts.glove2word2vec import glove2word2vec
# word2vec_output_file = 'glove.6B.100d.txt.word2vec'
# glove2word2vec('/home/jeevankr/data/glove.840B.300d.txt', word2vec_output_file)
# from gensim.models import KeyedVectors
# filename = 'glove.6B.100d.txt.word2vec'
# model = KeyedVectors.load_word2vec_format(filename, binary=False)



# # result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# # model['boy'].shape
# import re
# questions_train[4864]
# model.get_vector

X_q = get_questions_tensor_timeseries(questions_train, nlp, 25)

# no_of_samples = len(questions_train)
# word_vec_dim = 300
# print("dd")
# timesteps=8
# questions_tensor = np.zeros((no_of_samples, timesteps, word_vec_dim))
# print("rr")
# for i in range(len(questions_train)):
#     questions_train[i] = re.sub(' +', ' ', questions_train[i])
#     tokens = nlp(questions_train[i])
#     print(i)
#     for j in range(len(tokens)):
#         if j<timesteps:
#             try:
#                 temp = model[str(tokens[j])]
#             except KeyError:
#                 temp=tokens[j].vector
#             else:
#                 a=1
#             questions_tensor[i,j,:] = temp



X_i = image_vector(images_train, map, features)


# timesteps2 = len(nlp(questions_val[-1]))
print(images_train[84326])
timesteps2=25
questions_train[84326]

X_q_val = get_questions_tensor_timeseries(questions_val, nlp, 25)
# no_of_samples = len(questions_train)
# word_vec_dim = 300
# print("dd")
# timesteps=20
# questions2_tensor = np.zeros((no_of_samples, timesteps, word_vec_dim))
# print("rr")
# for i in range(len(questions_train)):
# #     questions_train[i] = re.sub(' +', ' ', questions_val[i])
#     tokens = nlp(questions_train[i])
#     print(i)
#     if(i==9888 or i==21374 or i==28737 or i==30862 or i==37547 or i==41113 or i==67298 or i==84260 or i==84326):
#         continue
#     for j in range(len(tokens)):
# #         if j<timesteps:
#         temp = tokens[j].vector
#             try:
#                 temp = tokens[j].vector
#             except KeyError:
#                 temp=tokens[j].vector
#             else:
#                 a=1
#         questions2_tensor[i,j,:] = temp


X_i_val = image_vector(images_val, map, features)

y_train = onehot_answers(answers_train, labelencoder)
y_val = onehot_answers(answers_val, labelencoder)

# newModel = multi_gpu_model(newModel,gpus=)
newModel.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])

newModel.fit([X_i, X_q], y_train, epochs=num_epochs,batch_size=batch_size,validation_data=([X_i_val,X_q_val],y_val),callbacks=[es])

newModel.save('LSTM_130_tanhhhh2.h5')


print(history.history.keys())

from keras.models import load_model
model = load_model('LSTM_130_tanhhhh2.h5')

strr='34582'
i=2
print(questions_val[images_val.index(strr)+i])
print(answers_val[images_val.index(strr)+i])
dd=questions_val[images_val.index(strr)+i]
dd

# dd='Is it night ?'
# x_i_trai = image_vector([strr],map, features)
# X_i_val = image_vector(images_val, map, features)
X_q_val = get_questions_tensor_timeseries(['What is the animal ?'], nlp, timesteps2)


er2=model.predict([x_i_trai, X_q_val])


labelencoder.classes_[np.argmax(er2)]
