
#twitter assignment
#----------------------#
#----------------------#

#libraries
import numpy as np
import numpy.matlib
import codecs
import re
import random
from operator import itemgetter
##########################
import json

from keras.callbacks import TensorBoard
from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D,GRU
from keras.layers.core import Flatten, Dense, Dropout,Reshape
from keras.models import Sequential

######################################
'''
#one hot representation
Here we make our one hot encoder for numbers ,alphabets,special characters 
'''
numbers="0123456789"
alphabetes="abcdefghijklmnopqrstuvwxyz"
special_char= " $%&'()*+,-./:;<=>?@[\]^_`{|}~!"
special_char=special_char+'"'
characters=numbers+alphabetes+special_char

no_characters=len(numbers)+len(alphabetes)+len(special_char)
max_len=140
#^^tweets max length

#a dictionary for indicating different characters with a increasing counter
char_dict=dict()
for i in range(len(characters)):
	char_dict.update( { characters[i]:i } )

'''
#opeing of files the list of entity and hashtags
#creating a dictionary with entity as keys
'''
list_read= open("final_entity_hashtag_list.txt", "r")
list_list = list_read.read().split("\n")
list_list = list_list[:-1]

entity_dict =dict()
counter=0
for x in list_list:
	entity_dict.update({x:counter})
	counter=counter+1

#dict of labels
label_dict=dict()
label_dict.update({"false":0})
label_dict.update({"true":1})
label_dict.update({"unverified":2})
label_dict.update({"non-rumor":3})

#accesing the file having only twweets
tweet_read= open("only_tweets", "r")
tweets= tweet_read.read().split("\n")
tweets= tweets[:-1]

######################################################################3
#opening file of ids with their tweets
with codecs.open("tweets", "r",encoding='utf-8', errors='ignore') as fdata:
	lines = fdata.read().split("\n")

#division of first data in two columns
ID = ["" for x in range(len(lines))]
TWEET = ["" for x in range(len(lines))]
tweet_dict=dict()
#preprocessing of first data
ii=0
for l in lines:
	sr= l
	if not sr.strip():
        	continue
	else:
		sr = sr.split("\t")
		if len(sr)==2:
			ID[ii]=sr[0]
			TWEET[ii]=sr[1]
			ii=ii+1
		else :
			TWEET[ii-1]=TWEET[ii-1]+' '+l
file1= open("set of tweets", "w")

for i in range(ii):
	url=r'[a-z]*[:.]+\S+'
	TWEET[i]=re.sub(url,'',TWEET[i].lower())
	tweet_dict.update({ID[i]:TWEET[i]})
	file1.write(TWEET[i])
	file1.write('\n')
file1.close()

########################################################################
'''
# A funtion is defined taking parameters and then return
  the required matrix for the tweet and its respective label
'''
def generator(tweets,entity_dict,batch_size,char_dict):
	#for 2 d
	#batch_features = np.zeros((batch_size, no_characters, max_len, 1))
	#for 1 d
	batch_features = np.zeros((batch_size, no_characters, max_len))
	batch_labels = np.zeros((batch_size,len(list_list)))
 	
	while True:
		for i in range(batch_size):
			index= random.randint(0,len(tweets))
	   		#this matrix store matrix one hot representation of every tweet
			tweet=tweets[index]
			tweet=tweet.lower()
			#2d
			#tweet_matrix=np.zeros((no_characters,max_len,1))
			#1d
			tweet_matrix=np.zeros((no_characters,max_len))
			k=0
			for j in range(len(tweet)):
				ch=tweet[j]
				if characters.find(ch)!=-1:
					tweet_matrix[char_dict[ch],k]=1
					k=k+1
			label_matrix=np.zeros((len(list_list)))
			for key,value in entity_dict.items():
				if tweet.find(key)!=-1:
					label_matrix[entity_dict[key]]=1
			batch_features[i]= tweet_matrix
			batch_labels[i] = label_matrix
			yield batch_features, batch_labels

#opening all ids with their labels
data_open= open("id data", "r")
id_data = data_open.read().split("\n")
id_data=id_data[:-1]

def generator_gru(id_data,tweet_dict,char_dict,label_dict):
	while True:
		for k in range(1):
			index= random.randint(0,len(id_data))
	   		#this matrix store matrix one hot representation of every tweet
			id_value=id_data[index].split(":")
			
			read_tree= open("/media/code_drunk/5C7B9D870AB33716/backup/twitter_final /data/twitter15/tree/"+temp, "r")
			edge = hu.read().split("\n")
			edge=edge[:-1]

			unique =dict()
			#traversing the data 
			ii=0
			for e in edge:
				li = e
				li=li.split("'")[1::2]
				if ii==0:
					temp=li[4]
				else:
					if li[4]!=temp:
						unique.update( { li[4]:float(li[5]) } )
					if li[1]!=temp:
						unique.update( { li[1]:float(li[2]) } )
				ii=ii+1

			#organising the data in a good manner
			#final char_dict
			unique1 = sorted(unique.items(), key=itemgetter(1))

			tweets= ["" for x in range(len(unique1))]
			i=0
			for keys,values in unique1.items():
				tweets[i]=unique1[key]
				i=i+1

			batch_features = np.zeros((len(unique1), 1024, 1))
			batch_labels = np.zeros((len(unique1),4))

			i=0
			for tweet in range(tweets):
				test_case=tweets[tweet]
				test_case=test_case.lower()

				#1d
				tweet_matrix=np.zeros((no_characters,max_len))
				k=0
				for j in range(len(test_case)):
					ch=test_case[j]
					if characters.find(ch)!=-1:
						tweet_matrix[char_dict[ch],k]=1
						k=k+1

				label_matrix=np.zeros((4))
				label_matrix[label_dict[id_value[1]]]=1

				batch_features[i]= tweet_matrix
				batch_labels[i] = label_matrix
				i=i+1
				yield batch_features, batch_labels

#######################################################################			
#batch_size to train
batch_size = 64
# number of output classes
n_classes = len(list_list)
# number of epochs to train
nb_epoch = 100
# number of convolutional filters to use
nb_filters = 128
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4

model = Sequential()
model.add(Conv1D(nb_filters, nb_conv, activation='tanh', input_shape=(no_characters,max_len)))
model.add(Conv1D(nb_filters, nb_conv,  activation='tanh'))

model.add(MaxPooling1D(2))

model.add(Conv1D(nb_filters, nb_conv, activation='tanh'))

model.add(MaxPooling1D(2))

model.add(Flatten())

model.add(Dense(1024, activation='tanh'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='tanh'))
model.add(Dropout(0.25))
model.add(Dense(n_classes, activation='softmax'))
#compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#fitting the model
model.fit_generator(generator(tweets,entity_dict,batch_size,char_dict), samples_per_epoch=64, steps_per_epoch = 8,nb_epoch=nb_epoch)

#evaluating the test cases
scores=model.evaluate_generator(generator(tweets,entity_dict,batch_size), 10)
print("Evaluate : ",scores)
#predicting the test cases

#input of the gru from the output from CNN
gru_input = model.layers[-2]

gru_nb_epoch=100

#gru model
gru_model = Sequential()
gru_model.add(GRU(4,gru_input,return_sequences=u)
gru_model.add(Dense(4))
#compiling
gru_model.compile(optimizer='adam',loss=categorical_crossentropy,metrics=['accuracy'])
#fiiting the model
gru_model.fit_generator(generator_gru(id_data,tweet_dict,char_dict,label_dict),samples_per_epoch=64, steps_per_epoch = 8,nb_epoch=gru_nb_epoch))
#evaluatating
scores_gru=gru_model.evaluate_generator(generator_gru(id_data,tweet_dict,char_dict,label_dict), 10)
#predicting
gru_model.predict_generator(generator_gru(id_data,tweet_dict,char_dict,label_dict), 10)

###################################################
'''
model = Sequential()
model.add( Conv2D(32, (3, 3), padding='same',activation='relu', input_shape=(no_characters,max_len,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, (4, 4), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(128,(5,5), padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256,(6,6),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=OPTIM)
model.fit_generator(generator(tweets,entity_dict,batch_size), samples_per_epoch=100, steps_per_epoch = 10,nb_epoch=10)


#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

scores=model.evaluate_generator(generator(tweets,entity_dict,batch_size), 1000)
print("Evaluate : ",scores)
#model.fit_generator(generator(tweets,entity_dict,batch_size), samples_per_epoch=100, steps_per_epoch = 10,nb_epoch=10)
'''