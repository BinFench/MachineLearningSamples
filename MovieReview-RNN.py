from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb

#Load model data.  x is indices for words, 7 is a binary for good or bad review.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)

#Cut the RNN size to 100 words.
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

#This model has an embedding layer to make our text data uniform.
#Followed by a Long Short Term Memory cell with dropout, then a Dense layer.
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
#Binary classifier
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=15, verbose=2, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)