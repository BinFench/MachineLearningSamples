import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def create_model():
    #Creating the multilayer perceptron
    model = Sequential()
    #Input is 4 values, connecting to a densely connected layer of 4 neurons
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    #Output of hidden layer is ReLU activated and given to single neuron.
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #Output of network is sigmoid activated.  Binary Cross Entropy is loss
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Load mass data, declare names
masses_data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'],
names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
			  
#Drop rows with missing data
masses_data.dropna(inplace=True)

#Convert data into numpy arrays
all_features = masses_data[['age', 'shape', 'margin', 'density']].values
all_classes = masses_data['severity'].values
feature_names = ['age', 'shape', 'margin', 'density']


#Normalize input data for network use
scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

#Put the classifier into an estimator to gauge performance.
estimator = KerasClassifier(build_fn=create_model, epochs=50, verbose=0)
cv_scores = cross_val_score(estimator, all_features_scaled, all_classes, cv=10)
print(cv_scores.mean())