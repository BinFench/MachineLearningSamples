import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist #Load handwritten digits

#Unpack MNIST data into training and testing groups.
(train_images_2d, train_labels_n),(test_images_2d, test_labels_n) = mnist.load_data()

#Reshape images from 2D bitmaps to One-Hot format
train_images = train_images_2d.reshape(train_images_2d.shape[0], 784)
test_images = test_images_2d.reshape(test_images_2d.shape[0], 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
#Data in range of 0.0-1.0 for neural net
train_images /= 255
test_images /= 255

#Turning labels into softmax outputs
train_labels = tf.keras.utils.to_categorical(train_labels_n, 10)
test_labels = tf.keras.utils.to_categorical(test_labels_n, 10)

def display_sample(num):
    #Print the one-hot array of this sample's label 
    print(train_labels[num])  
    #Print the label converted back to a number
    label = train_labels[num].argmax(axis=0)
    #Reshape the 784 values to a 28x28 image
    image = train_images[num].reshape([28,28])
    plt.title('Sample: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
	
def next_batch(num, data, labels):
    #Generate a random sampling of data to train via batch
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
	
display_sample(0)

#Displaying the first 500 samples as their own rows in one hot format, for visual clarity
images = train_images[0].reshape([1,784])
for i in range(1, 500):
    images = np.concatenate((images, train_images[i].reshape([1,784])))
plt.imshow(images, cmap=plt.get_cmap('gray_r'))
plt.show()

#Defining input and output (one hot in, classifier confidence out)
input_images = tf.placeholder(tf.float32, shape=[None, 784])
target_labels = tf.placeholder(tf.float32, shape=[None, 10])

#The hidden layer has 512 nodes
hidden_nodes = 512

#Defining our first unknowns, the weights and biases in the input layer
input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))

#Weights and biases in the hidden layer
hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))

#Matrix multiplier, followed by ReLU activation, then matrix multiply again
#The layer weights are computed with the layer, passed to the next layer, then the layer biases are added.
input_layer = tf.matmul(input_images, input_weights)
hidden_layer = tf.nn.relu(input_layer + input_biases)
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

#Cross entropy loss function, to severely mitigate errors
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit_weights, labels=target_labels))
#Gradient descent is the algorithm used to train.
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#Finds the most confident digit prediction in softmax and compares with the label to check correctness
correct_prediction = tf.equal(tf.argmax(digit_weights,1), tf.argmax(target_labels,1))
#Tracks accuracy of current model.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) #Avoid GPU errors on Windows

#Begin training
sess.run(tf.global_variables_initializer())

#Run 2000 epochs, output every 100th epoch
for x in range(2000):
    (x_batch, y_batch) = next_batch(100, train_images, train_labels)
    sess.run(optimizer, feed_dict={input_images: x_batch, target_labels: y_batch})
    if (x % 100 == 0):
        print("Training epoch " + str(x+1))
        print("Accuracy: " + str(sess.run(accuracy, feed_dict={input_images: test_images, target_labels: test_labels})))
		
#After training, output all the mispredicted images in the first 200 images
for x in range(200):
    # Load a single test image and its label
    x_train = test_images[x,:].reshape(1,784)
    y_train = test_labels[x,:]
    # Convert the one-hot label to an integer
    label = y_train.argmax()
    # Get the classification from our neural network's digit_weights final layer, and convert it to an integer
    prediction = sess.run(digit_weights, feed_dict={input_images: x_train}).argmax()
    # If the prediction does not match the correct label, display it
    if (prediction != label) :
        plt.title('Prediction: %d Label: %d' % (prediction, label))
        plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
        plt.show()