
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()   #Split the dataset into test and train data.

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)   #Normalizes a Numpy array and axis gives the axis along which we normalize.

print(x_train[1].shape)


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     #input layer
model.add(tf.keras.layers.Dense(128,activation='relu'))     #hidden layer
model.add(tf.keras.layers.Dense(128,activation='relu'))     #hidden layer
model.add(tf.keras.layers.Dense(10,activation='softmax'))   #output neuron

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])     #compiling the neural network

model.fit(x_train,y_train, epochs=50)   #train the model.

model.save('handwritten.model')         #saving the model



#--------------------------------------------------------------------------------------------------#
model=tf.keras.models.load_model('handwritten.model')   #loading the saved model
#loss,accuracy=model.evaluate(x_test,y_test)             #calculating accuracy and loss of data
# print(loss)
# print(accuracy)


img=cv.imread("D:\\pythonProject\\IBM_digit_recognizer\\images2\\digit14.png")[:,:,0]   #loading the image saved for testing
print(img.shape)    #checking the shape(pixel ratio) of the testing image
# plt.imshow(img,cmap=plt.cm.binary)
img=np.invert(np.array([img]))  #inverting image colors

prediction=model.predict(img)   #prediction
#print(prediction)
print("The digit is ",np.argmax(prediction))    #np.argmax returns the maximum value found from the prediction array
# plt.imshow(img[0],cmap=plt.cm.binary)         #displaying the test image
# plt.imshow(x_train[1],cmap=plt.cm.binary)     #sample data from dataset
plt.show()

