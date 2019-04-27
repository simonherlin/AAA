import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

x_train = train_images
y_train = train_labels
x_test = test_images
y_test = test_labels

nb_img = 10061

num_train, img_rows, img_cols = x_train.shape
depth = 1
x_train = x_train.reshape(x_train.shape[0],
img_rows, img_cols, depth)
x_test = x_test.reshape(x_test.shape[0],
img_rows, img_cols, depth)
input_shape = (img_rows, img_cols, depth)

nb_filters = 500
# pooling size
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

nb_classes = 10
batch_size = 16
nb_epoch = 50





# model = keras.Sequential([
# # Le modèle Sequential est un ensemble linéaire de couches
# keras.layers.Flatten(input_shape=(28,28)),
# # Transforme une matrice 28x28 en un tableau de 784
# keras.layers.Dense(512, activation=tf.nn.relu),
# keras.layers.Dense(256, activation=tf.nn.relu),
# keras.layers.Dense(128, activation=tf.nn.relu),
# keras.layers.Dense(64, activation=tf.nn.relu),
# keras.layers.Dense(32, activation=tf.nn.relu),
# # Couche entièrement connectée de 128 neurones
# keras.layers.Dense(10, activation=tf.nn.softmax)
# # Couche entièrement connectée de 10 neurones:
# # 10 probabilités de sortie
# ])


# model.compile(optimizer='adam',
# # On choisit la descente de gradient
# # stochastique commme optimisation
# loss='sparse_categorical_crossentropy',
# # Définition de la mesure de perte
# # Ici l'entropie croiée
# metrics=['accuracy']
# # Définition de la mesure de performance
# # que l'on souhaite utiliser. Ici la accuracy
# )

#model.fit(train_images, train_labels, epochs=10)

model = keras.Sequential()
model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size=kernel_size,activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()


model.fit(x_train, y_train, batch_size=batch_size,
epochs=nb_epoch, verbose=1,
validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("perte: {}, accuracy: {}".format(
#test_loss, test_acc))

