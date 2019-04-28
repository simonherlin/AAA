from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import matplotlib.pyplot as plt

K.tensorflow_backend._get_available_gpus()
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = './simpsons_dataset/'

epochs = 30
batch_size = 16

input_shape = (img_width, img_height, 3)
nb_img = 10061
kernel_size = (3, 3)
pool_size = (2, 2)

"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))
"""

model = Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=kernel_size,activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(64, kernel_size=kernel_size,activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Conv2D(128, kernel_size=kernel_size,activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Conv2D(256, kernel_size=kernel_size,activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
#model.add(keras.layers.Dense(256, activation='relu'))
#model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(32, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(16, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(8, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_img // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= nb_img // batch_size)


# loss_curve = history.history["loss"]
# acc_curve = history.history["accuracy"]
#
# loss_val_curve = history.history["val_loss"]
# acc_val_curve = history.history["val_accuracy"]
#
# plt.plot(loss_curve, label="Train")
# plt.plot(loss_val_curve, label="Val")
# plt.legend(loc='upper left')
# plt.title("Loss")
# plt.show()
#
# plt.plot(acc_curve, label="Train")
# plt.plot(acc_val_curve, label="Val")
# plt.legend(loc='upper left')
# plt.title("Accuracy")
# plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate_generator(generator=validation_generator,
                                 steps=STEP_SIZE_VALID)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

