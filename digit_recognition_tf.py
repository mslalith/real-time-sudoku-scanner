import tensorflow as tf
import numpy as np
import cv2
from scipy.io import loadmat
import datetime
import os

file_train_data = 'dataset\\train_32x32.mat'
file_test_data = 'dataset\\test_32x32.mat'


def process_data(x, y):
    to_remove = []
    for i in range(len(y)):
        if y[i] == 10:
            to_remove.append(i)
    x = np.delete(x, to_remove, axis=0)
    y = np.delete(y, to_remove)
    return x, y


def load_data():
    train = loadmat(file_train_data)
    test = loadmat(file_test_data)

    xtr = np.transpose(train['X'], (3, 0, 1, 2))
    xtr = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in xtr])
    ytr = train['y']
    ytr = ytr.flatten()
    (xtr, ytr) = process_data(xtr, ytr)

    xte = np.transpose(test['X'], (3, 0, 1, 2))
    xte = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in xte])
    yte = test['y']
    yte = yte.flatten()
    (xte, yte) = process_data(xte, yte)

    return (xtr, ytr), (xte, yte)


model_name = 'digit_classifier.h5'
IMAGE_SIZE = 32

(x_train, y_train), (x_test, y_test) = load_data()
# print(x_train.shape)        # (60000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(x_train[0].shape)     # (28, 28)
# print(y_train[0])           # 5

x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32') / 255
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

if os.path.exists(model_name):
    model = tf.keras.models.load_model(model_name)
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=6,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback]
    )
    loss, metrics = model.evaluate(x_test, y_test)
    print('Loss = {}, Metrics: {}'.format(loss, metrics))

    model.save(model_name)

image = x_test[np.random.randint(0, len(x_test))]
prediction = model.predict(image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1))
print(prediction.argmax())
cv2.imshow('image', image.reshape(IMAGE_SIZE, IMAGE_SIZE))
cv2.waitKey(0)
cv2.destroyAllWindows()
