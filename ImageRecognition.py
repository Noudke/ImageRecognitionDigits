import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from keras import layers
from keras.models import Sequential, Functional


# Load train.csv and test.csv
train = np.loadtxt('train.csv', skiprows=1, delimiter=',')
test = np.loadtxt('test.csv', skiprows=1, delimiter=',')

# Split train.csv into train and validation
train_images = train[:, 1:]
train_labels = train[:, 0]
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images / 255.0
validation_images = train_images[:4200]
validation_labels = train_labels[:4200]


def sequential_model():
    # Create the model
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=20, validation_data=(validation_images, validation_labels))

    # Plot the accuracy and loss
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    return model


def functional_model():
    # Create the model
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = Functional(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=5, validation_data=(validation_images, validation_labels))

    # Plot the accuracy and loss
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    return model


def convolutional_model():
    # Create the model
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=5, validation_data=(validation_images, validation_labels))

    # Plot the accuracy and loss
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    return model


sequential = sequential_model()
functional = functional_model()
convolutional = convolutional_model()

# Create predictions for test.csv
test_images = test.reshape((test.shape[0], 28, 28, 1))
test_images = test_images / 255.0
predictions = sequential.predict(test_images)

# Save predictions to a csv file
with open('submission.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for i, p in enumerate(predictions):
        f.write('{},{}\n'.format(i+1, np.argmax(p)))

