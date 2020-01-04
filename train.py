from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt

import tensorflow as tf
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


from model import model as CNNMODEL
import utils
import settings


# FUENTE : https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def read_image(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=settings.IMG_CHANNELS)
    image_resized = tf.image.resize(image_decoded, (settings.IMG_WIDTH, settings.IMG_HEIGHT))
    image = tf.image.per_image_standardization(image_resized)
    # image_reshaped = tf.image.to
    # image_reshaped = tf.image.rgb_to_grayscale(image_resized)
    # image_reshaped = tf.expand_dims(image_resized, axis=0)
    # return image_reshaped, label
    return image, label


def prepare_dataset(csv: str) -> tuple:
    l = []
    with open(csv) as f:
        l += f.readlines()
    # ejemplo de linea D573_901.frontal_p398543.jpg.png
    return [os.path.join(settings.PLACAS_GENERADAS_TRANSFORMADAS_DIR, fn)[:-1] for fn in l],\
           [utils.get_word_matrix(fn[:8].replace('_', '')) for fn in l]


# train_images, train_labels = prepare_dataset(settings.TRAIN_CSV_DIR)
train_slices = prepare_dataset(settings.TRAIN_CSV_DIR)
dataset_train = tf.data.Dataset.from_tensor_slices(train_slices)
dataset_train = dataset_train.map(read_image)\
    .batch(settings.TRAIN_BATCH_SIZE)\
    .shuffle(buffer_size=settings.TRAIN_BATCH_SIZE, reshuffle_each_iteration=True)
print(dataset_train)

test_slices = prepare_dataset(settings.TEST_CSV_DIR)
dataset_test = tf.data.Dataset.from_tensor_slices(test_slices)
dataset_test = dataset_test.map(read_image)\
    .batch(settings.TEST_BATCH_SIZE)\
    .shuffle(buffer_size=settings.TEST_BATCH_SIZE, reshuffle_each_iteration=True)


# aprender como guardar
# https://www.tensorflow.org/beta/tutorials/keras/save_and_restore_models
# https://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/

print('================ TRAINING ================')
# CNNMODEL.fit(train_images, train_labels, epochs=5)
# CNNMODEL.fit(dataset_train, epochs=settings.EPOCHS, validation_data=dataset_test)
total_train = len(train_slices[0])
batch_size = settings.TRAIN_BATCH_SIZE

total_val = len(test_slices[0])
val_batch_size = settings.TEST_BATCH_SIZE

history = CNNMODEL.fit_generator(
    dataset_train,
    # steps_per_epoch=total_train // batch_size,
    epochs=settings.EPOCHS,
    validation_data=dataset_test,
    # validation_steps=total_val // val_batch_size
)
CNNMODEL.save(settings.ARCHIVO_MODELO)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(settings.EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#test_loss, test_acc = CNNMODEL.evaluate(test_images, test_labels)
#