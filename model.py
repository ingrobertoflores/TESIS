from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import os
from datetime import datetime

import settings
import utils


def get_model(nuevo=False):
    if nuevo:
        layers = tf.keras.layers
        mdl = tf.keras.models.Sequential([
            # Convolucional 5x5 x128 y64 z48
            layers.Conv2D(48, (5, 5), activation='relu', input_shape=settings.INPUT_SHAPE),
            # Maxpool 2x2 x64 y32 z48
            layers.MaxPooling2D((2, 2)),
            # Convolucional 5x5 x64 y32 z64
            layers.Conv2D(48, (5, 5), activation='relu'),
            # Maxpool 1x2 x64 y16 z64
            layers.MaxPooling2D((1, 2)),
            # Convolucional 5x5 x64 y16 z128
            layers.Conv2D(128, (5, 5), activation='relu'),
            # Maxpool 2x2 x32 y8 z128
            layers.MaxPooling2D((2, 2)),
            # Flatten model
            layers.Flatten(),
            # Fullyconnected 1024
            layers.Dense(1024, activation='relu'),
            # Fullyconnected 850
            layers.Dense(850, activation='relu'),
            # Fullyconnected 567
            layers.Dense(567, activation='sigmoid'),
            # Fullyconnected 378
            layers.Dense(378, activation='sigmoid'),
            # Fullyconnected 252
            layers.Dense((settings.VALID_PLATE_SIZE * settings.SINGLE_OUTPUT_SIZE),
                         activation='relu'),
            layers.Reshape((settings.VALID_PLATE_SIZE, settings.SINGLE_OUTPUT_SIZE))
        ])

        mdl.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['mae', 'mse', 'acc']
        )
        return mdl
    else:
        model_dir = settings.ARCHIVO_MODELO
        if os.path.exists(model_dir):
            utils.copytree(model_dir, "%s.%s" % (
                model_dir,
                datetime.now().strftime("%Y.%m.%d.%H.%M.%S")))
            return tf.keras.models.load_model(model_dir)
        else:
            return get_model(True)


model = get_model()

model.summary()