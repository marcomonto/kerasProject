import tensorflow as tf
from tensorflow.keras import models, layers


def create_basic_model(input_shape: int) -> tf.keras.Sequential:
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


def create_funnel_model(input_shape: int) -> tf.keras.Sequential:
    layer_sizes = []
    current_size = input_shape
    percentages = [0.75, 0.50, 0.25, 0.125, 0.06]

    for percentage in percentages:
        current_size = max(1, int(current_size * percentage))
        layer_sizes.append(current_size)

    model = models.Sequential()

    model.add(layers.Dense(
        layer_sizes[0],
        activation='relu',
        input_shape=(input_shape,)
    ))

    for size in layer_sizes[1:]:
        model.add(layers.Dense(size, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model
