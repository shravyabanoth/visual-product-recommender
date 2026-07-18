import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

IMG_SIZE = (224, 224)
EMBEDDING_DIM = 128


def build_embedding_model(embedding_dim=EMBEDDING_DIM):
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(*IMG_SIZE, 3),
    )

    for layer in base.layers[:-15]:
        layer.trainable = False

    x = base.output
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(embedding_dim)(x)

    model = Model(
        inputs=base.input,
        outputs=x,
        name="embedding_model",
    )

    return model


def build_siamese_triplet_model(embedding_model):
    anchor = layers.Input(shape=(*IMG_SIZE, 3), name="anchor")
    positive = layers.Input(shape=(*IMG_SIZE, 3), name="positive")
    negative = layers.Input(shape=(*IMG_SIZE, 3), name="negative")

    anchor_embedding = embedding_model(anchor)
    positive_embedding = embedding_model(positive)
    negative_embedding = embedding_model(negative)

    outputs = layers.Concatenate(axis=1)(
        [
            anchor_embedding,
            positive_embedding,
            negative_embedding,
        ]
    )

    model = Model(
        inputs=[anchor, positive, negative],
        outputs=outputs,
        name="siamese_triplet_model",
    )

    return model


def triplet_loss(margin=0.3):
    def loss(y_true, y_pred):
        embedding_dim = tf.shape(y_pred)[1] // 3

        anchor = y_pred[:, :embedding_dim]
        positive = y_pred[:, embedding_dim:2 * embedding_dim]
        negative = y_pred[:, 2 * embedding_dim:]

        positive_distance = tf.reduce_sum(
            tf.square(anchor - positive),
            axis=1,
        )

        negative_distance = tf.reduce_sum(
            tf.square(anchor - negative),
            axis=1,
        )

        losses = tf.maximum(
            positive_distance - negative_distance + margin,
            0.0,
        )

        return tf.reduce_mean(losses)

    return loss


if __name__ == "__main__":
    embedding_model = build_embedding_model()
    siamese_model = build_siamese_triplet_model(embedding_model)

    siamese_model.compile(
        optimizer="adam",
        loss=triplet_loss(),
    )

    siamese_model.summary()