from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, 
    BatchNormalization, Activation, Dropout, Concatenate
)

class AMUNet:
    def __init__(self, input_shape=(96, 128, 1), weights_path=None):
        # Conv2D: (batch_size, height, width, channels), so the input dimension should be (height, width, channels)
        """
        Initialize the AM-UNet model.

        Parameters:
            input_shape (tuple): Input shape of the model.
            weights_path (str): Path to the pre-trained weights file (optional).
        """
        self.input_shape = input_shape
        self.model = self._build_model()

        if weights_path:
            self.model.load_weights(weights_path)
            print(f"Weights loaded from {weights_path}")

    def _conv_block(self, inputs, filters, layer_name, kernal_size=(3, 3)):
        """
        Convolutional block with one Conv2D layer, BatchNormalization, and Activation.
        """
        # Conv2D: (batch_size, height, width, channels)
        x = Conv2D(filters, kernal_size, padding="same", name=f"{layer_name}_conv")(inputs)
        x = BatchNormalization(name=f"{layer_name}_batchnorm")(x)
        x = Activation("relu", name=f"{layer_name}_activation")(x)
        return x

    def _build_model(self):
        """
        Build the AM-UNet architecture based on the structure from the .h5 file.
        """
        inputs = Input(self.input_shape, name="img")

        # Encoder
        c1_1 = self._conv_block(inputs, 32, "conv2d_1")
        c1_2 = self._conv_block(inputs, 32, "conv2d_2", kernal_size=(1, 1))
        c1 = Concatenate(name="concatenate_1")([c1_1, c1_2])
        p1 = MaxPooling2D((2, 2), name="max_pooling2d_1")(c1)
        p1 = Dropout(0.1, name="dropout_1")(p1)

        c2_1 = self._conv_block(p1, 64, "conv2d_3")
        c2_2 = self._conv_block(p1, 64, "conv2d_4", kernal_size=(1, 1))
        c2 = Concatenate(name="concatenate_2")([c2_1, c2_2])
        p2 = MaxPooling2D((2, 2), name="max_pooling2d_2")(c2)
        p2 = Dropout(0.2, name="dropout_2")(p2)

        # Bottleneck
        b1 = self._conv_block(p2, 32, "conv2d_5")
        b2 = self._conv_block(p2, 32, "conv2d_6", kernal_size=(1, 1))
        b = Concatenate(name="concatenate_3")([b1, b2])
        b = Dropout(0.3, name="dropout_3")(b)

        # Decoder
        u2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", name="conv2d_transpose_1")(b)
        merge2 = Concatenate(name="concatenate_4")([u2, c2])
        d2_1 = self._conv_block(merge2, 64, "conv2d_7")
        d2_2 = self._conv_block(merge2, 64, "conv2d_8", kernal_size=(1, 1))
        d2 = Concatenate(name="concatenate_5")([d2_1, d2_2])
        d2 = Dropout(0.2, name="dropout_4")(d2)

        u1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", name="conv2d_transpose_2")(d2)
        merge1 = Concatenate(name="concatenate_6")([u1, c1])
        d1_1 = self._conv_block(merge1, 32, "conv2d_9")
        d1_2 = self._conv_block(merge1, 32, "conv2d_10", kernal_size=(1, 1))
        d1 = Concatenate(name="concatenate_7")([d1_1, d1_2])
        d1 = Dropout(0.1, name="dropout_5")(d1)

        # Final output layers
        output1 = Conv2D(2, (3, 3), activation=None, padding="same", name="conv2d_11")(d1)
        output1 = BatchNormalization(name="batch_normalization_11")(output1)
        output1 = Activation("relu", name="activation_11")(output1)

        output2 = Conv2D(2, (1, 1), activation=None, padding="same", name="conv2d_12")(d1)
        output2 = BatchNormalization(name="batch_normalization_12")(output2)
        output2 = Activation("relu", name="activation_12")(output2)

        final_concat = Concatenate(name="concatenate_8")([output1, output2])
        outputs = Conv2D(1, (1, 1), activation="sigmoid", name="conv2d_13")(final_concat)

        return Model(inputs, outputs)

    def predict(self, data):
        """
        Predict on input data.

        Parameters:
            data (numpy.ndarray): Input data for prediction.

        Returns:
            predictions (numpy.ndarray): Predictions from the model.
        """
        return self.model.predict(data)

if __name__ == "__main__":
    from tensorflow.keras.optimizers import Adam

    # Initialize the model
    am_unet = AMUNet(input_shape=(96, 128, 1))

    # Compile the model
    am_unet.model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    # Print the model summary
    am_unet.model.summary()