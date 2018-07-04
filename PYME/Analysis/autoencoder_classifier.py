def gen_autoencoder():
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
    from keras.models import Model

    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (7, 7, 32)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    #decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    return autoencoder


def gen_training_data(model, data, labels):
    _, kernel_size_x, kernel_size_y, _ = model.input_shape
    
    
    for i_x in range(0, data.shape[0] - kernel_size_x, self.step_size):
        for i_y in range(0, out.shape[1] - kernel_size_y, self.step_size):
            d_i = data[i_x:(i_x + kernel_size_x), i_y:(i_y + kernel_size_y)].reshape(1, kernel_size_x, kernel_size_y, 1)
            p = self._model.predict(d_i).reshape(kernel_size_x, kernel_size_y)
            out[i_x:(i_x + kernel_size_x), i_y:(i_y + kernel_size_y)] += scale_factor * p


def fit_model(model, train_data, train_labels):
    model.fit(train_data, train_labels,
                    epochs=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy[:500], x_test[:500]),
                    )#callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
    