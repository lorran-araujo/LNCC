import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
import matplotlib.pyplot as plt

# Carrega o Oxford-IIIT Pet Dataset com informações de segmentação
dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)

IMG_SIZE = 128  # Tamanho das imagens de entrada

def preprocess_data(data):
    image = tf.image.resize(data['image'], (IMG_SIZE, IMG_SIZE))
    mask = tf.image.resize(data['segmentation_mask'], (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0  # Normaliza a imagem
    mask = tf.cast(mask, tf.int32) - 1  # Ajusta as classes da máscara
    return image, mask

def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder (Downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder (Upsampling)
    u1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u1 = concatenate([u1, c3])  # c3 deve ser ajustado para ter a mesma dimensão que u1
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = concatenate([u2, c2])  # c2 deve ser ajustado para ter a mesma dimensão que u2
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = concatenate([u3, c1])  # c1 deve ser ajustado para ter a mesma dimensão que u3
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    # Camada final de saída
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    return model


#Instanciando o modelo
input_shape = (128, 128, 3)  # Pode ser alterado conforme a resolução da base de dados
model = unet_model(input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preparando os dados
train_data = dataset['train'].map(preprocess_data).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = dataset['test'].map(preprocess_data).batch(32).prefetch(tf.data.AUTOTUNE)

# Treinando a U-Net
model.fit(train_data, epochs=10, validation_data=test_data)

def display_sample(display_list):
    plt.figure(figsize=(15, 15))
    titles = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Pegando uma amostra para testar
for image, mask in test_data.take(1):
    pred_mask = model.predict(image)
    display_sample([image[0], mask[0], tf.argmax(pred_mask[0], axis=-1)])