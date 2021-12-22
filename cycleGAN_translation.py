# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:57:23 2020

@author: Asma Baccouche
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
autotune = tf.data.experimental.AUTOTUNE
from sklearn.model_selection import train_test_split
import glob
from PIL import Image

#identify GPU
device_name = tf.test.gpu_device_name()
if not tf.config.list_physical_devices('GPU'):
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.75)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
tf.compat.v1.keras.backend.set_session(sess)

size = 448
#mydata = glob.glob("MYDATA/Breast Cancer-Begonya/Images/Originales_cropped/*.png")
#mydata = glob.glob("/Patches/*png")
prior = glob.glob("D:/Yufeng_Data/Prior_augmented/*.jpg")
trn_prior, tst_prior = train_test_split(prior, test_size=0.1, random_state=42)
#cbis = glob.glob("D:/MALIG/*png")
current = glob.glob("D:/Yufeng_Data/Current_augmented/*.jpg")
trn_current, tst_current = train_test_split(current, test_size=0.1, random_state=42)
#trn_cbis = glob.glob("D:/CBIS-DDSM Train Mass/Train_Patches/*.png")
#tst_cbis = glob.glob("D:/CBIS-DDSM Test Mass/Test_Patches/*.png")

def generator1():
    #train_x_mydata = np.ndarray(shape=(len(trn_mydata), 256, 256, 3),dtype=np.float32)
    #train_y_mydata = np.ndarray(shape=(len(trn_mydata),),dtype=np.int16)
    for i in range(len(trn_prior)):
        img = Image.open(trn_prior[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        train_x_mydata = img
        train_y_mydata = np.array([0])
        train_name_mydata = np.array([trn_prior[i].split('\\')[-1]])
        yield train_x_mydata, train_y_mydata, train_name_mydata
    
def generator2():
    #test_x_mydata = np.ndarray(shape=(len(tst_mydata), 256, 256, 3),dtype=np.float32)
    #test_y_mydata = np.ndarray(shape=(len(tst_mydata),),dtype=np.int16)
    for i in range(len(tst_prior)):
        img = Image.open(tst_prior[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        test_x_mydata = img.resize((size,size), Image.NEAREST)
        test_y_mydata = np.array([0]) 
        test_name_mydata = np.array([tst_prior[i].split('\\')[-1]])
        yield test_x_mydata, test_y_mydata, test_name_mydata
    
def generator3():
    #train_x_cbis = np.ndarray(shape=(len(trn_cbis), 256, 256, 3),dtype=np.float32)
    #train_y_cbis = np.ndarray(shape=(len(trn_cbis),),dtype=np.int16)
    for i in range(len(trn_current)):
        img = Image.open(trn_current[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        train_x_cbis = img
        train_y_cbis = np.array([1])
        train_name_cbis = np.array([trn_current[i].split('\\')[-1]])
        yield train_x_cbis, train_y_cbis, train_name_cbis
    
def generator4():
    #test_x_cbis = np.ndarray(shape=(len(tst_cbis), 256, 256, 3),dtype=np.float32)
    #test_y_cbis = np.ndarray(shape=(len(tst_cbis),),dtype=np.int16)
    for i in range(len(tst_current)):
        img = Image.open(tst_current[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        test_x_cbis = img.resize((size,size), Image.NEAREST)
        test_y_cbis = np.array([1])
        test_name_cbis = np.array([tst_current[i].split('\\')[-1]])
        yield test_x_cbis, test_y_cbis, test_name_cbis
    
train_prior = tf.data.Dataset.from_generator(generator1, (tf.float32, tf.int16, tf.string))  
test_prior = tf.data.Dataset.from_generator(generator2, (tf.float32, tf.int16, tf.string))   

train_current = tf.data.Dataset.from_generator(generator3, (tf.float32, tf.int16, tf.string))  
test_current = tf.data.Dataset.from_generator(generator4, (tf.float32, tf.int16, tf.string))   

# Define the standard image size.
orig_img_size = (size, size)
# Size of the random crops to be used during training.
input_img_size = (size, size, 3)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = size
batch_size = 1


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label, name):
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    img = tf.image.resize(img, [*orig_img_size])
    # Random crop to 256X256
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label, name):
    # Only resizing and normalization for the test images.
    #img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img

# Apply the preprocessing operations to the training data
train_prior = (
    train_prior.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size))

train_current = (
    train_current.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size))

# Apply the preprocessing operations to the test data
test_prior = (
    test_prior.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size))

test_current = (
    test_current.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size))


buffer_size = size
batch_size = 1

class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model

def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")

class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity)
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity)

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables))

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables))
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables))

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }
        
# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

# Create cycle gan model
cycle_gan_model = CycleGan(generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)
# Callbacks
checkpoint_filepath = "cyclegan_log/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)

# Here we will train the model for just one epoch as each epoch takes around
# 7 minutes on a single P100 backed machine.
cycle_gan_model.fit(tf.data.Dataset.zip((train_prior, train_current)),initial_epoch=89,epochs=100,callbacks=[model_checkpoint_callback])

# Load the checkpoints
#weight_file = "cyclegan_log/cyclegan_checkpoints.089.index"
weight_file = "cyclegan_log/cyclegan_checkpoints.100"
cycle_gan_model.load_weights(weight_file).expect_partial()
print("Weights loaded successfully")

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, img in enumerate(test_prior.take(4)):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input image")
    ax[i, 1].set_title("Translated image")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")

    prediction = keras.preprocessing.image.array_to_img(prediction)
    #prediction.save("predicted_"+name.format(i=i))
plt.tight_layout()
plt.show()


def generator1():
    #train_x_mydata = np.ndarray(shape=(len(trn_mydata), 256, 256, 3),dtype=np.float32)
    #train_y_mydata = np.ndarray(shape=(len(trn_mydata),),dtype=np.int16)
    for i in range(len(trn_prior)):
        img = Image.open(trn_prior[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((size,size), Image.NEAREST)   
        train_x_mydata = (np.asarray(img) / 127.5) - 1.0
        train_y_mydata = np.array([0])
        train_name_mydata = np.array([trn_prior[i].split('\\')[-1]])
        yield train_x_mydata, train_y_mydata, train_name_mydata
    
def generator2():
    #test_x_mydata = np.ndarray(shape=(len(tst_mydata), 256, 256, 3),dtype=np.float32)
    #test_y_mydata = np.ndarray(shape=(len(tst_mydata),),dtype=np.int16)
    for i in range(len(tst_prior)):
        img = Image.open(tst_prior[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((size,size), Image.NEAREST)
        test_x_mydata = (np.asarray(img) / 127.5) - 1.0
        test_y_mydata = np.array([0]) 
        test_name_mydata = np.array([tst_prior[i].split('\\')[-1]])
        yield test_x_mydata, test_y_mydata, test_name_mydata
    
def generator3():
    #train_x_cbis = np.ndarray(shape=(len(trn_cbis), 256, 256, 3),dtype=np.float32)
    #train_y_cbis = np.ndarray(shape=(len(trn_cbis),),dtype=np.int16)
    for i in range(len(trn_current)):
        img = Image.open(trn_current[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((256,256), Image.NEAREST)
        train_x_cbis = (np.asarray(img) / 127.5) - 1.0
        train_y_cbis = np.array([1])
        train_name_cbis = np.array([trn_current[i].split('\\')[-1]])
        yield train_x_cbis, train_y_cbis, train_name_cbis
    
def generator4():
    #test_x_cbis = np.ndarray(shape=(len(tst_cbis), 256, 256, 3),dtype=np.float32)
    #test_y_cbis = np.ndarray(shape=(len(tst_cbis),),dtype=np.int16)
    for i in range(len(tst_current)):
        img = Image.open(tst_current[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((size,size), Image.NEAREST)
        test_x_cbis = (np.asarray(img) / 127.5) - 1.0
        test_y_cbis = np.array([1])
        test_name_cbis = np.array([tst_current[i].split('\\')[-1]])
        yield test_x_cbis, test_y_cbis, test_name_cbis
    
train_prior = tf.data.Dataset.from_generator(generator1, (tf.float32, tf.int16, tf.string))  
test_prior = tf.data.Dataset.from_generator(generator2, (tf.float32, tf.int16, tf.string))   

train_current = tf.data.Dataset.from_generator(generator3, (tf.float32, tf.int16, tf.string))  
test_current = tf.data.Dataset.from_generator(generator4, (tf.float32, tf.int16, tf.string))   

# Define the standard image size.
orig_img_size = (size, size)
# Size of the random crops to be used during training.
input_img_size = (size, size, 3)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = size
batch_size = 1

# Apply the preprocessing operations to the training data
train_prior = train_prior.cache().shuffle(buffer_size).batch(batch_size)
test_prior = test_prior.cache().shuffle(buffer_size).batch(batch_size)
train_current = train_current.cache().shuffle(buffer_size).batch(batch_size)
test_current = test_current.cache().shuffle(buffer_size).batch(batch_size)

  
for i, (img, _, name) in enumerate(test_prior.take(len(tst_prior))):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    prediction = keras.preprocessing.image.array_to_img(prediction)
    name = name[0].numpy()[0].decode("utf-8")
    prediction.save("D:/Yufeng_Data/Prior_to_Current/"+name)   
     
for i, (img, _, name) in enumerate(train_prior.take(len(trn_prior))):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    prediction = keras.preprocessing.image.array_to_img(prediction)
    name = name[0].numpy()[0].decode("utf-8")
    prediction.save("D:/Yufeng_Data/Prior_to_Current/"+name)
