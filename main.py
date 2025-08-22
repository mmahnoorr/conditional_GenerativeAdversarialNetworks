import tensorflow as tf

import os
import pathlib
import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

#set the path to the directory where the train, val, and test data exist

PATH = "Data Moj"

sample_image = tf.io.read_file(PATH + '/trainingData-/combined_104.jpg')
sample_image = tf.io.decode_jpeg(sample_image)
print(sample_image.shape)

plt.figure()
plt.imshow(sample_image)

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

inp, re = load(PATH +'/trainingData-/combined_104.jpg')
# Casting to int for matplotlib to display the images
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)

# The training set consist of 1680 images
BUFFER_SIZE = 2000
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 1])

  return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

plt.figure(figsize=(6, 6))
rj_inp, rj_re = resize(inp, re,IMG_HEIGHT, IMG_WIDTH)
plt.imshow(rj_inp / 255.0)
plt.axis('off')
plt.show()
plt.imshow(rj_re / 255.0)
plt.axis('off')
plt.show()
print(np.shape(rj_inp), np.shape(rj_re))

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

#trainset contains 1643 samples
train_dataset = tf.data.Dataset.list_files(PATH+'/trainingData-/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

## here we build the test (480 samples) and validation (240 samples) sets.
#Please note that in the data folder, we have about 600 Totally_Unseen dataset which I beleive we can use for creating totally non-existing cases.
#my idea is that we can combine the white matter mask from healthy cases and lesion masks from the abnormal cases to create a new mask and create new inputs.
# By combining the masks from healthy and nonhealthy cases, we create a new subject and let the model to synthesize a new brain for us.
#Using the synthetic images you can train a new classifier or mix them with real images and retrain the classifier.
try:
 val_dataset = tf.data.Dataset.list_files(PATH+'/validation/*.jpg')
except tf.errors.InvalidArgumentError:
 val_dataset = tf.data.Dataset.list_files(PATH+'/validation/*.jpg')
val_dataset = val_dataset.map(load_image_test)
val_dataset = val_dataset.batch(BATCH_SIZE)

try:
 test_dataset = tf.data.Dataset.list_files(PATH+'/testingData-/*.jpg')
except tf.errors.InvalidArgumentError:
 test_dataset = tf.data.Dataset.list_files(PATH+'/testingData-/*jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

from tensorflow.keras.utils import plot_model

generator = Generator()
plot_model(generator, to_file='generatorrr_model.png', show_shapes=True, dpi=64)

from tensorflow.keras.utils import plot_model
from keras.models import Model  # or your model class
import os

# Assuming your model is called generator
generator = Generator()

# Plot horizontally
plot_model(generator,
           to_file='generator_horizontal.png',
           show_shapes=True,
           dpi=96,
           rankdir='LR')  # Set direction to Left to Right

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss
  print(total_disc_loss)

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#make sure to provide a directory so save your checkpoints. You need them to restore your models. Alternatively, you can save your model as .h5 somewhere on your disk space.
checkpoint_dir = './partallneg_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 5 + 5)
    plt.axis('off')
  plt.show()

 for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target)

  
log_dir="logsallneg/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
    
  return(gen_gan_loss, disc_loss)




generator_gan_losses = []

discriminator_losses = []


def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    gen_gan_loss, disc_loss = train_step(input_image, target, step)
    generator_gan_losses.append(gen_gan_loss.numpy())  # Storing discriminator loss
    discriminator_losses.append(disc_loss.numpy())  # Storing discriminator loss

    
    
    
    if (step+1) % 10 == 0:
      print(f'Step {step+1}: Gen Gan Loss = {gen_gan_loss.numpy()}, Discriminator Loss = {disc_loss.numpy()}')

    if (step + 1) % 20000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)    


%load_ext tensorboard
%tensorboard --logdir {log_dir}

  #steps= number of epochs x total number of training cases ==> for example 50 x 1643 = 82150 >> this means you need to set your steps to 82150 if you want to run your model for 50 epochs.
fit(train_dataset, val_dataset, steps=100000) ## change the number of steps to 82150 if you want 50 epochs.

import matplotlib.pyplot as plt

# Plotting the generator and discriminator loss
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_gan_losses, label="generator_gan_losses")
plt.plot(discriminator_losses, label="Discriminator loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()

!ls {checkpoint_dir}

# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

 import tensorflow as tf

# Assume the models are defined as functions or imported from your model script
generator = Generator()
discriminator = Discriminator()

# Create a checkpoint instance
checkpoint_dir = '/partallneg_checkpoints'
checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)

# Restore the latest checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Checkpoint restored successfully from {}".format(latest_checkpoint))
else:
    print("No checkpoint found at {}".format(checkpoint_dir))

 # Save the entire model to a file
generator.save('generatorallneg_model.h5')
discriminator.save('discriminatorallneg_model.h5')
print("Models saved to disk.")

# Later, load the models back
#loaded_generator = tf.keras.models.load_model('generator_model.h5')
#loaded_discriminator = tf.keras.models.load_model('discriminator_model.h5')
#print("Models loaded from disk.")

# Run the trained model on a few examples from the test set.
#Also change the number_of_samples to 480 if you want to test all cases in the test folder.
#You must modify the generate_images function to save your predictions as jpg files.
number_of_samples=1000
for inp, tar in test_dataset.take(number_of_samples):
  generate_images(generator, inp, tar)

import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Directory to save generated images
output_directory = "partallneg50imagesss_generated"
os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists

def generate_and_save_images(model, test_input, tar, idx):
    prediction = model(test_input, training=True)
    # Normalize the pixel values to [0, 255]
    prediction = (prediction[0] * 0.5 + 0.5) * 255.0
    prediction = tf.cast(prediction, tf.uint8)

    # Save the generated image as JPEG
    save_path = os.path.join(output_directory, f"generated_image_{idx + 1}.jpg")
    encoded_image = tf.image.encode_jpeg(prediction)
    tf.io.write_file(save_path, encoded_image)
    print(f"Saved: {save_path}")

    # Optional: Display images
    plt.figure(figsize=(15, 15))
    # Cast prediction to float32 for consistent dtype operations
    prediction_float = tf.cast(prediction, tf.float32)
    display_list = [test_input[0] * 0.5 + 0.5, tar[0] * 0.5 + 0.5, prediction_float / 255.0]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()

# Running the modified function on test data
number_of_samples = 1000  # Number of samples to generate and save
for idx, (inp, tar) in enumerate(test_dataset.take(number_of_samples)):
    generate_and_save_images(generator, inp, tar, idx)

 import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Directory to save generated images
output_directory = "partallneg50imagesssallmerged_generated"
os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists

def generate_and_save_images(model, test_input, tar, idx):
    prediction = model(test_input, training=True)
    
    # Normalize pixel values to [0, 255] and convert to numpy arrays
    input_image = np.array((test_input[0].numpy() * 0.5 + 0.5) * 255.0, dtype=np.uint8)
    ground_truth = np.array((tar[0].numpy() * 0.5 + 0.5) * 255.0, dtype=np.uint8)
    prediction = np.array((prediction[0].numpy() * 0.5 + 0.5) * 255.0, dtype=np.uint8)
    
    # Ensure images are 2D grayscale
    def ensure_grayscale(image):
        if image.ndim == 3:
            return image[:, :, 0]  # Take the first channel if it has three
        return image
    
    input_image = ensure_grayscale(input_image)
    ground_truth = ensure_grayscale(ground_truth)
    prediction = ensure_grayscale(prediction)
    
    # Ensure all images have the same height and width
    height, width = input_image.shape[:2]
    ground_truth = np.array(Image.fromarray(ground_truth).resize((width, height)))
    prediction = np.array(Image.fromarray(prediction).resize((width, height)))
    
    # Stack images horizontally
    combined_image = np.hstack([input_image, ground_truth, prediction])
    
    # Save the combined image
    save_path = os.path.join(output_directory, f"comparison_{idx}.jpg")
    Image.fromarray(combined_image).convert("L").save(save_path)  # Save as grayscale
    print(f"Saved: {save_path}")

# Running the modified function on test data
number_of_samples = 1000  # Number of samples to generate and save
for idx, (inp, tar) in enumerate(test_dataset.take(number_of_samples), start=1):
    generate_and_save_images(generator, inp, tar, idx)

