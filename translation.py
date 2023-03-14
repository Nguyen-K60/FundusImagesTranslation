import tensorflow as tf
import pix2pix
import os
import time
from IPython.display import clear_output, display
from utils import *
tf.random.set_seed(1234)
np.random.seed(1234)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


TRAIN = False
checkpoint_path = "checkpoint"
MakeFolder(checkpoint_path)
EPOCHS = 100
TEST_ALL = True
IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.AUTOTUNE
# if TRAIN==False:
#   tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=6000)])
size = (IMG_HEIGHT, IMG_WIDTH)
# train_ufi, _ = load_dataset("data/masked_bad_op_spot_blur/", size)
# train_ofi, _ = load_dataset("data/good_op/", size)
def random_crop(input_image):
  stacked_image = tf.stack([input_image, input_image], axis=0)
  cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image[0]

def preprocess_image_train(image):
  image = random_crop(image)
  image = tf.image.random_flip_left_right(image)
  return image
# train_ufi = tf.image.random_flip_left_right(train_ufi)
# train_ofi = tf.image.random_flip_left_right(train_ofi)
# train_ufi = tf.image.rot90(train_ufi, k=1)
# rot90_ofi = tf.image.rot90(train_ofi, k=1)
# train_ofi = np.concatenate((train_ofi, rot90_ofi))
# train_ufi = tf.image.rot90(train_ufi, k=2)
# train_ofi = tf.image.rot90(train_ofi, k=2)

# train_ufi = tf.image.rot90(train_ufi, k=1)
# rot90_ofi = tf.image.rot90(train_ofi, k=1)
# train_ofi = np.concatenate((train_ofi, rot90_ofi))
# train_ufi = tf.image.rot90(train_ufi, k=2)
# train_ofi = tf.image.rot90(train_ofi, k=2)

# train_ufi = tf.image.random_contrast(train_ufi, 0.7, 1.3)
# train_ofi = tf.image.random_contrast(train_ofi, 0.7, 1.3)
batch_size = 1
if TRAIN==True:
  print('load train dataset')
  train_ufi, _ = load_dataset("/home/pham/Desktop/eyepacs/masked_enhanced_ufi_1000/", (286, 286))
  train_ofi, _ = load_dataset("/home/pham/Desktop/eyepacs/masked_slight/", (286, 286))
  # artifact reduction
  # train_ufi, _ = load_dataset("data/bad/", (286, 286))
  # train_ofi, _ = load_dataset("data/good/", (286, 286))
  num_samples = train_ufi.shape[0]//batch_size
  if train_ufi.shape[0] < train_ofi.shape[0]:
    num_samples = train_ofi.shape[0]//batch_size
  print('UFI: ', train_ufi.shape)
  print('OFI: ', train_ofi.shape)
  train_ufi = tf.data.Dataset.from_tensor_slices((train_ufi)).cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(buffer_size=train_ufi.shape[0], reshuffle_each_iteration=True).batch(batch_size)
  train_ofi = tf.data.Dataset.from_tensor_slices((train_ofi)).cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(buffer_size=train_ofi.shape[0], reshuffle_each_iteration=True).batch(batch_size)

if TRAIN==False and TEST_ALL==True:
  print('load test dataset')
  # test_ufi, names = load_dataset("data/masked_enhanced_new_ufi/", size)
  test_ufi, names = load_dataset("/home/pham/Desktop/eyepacs/masked_enhanced_ufi_test/", size)
  print('UFI: ', test_ufi.shape)
  
  batch_test=1
  test_ufi = tf.data.Dataset.from_tensor_slices((test_ufi)).batch(batch_test)
    # test_ofi = tf.data.Dataset.from_tensor_slices((test_ofi)).batch(batch_test)
  # else: 
  #   test_ufi, names = load_dataset("data/masked_enhanced_89/", size)
  #   test_ofi, _ = load_dataset("data/masked_enhanced_89/", size) 

  # print('UFI_test: ', test_ufi.shape)
  # print('OFI_test: ', test_ofi.shape)
    
OUTPUT_CHANNELS = 3

print('create network')
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
# generator_g = pix2pix.unet_generator_cbam(OUTPUT_CHANNELS, norm_type='instancenorm')
# generator_f = pix2pix.unet_generator_cbam(OUTPUT_CHANNELS, norm_type='instancenorm')
# generator_g = pix2pix.my_unet_generator(OUTPUT_CHANNELS)
# generator_f = pix2pix.my_unet_generator(OUTPUT_CHANNELS)
# tf.keras.utils.plot_model(generator_g, show_shapes=True, dpi=64)

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
# discriminator_x = pix2pix.discriminator_cbam(norm_type='instancenorm', target=False)
# discriminator_y = pix2pix.discriminator_cbam(norm_type='instancenorm', target=False)
# discriminator_x = pix2pix.custom_discriminator_cbam(norm_type='instancenorm', target=False)
# discriminator_y = pix2pix.custom_discriminator_cbam(norm_type='instancenorm', target=False)
# discriminator_x = pix2pix.my_discriminator(norm_type='instancenorm', target=False)
# discriminator_y = pix2pix.my_discriminator(norm_type='instancenorm', target=False)
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# loss_obj = tf.keras.losses.MeanSquaredError()
def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def generator_loss(generated):
    gan_loss = loss_obj(tf.ones_like(generated), generated)
    return gan_loss
def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    loss1 = LAMBDA * loss1
    return loss1
def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    loss = LAMBDA * 0.5 * loss
    return loss
print('create optimizer')
# class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

#   def __init__(self, initial_learning_rate, start_decay=100, numSteps=100):

#     super(MyLRSchedule, self).__init__()
#     self.initial_learning_rate = initial_learning_rate
#     self.start_decay = start_decay
#     self.numSteps = numSteps

#   def __call__(self, step):
#     lr = self.initial_learning_rate*(1-(step-self.start_decay)/self.numSteps)
#     return tf.math.minimum(lr, self.initial_learning_rate)
# gen_schedule1 = MyLRSchedule(2e-4)
# gen_schedule2 = MyLRSchedule(2e-4)
# dis_schedule1 = MyLRSchedule(1e-4)
# dis_schedule2 = MyLRSchedule(1e-4)
generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)


ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           # generator_g_optimizer=generator_g_optimizer,
                           # generator_f_optimizer=generator_f_optimizer,
                           # discriminator_x_optimizer=discriminator_x_optimizer,
                           # discriminator_y_optimizer=discriminator_y_optimizer
                           )

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=300)
import datetime
if TRAIN:
  log_dir = "log_cyclegan/"
  summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



@tf.function
def train_step(real_x, real_y, epoch):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    total_identity_loss = identity_loss(real_y, same_y) + identity_loss(real_x, same_x)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
  with summary_writer.as_default():
    tf.summary.scalar('total_gen_g_loss', total_gen_g_loss, step=epoch)
    tf.summary.scalar('total_gen_f_loss', total_gen_f_loss, step=epoch)
    tf.summary.scalar('disc_x_loss', disc_x_loss, step=epoch)
    tf.summary.scalar('disc_y_loss', disc_y_loss, step=epoch)
  return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss, total_cycle_loss, total_identity_loss
# count = 0
global imageid 
imageid = 0
# max_psnr = 0
# max_ssim = 0
# id_max_psnr = -1
# id_max_ssim = -1
save_path = 'TestResult/'
def test_image(forward_g, backward_g, test_input, path): # test_input shape: (1, height, width, 3)
  fake_y = forward_g(test_input, training=True)
  # cycled_x = backward_g(fake_y, training=True)
  # cycle_loss = calc_cycle_loss(test_input, cycled_x)
  # same_x = backward_g(test_input, training=True)
  # identity = identity_loss(test_input, same_x)
  global imageid
  # tf.keras.preprocessing.image.save_img("test/fake_ufi/"+names[imageid], fake_y[0]*0.5+0.5, data_format='channels_last')
  # tf.keras.preprocessing.image.save_img("test/cycle_ofi/"+names[imageid], cycled_x[0]*0.5+0.5, data_format='channels_last')
  # tf.keras.preprocessing.image.save_img("test/identity_ofi/"+names[imageid], same_x[0]*0.5+0.5, data_format='channels_last')
  # imageid += 1
  
  for i in range(test_input.shape[0]):
    display_list = [test_input[i]*0.5 + 0.5, test_input[i]*0.5 + 0.5, fake_y[i]*0.5 + 0.5] ########## input input prediction
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    # count += 1
    # psnr = tf.image.psnr(display_list[1], display_list[2], 1)
    # ssim = tf.image.ssim(tf.convert_to_tensor(display_list[1]), tf.convert_to_tensor(display_list[2]), 1)
    # img = tf.concat([display_list[0], tf.cast(tf.fill([256, 10, 3], 1.0), display_list[0].dtype)], axis=1)
    # img = tf.concat([img, display_list[2]], axis=1)
    
    # tf.keras.preprocessing.image.save_img(path+names[imageid], img, data_format='channels_last')
    tf.keras.preprocessing.image.save_img(path+names[imageid], display_list[2], data_format='channels_last')
    imageid += 1
  # return cycle_loss, identity
  return 0, 0
def test_ds(forward_g, backward_g, forward_dataset, epoch):
  # global count
  global imageid
  # average_psnr = 0
  # average_ssim = 0
  # count = 0
  imageid = 0
  path = save_path+str(epoch)+'/'
  MakeFolder(path)
  print('test ds')
  total_cycle_loss = 0
  total_identity_loss = 0
  for x in forward_dataset.as_numpy_iterator():
    cycle_loss, identity = test_image(forward_g, backward_g, x, path)
    total_cycle_loss += cycle_loss
    total_identity_loss += identity
  #   average_psnr += psnr
  #   average_ssim += ssim
  # average_psnr /= count
  # average_ssim /= count
  # print('num_images: ', count)
  # print('psnr: ', average_psnr)
  # print('ssim: ', average_ssim)
  # return average_psnr, average_ssim
  # total_cycle_loss /= imageid
  # total_identity_loss /= imageid
  # print('cycle ', total_cycle_loss)
  # print('identity ', total_identity_loss)
  # print('total ', total_identity_loss+total_cycle_loss)
  return total_cycle_loss, total_identity_loss

def test_multiple_checkpoints(forward_g, backward_g, forward_dataset,  start, end):
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print(latest)
    # checkpoint.restore(latest)
  # PSNR_list = list()
  # SSIM_list = list()
  # global max_psnr 
  # global max_ssim 
  # global id_max_psnr 
  # global id_max_ssim 
  min_cycle = 999
  id_min_cycle = -1
  min_identity = 999
  id_min_identity = -1
  min_total = 999
  id_min_total = -1

  for i in range(start, end+1):
    print('checkpoint ', i)
    ckpt.restore(checkpoint_path+'/ckpt-'+str(i)).expect_partial()
    cycle, identity = test_ds(forward_g, backward_g, forward_dataset,  i-1)
    total = cycle + identity
    if cycle < min_cycle:
      min_cycle = cycle
      id_min_cycle = i
    if identity < min_identity:
      min_identity = identity
      id_min_identity = i
    if total < min_total:
      min_total = total
      id_min_total = i
  #   if average_psnr > max_psnr:
  #     max_psnr = average_psnr
  #     id_max_psnr = i
  #   if average_ssim > max_ssim:
  #     max_ssim = average_ssim
  #     id_max_ssim = i
  print('min cycle ', min_cycle)
  print('at ', id_min_cycle)
  print('min identity ', min_identity)
  print('at ', id_min_identity)
  print('min_total ', min_total)
  print('at ', min_total)

def fit():
    print('fit')
    global max_psnr, max_ssim, id_max_psnr, id_max_ssim
    for epoch in range(EPOCHS):
        # if epoch < 100:
        #   generator_g_optimizer.lr.assign(1e-4)
        #   generator_f_optimizer.lr.assign(1e-4)
        #   discriminator_x_optimizer.lr.assign(1e-4)
        #   discriminator_y_optimizer.lr.assign(1e-4)
        # else:
        #   g_lr = 1e-4*(1-((epoch+1)-100)/100)
        #   d_lr = 1e-4*(1-((epoch+1)-100)/100)
        #   generator_g_optimizer.lr.assign(g_lr)
        #   generator_f_optimizer.lr.assign(g_lr)
        #   discriminator_x_optimizer.lr.assign(d_lr)
        #   discriminator_y_optimizer.lr.assign(d_lr)
        g1_loss = 0
        g2_loss = 0
        d1_loss = 0
        d2_loss = 0
        cycle_loss = 0
        identity_loss = 0

        start = time.time()  
        print('epoch ', epoch)
        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_ufi, train_ofi)):
          # print('xxxxxxxxx')
          if n%10 == 0:
            print ('.', end='')
          g1, g2, d1, d2, c, iden = train_step(image_x, image_y, epoch)
          g1_loss += g1
          g2_loss += g2
          d1_loss += d1
          d2_loss += d2
          cycle_loss += c
          identity_loss += iden
          n += 1
        print('g1 loss {0}, g2 loss {1}, d1 loss {2}, d2 loss {3}, cycle loss {4}, identity loss {5}'.format(g1_loss/num_samples, g2_loss/num_samples, d1_loss/num_samples, d2_loss/num_samples, cycle_loss/num_samples, identity_loss/num_samples) )
          # if n == 150:
          #   break
        # clear_output(wait=True)
        if epoch%1 == 0:
          ckpt_manager.save()
        # test_ds(generator_g, test_ufi,  epoch)
        # print('psnr = %f, ssim = %f'%(psnr,ssim))
        print ('Time taken for epoch {} is {} sec\n'.format(epoch ,
                                                      time.time()-start))
if TRAIN:
# train
# ckpt.restore('checkpoints/cyclegan/ckpt-3').expect_partial()
	# max_psnr, max_ssim = test_ds(generator_g, validation_dataset)
	# f = open('log_cyclegan.txt', 'w')
	fit()
	# f.close()
else:
# test
  import time
  begin = time.time()

  test_multiple_checkpoints(generator_g, generator_f, test_ufi, 1, 100)
  end = time.time()
  print('total time: ', (end-begin))
  # ckpt.restore(checkpoint_path+'/ckpt-'+str(19)).expect_partial()
  # test_ds(generator_g, test_ufi, test_ofi, 1)
  # fake_ufi = load('fake_ufi.jpg', (256, 256))
  # fake_ofi = load('fake_ofi.jpg', (256, 256))
  # fake_ufi = np.expand_dims(fake_ufi, 0)
  # fake_ofi = np.expand_dims(fake_ofi, 0)
  # test_image(generator_g, fake_ufi, fake_ofi)
  # test_image(generator_f, fake_ofi, fake_ufi)
  # 28 26  393
  # 359 380 408os 415 502 517 807
  # 907osod 657od 610osod 575osod 207os 529 540od 544 545 617 797os 826 od 882 881od 55os 128os 145od