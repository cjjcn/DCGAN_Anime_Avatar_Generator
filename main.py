import os
import glob
import numpy as np

import tensorflow as tf
from tensorflow import keras

from GAN import Generator, Discriminator
from Data import make_anime_dataset

from PIL import Image
import matplotlib.pyplot as plt


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 计算D 的损失函数
    fake_image = generator(batch_z, is_training) # 假样本
    d_fake_logits = discriminator(fake_image, is_training) # 假样本的输出
    d_real_logits = discriminator(batch_x, is_training) # 真样本的输出
    # 计算梯度惩罚项
    gp = gradient_penalty(discriminator, batch_x, fake_image)
    # WGAN-GP D 损失函数的定义，这里并不是计算交叉熵，而是直接最大化正样本的输出
    # 最小化假样本的输出和梯度惩罚项
    loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 10. * gp

    return loss, gp


def celoss_ones(logits):
    # 计算属于与标签为1 的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)

    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与便签为0 的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)

    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, batch_x, fake_image):
    # 梯度惩罚项计算函数
    batchsz = batch_x.shape[0]

    # 每个样本均随机采样t,用于插值
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # 自动扩展为x 的形状，[b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    # 在真假图片之间做线性插值
    interplate = t * batch_x + (1 - t) * fake_image
    # 在梯度环境中计算D 对插值样本的梯度
    with tf.GradientTape() as tape:
        tape.watch([interplate])  # 加入梯度观察列表
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)

    # 计算每个样本的梯度的范数:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    # 计算梯度惩罚项
    gp = tf.reduce_mean((gp - 1.) ** 2)

    return gp


def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 生成器的损失函数
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    # WGAN-GP G 损失函数，最大化假样本的输出值
    loss = - tf.reduce_mean(d_fake_logits)

    return loss


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])

    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    im = Image.fromarray(final_image)
    im.show()
    im.save('exam11_WGAN_final_image.png')
    # Image.save(final_image)
    # Image(final_image).save(image_path)


d_losses, g_losses = [], []


def draw():
    plt.figure()
    plt.plot(d_losses, 'b', label='generator')
    plt.plot(g_losses, 'r', label='discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig('exam11.2_train_test_VAE.png')
    plt.show()


def main():
    batch_size = 512
    learning_rate = 0.002
    z_dim = 100
    is_training = True
    epochs = 300

    img_path = glob.glob(r'D:\ML\faces\1\*.jpg')
    print('images num:', len(img_path))  # images num: 51223
    # 构建数据集对象，返回数据集Dataset 类和图片大小
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)  # (512, 64, 64, 3) (64, 64, 3)
    sample = next(iter(dataset))  # 采样  (512, 64, 64, 3)
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())  # (512, 64, 64, 3) 1.0 -1.0
    dataset = dataset.repeat(100)  # 重复循环
    db_iter = iter(dataset)

    generator = Generator()  # 创建生成器
    generator.build(input_shape=(4, z_dim))
    discriminator = Discriminator()  # 创建判别器
    discriminator.build(input_shape=(4, 64, 64, 3))
    # 分别为生成器和判别器创建优化器
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    # generator.load_weights('exam11.1_generator.ckpt')
    # discriminator.load_weights('exam11.1_discriminator.ckpt')
    # print('Loaded chpt!!')

    for epoch in range(epochs):  # 训练epochs 次
        print("epoch=", epoch)
        # 采样隐藏向量
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(db_iter)

        # 判别器前向计算
        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss), 'gp:', float(gp))
            z = tf.random.uniform([100, z_dim])

            fake_image = generator(z, training=False)
            img_path = os.path.join('images', 'wgan-%d.png'%epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

        if epoch % 10000 == 1:
            # print(d_losses)
            # print(g_losses)
            generator.save_weights('exam11.2_generator.ckpt')
            discriminator.save_weights('exam11.2_discriminator.ckpt')


if __name__ == '__main__':
    main()
    draw()