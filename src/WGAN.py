"""
WGAN module implementing the Wasserstein GAN with Gradient Penalty.

This module defines the WGAN class, which extends the Keras Model class to
implement the training loop for the Wasserstein GAN with Gradient Penalty.
It includes methods for compiling the model, calculating the gradient penalty,
and performing a training step.
"""

import tensorflow as tf
from tensorflow import keras

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.current_g_loss = 0.0
        self.current_d_loss = 0.0

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """
        Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.

        Args:
            batch_size (integer): Batch size
            real_images (tensor): Real images
            fake_images (tensor): Fake images

        Returns:
            tensor: Gradient penalty
        """
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_images):
        """
        The train step function.
        Args:
            real_images (tensor): Real images

        Returns:
            dict: Dictionary with the generator and discriminator losses
        """
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
            
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        self.current_g_loss = g_loss
        self.current_d_loss = d_loss    
            
        return {"d_loss": d_loss, "g_loss": g_loss}
    

def discriminator_loss(real_img, fake_img):
    """
    Calculates the discriminator loss.

    Args:
        real_img (tensor): Real images
        fake_img (tensor): Fake images
        
    Returns:
        tensor: Discriminator loss
    """
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    """
    Calculates the generator loss.
    Args:
        fake_img (tensor): Fake images

    Returns:
        tensor: Generator loss
    """
    return -tf.reduce_mean(fake_img)
