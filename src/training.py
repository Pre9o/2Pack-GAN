"""
Training module for WGAN to generate synthetic network packets.

This module sets up the training environment, manages configurations,
and handles the training loop for the WGAN model to generate realistic
network packet representations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from WGAN import WGAN, generator_loss, discriminator_loss
from models import build_generator, build_discriminator
from packets_generation import save_packets_on_training
from data_loader import load_data_npz
from argparse import ArgumentParser
from datetime import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Available GPU: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
        print("Using CPU...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    print("No GPU found. Using CPU...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TrainingConfig:
    """Manages all directory paths for training outputs."""
    
    def __init__(self, dataset_name, base_dir=None):
        """Initialize training configuration with directory structure.
        
        Args:
            dataset_name (str): Name of the dataset being trained
            base_dir (str, optional): Base directory for results. Defaults to ../results
        """
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            base_dir = os.path.join(parent_dir, 'results')
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.training_dir = os.path.join(base_dir, f"{dataset_name}_{timestamp}")
        
        self.output_images = os.path.join(self.training_dir, 'output_images')
        self.models = os.path.join(self.training_dir, 'models')
        self.loss = os.path.join(self.training_dir, 'loss')
        self.logs = os.path.join(self.training_dir, 'logs', timestamp)
        
        weights_dir = os.path.join(self.training_dir, 'weights')
        self.generator_weights = os.path.join(weights_dir, 'generator')
        self.discriminator_weights = os.path.join(weights_dir, 'discriminator')
        
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories for training."""
        directories = [
            self.output_images,
            self.models,
            self.loss,
            self.logs,
            self.generator_weights,
            self.discriminator_weights
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_generated_images_dir(self, epoch):
        """Get directory path for generated images of a specific epoch.
        
        Args:
            epoch (int): Epoch number
            
        Returns:
            str: Path to epoch-specific images directory
        """
        epoch_dir = os.path.join(self.output_images, f"generated_images_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        return epoch_dir


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, config, random_dim=1024, examples=10):
        """Initialize GANMonitor callback.
        
        Args:
            config (TrainingConfig): Training configuration with directory paths
            random_dim (int): Dimension of the latent vector
            examples (int): Number of examples to generate
        """
        super(GANMonitor, self).__init__()
        self.config = config
        self.random_dim = random_dim
        self.examples = examples
        self.file_writer = tf.summary.create_file_writer(config.logs)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        d_loss = logs.get('d_loss', self.model.current_d_loss)
        g_loss = logs.get('g_loss', self.model.current_g_loss)
        
        with self.file_writer.as_default():
            tf.summary.scalar('g_loss', g_loss, step=epoch)
            tf.summary.scalar('d_loss', d_loss, step=epoch)
        self.file_writer.flush()
        
        save_generated_images(
            epoch, 
            self.model.generator, 
            epoch, 
            self.examples, 
            self.random_dim,
            self.config
        )
        self.model.generator.save(f"{self.config.models}/generator_model{epoch}.keras")
        self.model.generator.save_weights(f"{self.config.generator_weights}/generator_weights{epoch}.weights.h5")
        self.model.discriminator.save_weights(f"{self.config.discriminator_weights}/discriminator_weights{epoch}.weights.h5")
        

def save_generated_images(epoch, generator, batch, examples, random_dim, config):
    """Saves generated images to a file.
    
    Args:
        epoch (int): Epoch number
        generator (Sequential): Generator model
        batch (int): Batch number
        examples (int): Number of examples to generate
        random_dim (int): Dimension of the latent vector
        config (TrainingConfig): Training configuration with directory paths
    """
    # Generate random noise from latent space
    noise = np.random.normal(0, 1, (examples, random_dim))

    # Generate packet representations
    generated_images = generator.predict(noise)

    # Denormalize from [-1, 1] to [0, 255]
    generated_images = (generated_images + 1) * 127.5
    generated_images = generated_images.astype(np.uint8)
    
    epoch_dir = config.get_generated_images_dir(epoch)

    plt.figure(figsize=(20, 4))
    for i in range(examples):
        image = generated_images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.subplot(1, examples, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
    plt.savefig(f"{epoch_dir}/generated_image_{batch}.png")
    save_packets_on_training(generated_images, f"{epoch_dir}/generated_image_{batch}.pcap", epoch, examples)
    plt.close()

def save_hyperparameters(config, l2_reg, learning_rate, epochs, batch_size, random_dim, discriminator_extra_steps, gp_weight):
    """Saves hyperparameters to a JSON file.
    
    Args:
        config (TrainingConfig): Training configuration with directory paths
        l2_reg (float): L2 regularization factor
        learning_rate (float): Learning rate for optimizers
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        random_dim (int): Dimension of the latent vector
        discriminator_extra_steps (int): Number of extra discriminator steps per training step
        gp_weight (float): Gradient penalty weight
    """
    hyperparameters = {
        "l2_reg": l2_reg,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "random_dim": random_dim,
        "discriminator_extra_steps": discriminator_extra_steps,
        "gp_weight": gp_weight
    }
    
    with open(os.path.join(config.training_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)


def save_training_history(history, config):
    """Saves training history plots and data to files.
    
    Args:
        history (tf.keras.callbacks.History): Training history object
        config (TrainingConfig): Training configuration with directory paths
    """
    history_path = os.path.join(config.loss, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=4)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if 'g_loss' in history.history:
        plt.plot(history.history['g_loss'], label='Generator Loss', color='blue', linewidth=2)
    if 'd_loss' in history.history:
        plt.plot(history.history['d_loss'], label='Discriminator Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Losses', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if 'g_loss' in history.history and 'd_loss' in history.history:
        loss_diff = np.array(history.history['g_loss']) - np.array(history.history['d_loss'])
        plt.plot(loss_diff, label='G_loss - D_loss', color='green', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Difference', fontsize=12)
        plt.title('Generator vs Discriminator Balance', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.loss, 'training_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to: {config.loss}")
    print(f"  - training_history.json")
    print(f"  - training_losses.png")


def load_pretrained_models(generator, discriminator, pretrained_generator_dir, pretrained_discriminator_dir, epoch, freeze_layers=0):
    """Load pre-trained weights into generator and discriminator models.
    
    Args:
        generator: Generator model instance
        discriminator: Discriminator model instance
        pretrained_generator_dir (str): Directory containing pre-trained generator weights
        pretrained_discriminator_dir (str): Directory containing pre-trained discriminator weights
        epoch (int): Epoch number to load weights from
        freeze_layers (int): Number of initial layers to freeze
        
    Returns:
        tuple: (generator, discriminator) with loaded weights
    """
    gen_weights_path = os.path.join(pretrained_generator_dir, f'generator_weights{epoch}.weights.h5')
    disc_weights_path = os.path.join(pretrained_discriminator_dir, f'discriminator_weights{epoch}.weights.h5')
    
    if not os.path.exists(gen_weights_path):
        raise FileNotFoundError(f"Generator weights not found: {gen_weights_path}")
    if not os.path.exists(disc_weights_path):
        raise FileNotFoundError(f"Discriminator weights not found: {disc_weights_path}")
    
    print(f"Loading generator weights: {gen_weights_path}")
    generator.load_weights(gen_weights_path)
    
    print(f"Loading discriminator weights: {disc_weights_path}")
    discriminator.load_weights(disc_weights_path)
    
    if freeze_layers > 0:
        print(f"FREEZING LAYERS\n")
        print(f"Generator - freezing first {freeze_layers} layers:\n")
        for i, layer in enumerate(generator.layers[:freeze_layers]):
            layer.trainable = False
            print(f"  {i+1}. {layer.name} FROZEN")
        
        print(f"Discriminator - freezing first {freeze_layers} layers:\n")
        for i, layer in enumerate(discriminator.layers[:freeze_layers]):
            layer.trainable = False
            print(f"  {i+1}. {layer.name} FROZEN")
    
    return generator, discriminator


def setup_models_and_optimizers(args, input_shape, config):
    """Build and configure models and optimizers.
    
    Args:
        args: Parsed command line arguments
        input_shape (tuple): Input shape for discriminator
        
    Returns:
        tuple: (generator, discriminator, g_optimizer, d_optimizer)
    """
    generator_activation = LeakyReLU(alpha=0.2)
    discriminator_activation = 'relu'
    beta1 = 0.5
    
    generator = build_generator(args.random_dim, activation=generator_activation, l2_reg=args.l2_reg)
    discriminator = build_discriminator(input_shape, activation=discriminator_activation, l2_reg=args.l2_reg)
    
    if args.fine_tune:
        if args.pretrained_epoch is None:
            raise ValueError("--pretrained_epoch is required when using --fine_tune")
        
        generator, discriminator = load_pretrained_models(
            generator, discriminator, 
            config.pretrained_generator_dir, 
            config.pretrained_discriminator_dir,
            args.pretrained_epoch,
            args.freeze_layers
        )
    
    generator_optimizer = Adam(learning_rate=args.learning_rate, beta_1=beta1)
    discriminator_optimizer = Adam(learning_rate=args.learning_rate, beta_1=beta1)
    
    return generator, discriminator, generator_optimizer, discriminator_optimizer
        

def main():
    """
    Main function to set up and start WGAN training.

    This function parses command line arguments, initializes the training
    configuration, builds the generator and discriminator models, compiles the
    WGAN, and starts the training process with appropriate callbacks.
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--dataset_name', type=str, default='default_dataset', help='Name of the dataset (npz file without extension)')
    argument_parser.add_argument('--l2_reg', type=float, default=2.5e-5, help='L2 regularization factor')
    argument_parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizers')
    argument_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    argument_parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    argument_parser.add_argument('--random_dim', type=int, default=1024, help='Dimension of the latent vector')
    argument_parser.add_argument('--discriminator_extra_steps', type=int, default=5, help='Number of extra discriminator steps per training step')
    argument_parser.add_argument('--gp_weight', type=float, default=10.0, help='Gradient penalty weight')
    argument_parser.add_argument('--examples', type=int, default=10, help='Number of example images to generate')
    
    argument_parser.add_argument('--fine_tune', action='store_true', help='Enable fine-tuning mode (load pre-trained models)')
    argument_parser.add_argument('--pretrained_epoch', type=int, default=None, help='Epoch number to load from pre-trained model (e.g., 49)')
    argument_parser.add_argument('--freeze_layers', type=int, default=0, help='Number of initial layers to freeze during fine-tuning (0 = no freezing)')
    
    args = argument_parser.parse_args()

    config = TrainingConfig(args.dataset_name)
    
    if args.fine_tune:
        print(f"FINE-TUNING MODE\n")
        print(f"Loading epoch: {args.pretrained_epoch}\n")
        print(f"New training outputs: {config.training_dir}\n")
    else:
        print(f"TRAINING FROM SCRATCH\n")
        print(f"Training outputs: {config.training_dir}\n")
    
    input_shape = (28, 28, 1)
    (X_train, _) = load_data_npz(args.dataset_name + '.npz')
    X_train = np.expand_dims(X_train, axis=-1)
    X_train = (X_train / 255.0) * 2.0 - 1.0
    
    generator, discriminator, gen_optimizer, disc_optimizer = setup_models_and_optimizers(args, input_shape, config)
    
    wgan = WGAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=args.random_dim,
        discriminator_extra_steps=args.discriminator_extra_steps,
        gp_weight=args.gp_weight,
    )
    
    wgan.compile(
        d_optimizer=disc_optimizer,
        g_optimizer=gen_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    
    tensorboard_callback = GANMonitor(config, random_dim=args.random_dim, examples=args.examples)
    
    save_hyperparameters(
        config, args.l2_reg, args.learning_rate, args.epochs, 
        args.batch_size, args.random_dim, args.discriminator_extra_steps, args.gp_weight
    )
    
    print(f"STARTING TRAINING\n")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Learning rate: {args.learning_rate}\n")
    
    history = wgan.fit(X_train, batch_size=args.batch_size, epochs=args.epochs, callbacks=[tensorboard_callback])
    
    save_training_history(history, config)
    
    print(f"Training completed successfully!")
    print(f"All results saved to: {config.training_dir}")

if __name__ == "__main__":
    main()
