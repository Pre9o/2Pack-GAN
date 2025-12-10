import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.layers import Flatten, LeakyReLU
from tensorflow.keras.regularizers import l2

def build_generator(random_dim, activation=LeakyReLU(alpha=0.2), l2_reg=2.5e-5):
    """Builds the generator model.
    
    Args:
        random_dim (integer): Dimension of the latent vector
        
    Returns:
        Sequential: Generator model
    """
    model = Sequential()

    model.add(Dense(64, input_dim=random_dim, activation=activation, kernel_regularizer=l2(l2_reg)))
    model.add(Dense(1024, activation=activation, kernel_regularizer=l2(l2_reg)))
    
    model.add(Dense(12544, activation=activation, kernel_regularizer=l2(l2_reg))),

    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation=activation, kernel_regularizer=l2(l2_reg)))

    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation=activation, kernel_regularizer=l2(l2_reg)))

    model.add(Conv2D(1, (1, 1), padding='same', activation='tanh'))

    return model
            
            
def build_discriminator(input_shape, activation='relu', l2_reg=2.5e-5):
    """Builds the discriminator model.
    
    Args:
        input_shape (tuple): Shape of the input images
        
    Returns:
        Sequential: Discriminator model
    """
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation=activation, input_shape=input_shape, kernel_regularizer=l2(l2_reg), name='conv1'))
    
    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', activation=activation, kernel_regularizer=l2(l2_reg), name='conv2'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(1, activation='linear', kernel_regularizer=l2(l2_reg), name='dense'))

    return model