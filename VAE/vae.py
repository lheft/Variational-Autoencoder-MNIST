import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square


class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = 550  # H_d
                #define mu layer with sequential
        self.mu_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.latent_size)
        ])

        #define logvar layer with sequential
        self.logvar_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.latent_size)
        ])

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code
        #define encoder with sequential
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu')
        ])

        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.input_size, use_bias=True, activation='sigmoid')
        ])

    def call(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        mu = self.mu_layer(self.encoder(x))             
        
        logvar = self.logvar_layer(self.encoder(x))     
        
        reparam = reparametrize(mu,logvar)                    
        
        x_hat = self.decoder(reparam)                         
        x_hat = tf.reshape(x_hat, x.shape)
       
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar


class CVAE(tf.keras.Model):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C
        self.hidden_dim = 550 # H_d
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu')
        ])

        self.mu_layer = tf.keras.Sequential([tf.keras.layers.Dense(self.latent_size)])# shape (N, Z)

        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True, activation='relu'),
            tf.keras.layers.Dense(self.input_size, use_bias=True, activation='sigmoid')
        ])



        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def call(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x_flat = self.keras.layers.Flatten(x)        
        encoded = self.encoder(tf.concat([tf.cast(x_flat, tf.float32), tf.cast(c, tf.float32) ], axis=1))
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        re = reparametrize(mu, logvar)
        x_hat = self.decoder(tf.concat([ tf.cast(re, tf.float32), tf.cast(c, tf.float32) ], axis=1)) 

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    
    logvar_shape = logvar.shape
    mean=0
    stddev=1

    epsilon = tf.random.normal(shape = logvar_shape, mean = mean, stddev = stddev)
    y=tf.sqrt(tf.exp(logvar))
    z = mu + y * epsilon
    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z

def bce_function(x_hat, x):
    """
    Computes the reconstruction loss of the VAE.
    
    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    
    Returns:
    - reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
    """
    bce_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, 
        reduction=tf.keras.losses.Reduction.SUM,
    )
    reconstruction_loss = bce_fn(x, x_hat) * x.shape[-1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
    return reconstruction_loss

def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
    Returned loss is the average loss per sample in the current batch.

    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    recon_loss = bce_function(x_hat, x) #equations
    kl_d = tf.reduce_sum( 1 +  logvar - tf.math.square(mu) - tf.exp(logvar) ) * (-1/2)
    loss = tf.reduce_mean(recon_loss + kl_d) 

    
    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss
