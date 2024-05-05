# CS726---AD-on-time-series-data-using-GANs-and-VAEs

Work Under Progress

The original GAN employs an LSTM-based Generator and an LSTM-based Discriminator.

We try to incorporate a VAE into the original TanoGAN architecture to learn a low-dimensional representation of the time-series data, which may be able to capture complex patterns in the data better. 

The approach would be to train the VAE to minimize a reconstruction loss and a KL Divergence loss. We may have separate loss functions for the VAE and the GAN.
