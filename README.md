# CS726---AD-on-time-series-data-using-GANs-and-VAEs

Work Under Progress

The original GAN employs an LSTM-based Generator and an LSTM-based Discriminator.

Our novel approach integrates a Variational Autoencoder (VAE) into the architecture of Generative Adversarial Networks (GANs). This augmentation involves inserting an encoder and decoder between the LSTM layers of the discriminator and generator components.

In "L_V_G.py," a single linear layer with Rectified Linear Unit (ReLU) activation function is employed for both the encoder and decoder. Conversely, "LVG.py" extends the model architecture by incorporating two additional linear layers with ReLU activation.

Experimental evaluation was conducted using the ambient_temperature_system_failure dataset for "L_V_G.py" and three datasets (ambient_temperature_system_failure, machine_temperature_system_failure, and nyc_taxi dataset) for "LVG.py." Notably, the performance of our model was notably enhanced when operating on datasets with larger volumes of data, such as the machine_temperature_system_failure dataset.
![image](https://github.com/ArvindSN3/CS726---AD-on-time-series-data-using-GANs-and-VAEs/assets/82653679/e16a3dc1-6bdb-479a-8e9d-053c61d1ddfa)
