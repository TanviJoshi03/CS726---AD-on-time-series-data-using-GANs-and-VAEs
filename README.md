# CS726---AD-on-time-series-data-using-GANs-and-VAEs

Work Under Progress

The original GAN employs an LSTM-based Generator and an LSTM-based Discriminator.

Our new method proposes to include a VAE into this setup. We included an encoder and decoder in between the LSTM layer in the discriminator and generator of the GAN.
models has L_V_G.py and LVG.py. L_V_G.py includes one linear layer with ReLU loss for both the encoder and decoder. As far as LVG.py is concerned, the model architecture is increased with two linear layers with ReLU loss.

With L_V_G.py, ambient_temperature_system_failure dataset was trained tested.
Whereas with LVG.py, 3 datasets was trained and tested - ambient_temperature_system_failure, machine_temperature_system_failure and nyc_taxi dataset. As expected, our model performed much better when there were more datapoints which was the case with machine_temperature_system_failure dataset.
