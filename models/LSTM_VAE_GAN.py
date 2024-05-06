import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMVAEGenerator(nn.Module):
    """An LSTM-VAE hybrid model with a linear encoder, an LSTM, and a linear decoder.
    
    Args:
        in_dim: Input dimensionality
        out_dim: Output dimensionality
        latent_dim: Dimensionality of the latent space
        hidden_dim: Dimensionality of LSTM hidden states
        n_layers: Number of LSTM layers
    """
    def __init__(self, in_dim, out_dim, latent_dim, hidden_dim, n_layers, device=None):
        super().__init__()
        self.device = device  # Device (CPU/GPU) for computations

        # Linear encoder to transform input to latent dimension
        self.encoder = nn.Linear(in_dim, latent_dim)

        # LSTM layer for processing sequences
        self.lstm = nn.LSTM(latent_dim, hidden_dim, n_layers, batch_first=True)

        # Linear decoder to transform LSTM output back to original sequence dimension
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, out_dim ), nn.Sigmoid())

    def forward(self, input):
        batch_size = input.size(0)
        # Encode input
        encoded = self.encoder(input)  # Transform to latent dimension

        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        
        # LSTM processing
        lstm_out, _ = self.lstm(encoded, (h_0, c_0))

        # Decode LSTM output to the desired output dimension
        decoded = self.decoder(lstm_out)

        return decoded

class LSTMVAEDiscriminator(nn.Module):
    """An LSTM based discriminator with encoder, LSTM, and decoder. It outputs a probability for each element of the sequence.

    Args:
        in_dim: Input dimensionality
        hidden_dim: Dimensionality of LSTM hidden states
        latent_dim: Dimensionality of the encoded latent space
        n_layers: Number of LSTM layers
    """

    def __init__(self, in_dim, latent_dim, hidden_dim, n_layers, device=None):
        super().__init__()
        self.device = device

        # Encoder: Reducing dimension to latent space
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU()
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        
        # Decoder: Mapping LSTM output to a probability
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)

        # Encoding
        encoded = self.encoder(input.view(batch_size * seq_len, -1))
        encoded = encoded.view(batch_size, seq_len, -1)

        # LSTM processing
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        lstm_out, _ = self.lstm(encoded, (h_0, c_0))

        # Decoding to probabilities
        output = self.decoder(lstm_out.contiguous().view(batch_size * seq_len, -1))
        output = output.view(batch_size, seq_len, 1)

        return output

# Usage
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LSTMVAEGenerator(in_dim=100, out_dim=100, latent_dim=50, hidden_dim=256, n_layers=2, device=device)
# discriminator = LSTMVAEDiscriminator(in_dim=100, latent_dim=50, hidden_dim=128, n_layers=2, device=device)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Settings
    batch_size = 16
    seq_len = 32
    noise_dim = 100
    seq_dim = 4
    hidden_dim = 256
    latent_dim = 50
    n_layers = 2

    # Initialize models
    gen = LSTMVAEGenerator(noise_dim, seq_dim, latent_dim, hidden_dim, n_layers, device=device)
    dis = LSTMVAEDiscriminator(seq_dim, latent_dim, hidden_dim, n_layers, device=device)

    gen.to(device)
    dis.to(device)

    # Generate noise as input
    noise = torch.randn(batch_size, seq_len, noise_dim).to(device)

    # Generate output using the generator
    gen_out = gen(noise)

    # Discriminator output
    dis_out = dis(gen_out)

    # Print shapes of inputs and outputs
    print("Noise: ", noise.size())
    print("Generator output: ", gen_out.size())
    print("Discriminator output: ", dis_out.size())