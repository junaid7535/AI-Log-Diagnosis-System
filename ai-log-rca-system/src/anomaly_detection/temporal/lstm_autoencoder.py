import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z, seq_len):
        h = self.fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(h)
        return self.fc_out(out)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim)
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z, x.shape[1])
        return recon, mu, logvar

class TemporalAnomalyDetector:
    def __init__(self, seq_len: int = 100, hidden_dim: int = 64, lr: float = 1e-3):
        self.model = LSTMAutoencoder(input_dim=1, hidden_dim=hidden_dim)
        self.seq_len = seq_len
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train_model(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32):
        sequences = self._create_sequences(data)
        dataset = TensorDataset(torch.FloatTensor(sequences))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                recon, mu, logvar = self.model(x)
                
                recon_loss = nn.MSELoss()(recon, x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.001 * kl_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        sequences = self._create_sequences(data)
        dataset = TensorDataset(torch.FloatTensor(sequences))
        dataloader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                recon, _, _ = self.model(x)
                error = torch.mean((recon - x) ** 2, dim=(1, 2)).cpu().numpy()
                reconstruction_errors.extend(error)
        
        # Pad to original length
        anomaly_scores = np.zeros(len(data))
        anomaly_scores[self.seq_len-1:] = reconstruction_errors
        
        # Normalize
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
        
        return anomaly_scores
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            sequences.append(data[i:i+self.seq_len])
        return np.array(sequences).reshape(-1, self.seq_len, 1)