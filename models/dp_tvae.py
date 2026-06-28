"""
SynthoGen AI — Differentially Private TVAE (DP-TVAE)
======================================================
Trains a Variational Autoencoder (VAE) on tabular data using Opacus
for rigorous Differential Privacy (DP-SGD). 

Usage:
  python models/dp_tvae.py --dataset diabetes --input data/processed/diabetes_mcdd_clean.csv \
                           --output data/synthetic/dp_tvae_diabetes_eps5.csv \
                           --epsilon 5.0 --epochs 50
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.preprocessing import QuantileTransformer, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("dp_tvae")

class TabularPreprocessor:
    def __init__(self):
        self.num_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        self.cat_encoders = {}
        self.columns = []
        self.num_cols = []
        self.cat_cols = []
        
    def fit_transform(self, df):
        self.columns = df.columns.tolist()
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [c for c in self.columns if c not in self.num_cols]
        
        out = df.copy()
        
        # Numericals
        if self.num_cols:
            out[self.num_cols] = self.num_transformer.fit_transform(df[self.num_cols].fillna(df[self.num_cols].median()))
            
        # Categoricals
        for col in self.cat_cols:
            le = LabelEncoder()
            out[col] = le.fit_transform(df[col].astype(str))
            self.cat_encoders[col] = le
            # Scale cat to roughly [-1, 1] for neural net stability
            out[col] = (out[col] / max(1, len(le.classes_) - 1)) * 2 - 1
            
        return out.values.astype(np.float32)
        
    def inverse_transform(self, tensor_data):
        df = pd.DataFrame(tensor_data, columns=self.columns)
        
        if self.num_cols:
            df[self.num_cols] = self.num_transformer.inverse_transform(df[self.num_cols])
            
        for col in self.cat_cols:
            le = self.cat_encoders[col]
            # Unscale
            df[col] = (df[col] + 1) / 2 * max(1, len(le.classes_) - 1)
            df[col] = df[col].round().clip(0, len(le.classes_) - 1).astype(int)
            df[col] = le.inverse_transform(df[col])
            
        return df

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--epsilon", type=float, default=1.0, help="Privacy budget (lower = more private)")
    p.add_argument("--delta", type=float, default=1e-5, help="DP delta (should be < 1/N)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info(f"--- DP-TVAE Training (Epsilon: {args.epsilon}) ---")
    
    # 1. Load Data
    df = pd.read_csv(args.input)
    num_rows = args.num_rows or len(df)
    
    preprocessor = TabularPreprocessor()
    data_scaled = preprocessor.fit_transform(df)
    input_dim = data_scaled.shape[1]
    
    dataset = TensorDataset(torch.tensor(data_scaled))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=input_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Opacus Privacy Engine
    # Only attach Opacus if epsilon is not infinity
    privacy_engine = None
    if args.epsilon < 1000:
        logger.info("Attaching Opacus PrivacyEngine for DP-SGD...")
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=1.0,
        )
    else:
        logger.info("Epsilon is infinity. Training without Differential Privacy.")

    # 4. Training Loop
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x)
            # When using Opacus, reduction must be 'mean' to play nice with per-sample gradients
            loss = loss_function(recon_batch, x, mu, logvar) / len(x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            if privacy_engine:
                eps = privacy_engine.get_epsilon(args.delta)
                logger.info(f"Epoch {epoch} | Loss: {train_loss/len(dataloader):.4f} | Epsilon spent: {eps:.2f}")
            else:
                logger.info(f"Epoch {epoch} | Loss: {train_loss/len(dataloader):.4f}")

    # 5. Generate Data
    logger.info(f"Generating {num_rows} synthetic rows...")
    # Opacus wraps model in GradSampleModule; unwrap to access .decode()
    raw_model = model._module if hasattr(model, '_module') else model
    raw_model.eval()
    with torch.no_grad():
        z = torch.randn(num_rows, 128).to(device)
        synth_tensor = raw_model.decode(z).cpu().numpy()
        
    synth_df = preprocessor.inverse_transform(synth_tensor)
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    synth_df.to_csv(args.output, index=False)
    logger.info(f"Synthetic DP data saved to {args.output}")

if __name__ == "__main__":
    main()
