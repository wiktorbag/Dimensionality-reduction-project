
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TorchAutoencoderReducer:
    def __init__(
        self,
        input_dim=None,
        latent_dim=2,
        hidden_dim=16,
        epochs=500,
        lr=0.001,
        batch_size=16,
        random_state=42,
        device=None
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.loss_history_ = []

    def _build_model(self):
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()

                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim)
                )

                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )

            def forward(self, x):
                z = self.encoder(x)
                x_rec = self.decoder(z)
                return x_rec

        return Autoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )

    def fit(self, X, y=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)

        if self.input_dim is None:
            self.input_dim = X.shape[1]

        self.model = self._build_model().to(self.device)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32)
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.loss_history_ = []

        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch in loader:
                xb = batch[0].to(self.device)

                optimizer.zero_grad()

                x_rec = self.model(xb)
                loss = loss_fn(x_rec, xb)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            self.loss_history_.append(epoch_loss)

        return self

    def transform(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():
            Z = self.model.encoder(X_tensor).cpu().numpy()

        return Z

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)