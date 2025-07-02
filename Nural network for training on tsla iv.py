import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Neural Net for IV Surface ---
class IVNet(nn.Module):
    def __init__(self, hidden_dim=64, num_hidden_layers=4):
        super().__init__()
        layers = []
        input_dim = 2
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).clamp(min=0.0)

def fit_iv_surface(df, S0, epochs=100, batch_size=128, lr=1e-3, lambda_K=1e-3, device=None):
    df = df.copy()
    df['m'] = np.log(df['K'] / S0)
    df['tau'] = df['T']
    m_scale = 0.2
    tau_scale = 1.0
    df['m_norm'] = df['m'] / m_scale
    df['tau_norm'] = df['tau'] / tau_scale

    X = torch.tensor(df[['m_norm', 'tau_norm']].values, dtype=torch.float32)
    y = torch.tensor(df['sigma_IV'].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IVNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            mse_loss = nn.MSELoss()(preds, batch_y)
            # Smile convexity penalty
            eps = 1e-3
            m_plus = batch_X.clone(); m_plus[:,0] += eps
            m_minus = batch_X.clone(); m_minus[:,0] -= eps
            iv_plus = model(m_plus)
            iv_minus = model(m_minus)
            second_deriv = (iv_plus - 2*preds + iv_minus) / (eps**2)
            smile_penalty = torch.mean(torch.relu(-second_deriv))
            loss = mse_loss + lambda_K * smile_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += mse_loss.item() * batch_X.size(0)
        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")
    scaler = {'m_scale': m_scale, 'tau_scale': tau_scale}
    return model, scaler

def get_sigma_iv(model, K, S, tau, scaler, device=None):
    m = np.log(K / S)
    m_norm = m / scaler['m_scale']
    tau_norm = tau / scaler['tau_scale']
    inp = torch.tensor([[m_norm, tau_norm]], dtype=torch.float32)
    if device is None:
        device = next(model.parameters()).device
    inp = inp.to(device)
    with torch.no_grad():
        sigma_iv = model(inp).item()
    return sigma_iv

# --- Download TSLA Option Chain and Prepare Data ---
def fetch_tsla_iv_surface():
    ticker = "TSLA"
    stock = yf.Ticker(ticker)
    S0 = stock.history(period="1d")['Close'].iloc[-1]
    expiries = stock.options

    records = []
    for expiry in expiries:
        try:
            chain = stock.option_chain(expiry)
            for df in [chain.calls, chain.puts]:
                for _, row in df.iterrows():
                    if np.isnan(row['impliedVolatility']):
                        continue
                    expiry_date = pd.to_datetime(expiry)
                    today = pd.Timestamp.today().normalize()
                    tau = (expiry_date - today).days / 365.0
                    if tau <= 0: continue
                    records.append({
                        'K': row['strike'],
                        'T': tau,
                        'sigma_IV': row['impliedVolatility']
                    })
        except Exception as e:
            print(f"Error with expiry {expiry}: {e}")

    df = pd.DataFrame(records)
    # --- Robust scaling: only divide if needed ---
    if df['sigma_IV'].mean() > 2:  # likely in percent
        df['sigma_IV'] = df['sigma_IV'] / 100
        print("sigma_IV scaled from percent to decimal.")
    else:
        print("sigma_IV already in decimal form.")
    return df, S0

if __name__ == "__main__":
    # Fetch TSLA IV surface
    df, S0 = fetch_tsla_iv_surface()
    print(df.head())
    print("Strike range:", df['K'].min(), "-", df['K'].max())
    print("Expiry range:", df['T'].min(), "-", df['T'].max())
    print("Spot S0:", S0)

    # Train the neural net on TSLA IV surface
    model, scaler = fit_iv_surface(df, S0, epochs=100)

    # Query the trained net at a value within your data
    K_query = df['K'].median()
    tau_query = df['T'].median()
    sigma_iv = get_sigma_iv(model, K_query, S0, tau_query, scaler)
    print(f"NN Implied vol at K={K_query:.2f}, tau={tau_query:.2f}y: {sigma_iv:.4f}")

    # Now you can use get_sigma_iv(...) in your MC simulation!