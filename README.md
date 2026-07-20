# Amortized Fiducial Inference for Stochastic Volatility (AFI-SV)

Synthesizes Fiducial Inference, Amortized Variational RL, and GP-Vol. Trains an amortized inference network that directly maps a price path (via its signature) to a fiducial distribution over SV model parameters (H, ν, ρ). At inference time, it instantly outputs a full uncertainty distribution – making it feasible for intraday recalibration of parameter-rich models.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Path signature for feature extraction
- Amortized inference network (VAE-style)
- Fiducial distributions over H (Hurst), ν (vol-of-vol), ρ (leverage)
- Instant inference – no MCMC
- Score = H (higher = more persistent)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-afi-sv-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to neural net training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High AFI-SV score → high Hurst exponent → persistent, trending dynamics.
- Low score → mean-reverting dynamics.

## Requirements

See `requirements.txt`.
