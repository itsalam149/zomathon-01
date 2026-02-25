# Zomathon KPT Prediction System

A production-ready system for optimizing Kitchen Prep Time (KPT) prediction by correcting merchant marking biases through advanced signal engineering.

## 🚀 Overview

Merchant-marked "Food Order Ready (FOR)" timestamps are often noisy. This system improves accuracy by:
- Detecting merchant marking bias.
- Estimating real-time kitchen load.
- Engineering corrected signals (Bias Score, Rush Indicator, Load Estimate).
- Training an XGBoost model that out-performs naive FOR-based predictions.

## 🏗 Project Structure

```bash
.
├── ml-service/         # FastAPI + XGBoost Model
│   ├── data_simulation.py  # Synthetic data generation
│   ├── train_model.py      # Feature engineering & training
│   ├── main.py            # Inference API
│   └── requirements.txt
├── frontend/           # Next.js 14 Dashboard
│   ├── app/               # App Router & UI
│   └── components/        # Recharts & UI Components
├── data/               # Simulated datasets (CSV/JSON)
└── docker-compose.yml
```

## 🛠 Setup & Run

### 1. Generate Data & Train Model
```bash
cd ml-service
python3 -m pip install -r requirements.txt
python3 data_simulation.py
python3 train_model.py
```

### 2. Run with Docker
```bash
docker-compose up --build
```

### 3. Manual Run (Local)
**Start ML Service:**
```bash
cd ml-service
uvicorn main:app --reload
```

**Start Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 📊 ML Results (Simulation)
- **Baseline MAE:** ~3.13m
- **Improved MAE:** ~2.98m
- **Rider Wait Time Reduction:** ~2.56%

## 🎨 Premium UI/UX Features
- **Glassmorphic Design System**: Uses backdrop blurs and semi-transparent layers.
- **Advanced Data Viz**: Neon-accented Recharts with comparative convergence mapping.
- **Neural Sandbox**: Interactive simulator with live behavioral analysis.
- **SaaS Aesthetic**: Dark mode optimized with high-fidelity gradients and typography.

## 🛠 Tech Stack
- **Frontend:** Next.js 14 (App Router), Tailwind CSS, Recharts, Lucide Icons.
- **Backend:** FastAPI, Python 3.9.
- **ML:** XGBoost, Scikit-learn, Pandas.
- **DevOps:** Docker, Docker Compose.
# zomathon-01
