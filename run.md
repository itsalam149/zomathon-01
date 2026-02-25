# 🏃 How to Run the KPT Prediction System

Since Docker is not currently installed on your system, please use the **Manual Setup** instructions below.

## 🛠 Manual Setup (Recommended)

### Step 1: Prepare Data & Models
I have already fixed the training script. You can run it one last time to ensure everything is fresh:
```bash
cd ml-service
python3 train_model.py
```

### Step 2: Start ML Microservice
Open a new terminal and run:
```bash
cd ml-service
python3 main.py
```
*The service will start at http://localhost:8000*

### Step 3: Start Next.js Frontend
Open another terminal and run:
```bash
cd frontend
# (Only if you haven't installed dependencies yet)
npm install
npm run dev
```
*The dashboard will be available at http://localhost:3000*

---

## 🧪 Verification
- **API Check:** Visit [http://localhost:8000/api/stats/overview](http://localhost:8000/api/stats/overview) 
- **Dashboard:** Go to the **Simulation** tab in the UI to test bias correction.
