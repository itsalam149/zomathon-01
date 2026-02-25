import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

def engineer_signals(df):
    print("Engineering signals...")
    
    # Ensure datetime
    df['order_time'] = pd.to_datetime(df['order_time'])
    df['merchant_marked_FOR_time'] = pd.to_datetime(df['merchant_marked_FOR_time'])
    df['rider_arrival_time'] = pd.to_datetime(df['rider_arrival_time'])
    df['actual_food_ready_time'] = pd.to_datetime(df['actual_food_ready_time'])
    
    # 1. FOR Delay Offset (rider_arrival_time - merchant_marked_FOR_time in minutes)
    df['for_delay_offset'] = (df['rider_arrival_time'] - df['merchant_marked_FOR_time']).dt.total_seconds() / 60.0
    
    # 2. Corrected KPT Label (actual_food_ready_time - order_time in minutes)
    df['label_kpt'] = (df['actual_food_ready_time'] - df['order_time']).dt.total_seconds() / 60.0
    
    # 3. Naive KPT (for baseline)
    df['naive_kpt_feature'] = (df['merchant_marked_FOR_time'] - df['order_time']).dt.total_seconds() / 60.0

    # 4. Merchant Bias Score
    # In a real scenario, this would be historical avg. 
    # Here we simulate it by taking the mean difference between marked FOR and actual ready per merchant
    # To avoid data leakage, we should really do this on a per-merchant basis based on "past" orders.
    # For this simulation, we'll calculate it per merchant.
    bias_map = df.groupby('merchant_id').apply(
        lambda x: (x['merchant_marked_FOR_time'] - x['actual_food_ready_time']).dt.total_seconds().mean() / 60.0
    ).to_dict()
    df['merchant_bias_score'] = df['merchant_id'].map(bias_map)

    # 5. Kitchen Load Estimate
    # Logic: Number of orders placed by this merchant in the last 15 minutes
    df = df.sort_values('order_time')
    # Use a simpler approach to avoid index alignment issues in old pandas versions
    # We'll compute it by merchant
    results = []
    for mid, group in df.groupby('merchant_id'):
        group = group.sort_values('order_time')
        load = group.rolling('15min', on='order_time', closed='left')['order_id'].count()
        results.append(pd.DataFrame({'order_id': group['order_id'], 'orders_last_15m': load}))
    
    load_df = pd.concat(results)
    df = df.merge(load_df, on='order_id')
    
    # Weighted combination
    df['kitchen_load_estimate'] = (
        df['orders_last_15m'] * 0.5 + 
        df['dine_in_load_estimate'] * 0.3 + 
        df['competitor_load_estimate'] * 0.2
    )

    # 6. Rush Indicator
    # Binary flag based on 90th percentile of load for that merchant
    thresholds = df.groupby('merchant_id')['kitchen_load_estimate'].transform(lambda x: x.quantile(0.9))
    df['rush_indicator'] = (df['kitchen_load_estimate'] > thresholds).astype(int)

    # 7. Cuisine Type Encoding
    cuisine_map = {
        'Bakery': 0,
        'Cafe': 1,
        'QSR': 2,
        'Casual Dining': 3,
        'Fine Dining': 4
    }
    df['cuisine_enc'] = df['cuisine_type'].map(cuisine_map).fillna(-1)

    return df


def train_and_evaluate():
    if not os.path.exists('data/orders.csv'):
        print("Data not found. Run data_simulation.py first.")
        return

    df = pd.read_csv('data/orders.csv')
    df = engineer_signals(df)
    
    # Features for Baseline
    baseline_features = ['naive_kpt_feature', 'item_count', 'hour_of_day']
    
    # Features for Improved Model
    improved_features = [
        'item_count', 
        'kitchen_load_estimate', 
        'merchant_bias_score', 
        'rush_indicator', 
        'order_complexity_score', 
        'hour_of_day', 
        'seating_capacity',
        'cuisine_enc'
    ]

    
    target = 'label_kpt'
    
    X = df[list(set(baseline_features + improved_features))]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Training Baseline Model ---")
    model_baseline = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model_baseline.fit(X_train[baseline_features], y_train)
    
    print("--- Training Improved Model ---")
    model_improved = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model_improved.fit(X_train[improved_features], y_train)
    
    # Evaluation
    preds_baseline = model_baseline.predict(X_test[baseline_features])
    preds_improved = model_improved.predict(X_test[improved_features])
    
    # Add predictions for the entire dataset to compute merchant stats safely
    df['pred_kpt_imp'] = model_improved.predict(df[improved_features])
    df['rider_arrival_imp'] = df['pred_kpt_imp'] - 3
    df['rider_wait_imp'] = (df['label_kpt'] - df['rider_arrival_imp']).clip(lower=0)
    
    def get_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        p50 = np.percentile(np.abs(y_true - y_pred), 50)
        p90 = np.percentile(np.abs(y_true - y_pred), 90)
        return {
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'P50 Error': p50,
            'P90 Error': p90
        }

    m_base = get_metrics(y_test, preds_baseline, "Baseline (Naive FOR)")
    m_imp = get_metrics(y_test, preds_improved, "Improved (Signal Engineering)")
    
    results = pd.DataFrame([m_base, m_imp])
    print("\nModel Comparison:")
    print(results.to_string(index=False))
    
    # Save models and features
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_baseline, 'models/model_baseline.joblib')
    joblib.dump(model_improved, 'models/model_improved.joblib')
    joblib.dump(improved_features, 'models/features.joblib')
    
    # Business Metric Simulation
    # Rider arrival = predicted_ready_time - 3 minutes buffer
    # Rider wait time = max(0, actual_ready - rider_arrival)
    
    test_df = X_test.copy()
    test_df['actual_kpt'] = y_test
    test_df['pred_kpt_base'] = preds_baseline
    test_df['pred_kpt_imp'] = preds_improved
    
    # Simulation
    test_df['rider_arrival_base'] = test_df['pred_kpt_base'] - 3
    test_df['rider_arrival_imp'] = test_df['pred_kpt_imp'] - 3
    
    test_df['rider_wait_base'] = (test_df['actual_kpt'] - test_df['rider_arrival_base']).clip(lower=0)
    test_df['rider_wait_imp'] = (test_df['actual_kpt'] - test_df['rider_arrival_imp']).clip(lower=0)
    
    print("\nBusiness Metric: Rider Wait Time (minutes)")
    print(f"Baseline Avg Wait: {test_df['rider_wait_base'].mean():.2f}")
    print(f"Improved Avg Wait: {test_df['rider_wait_imp'].mean():.2f}")
    print(f"Reduction in Wait Time: {((test_df['rider_wait_base'].mean() - test_df['rider_wait_imp'].mean()) / test_df['rider_wait_base'].mean() * 100):.2f}%")

    results.to_json('data/metrics.json', orient='records')
    
    # Save Merchant specific data for API
    merchant_stats = df.groupby('merchant_id').agg({
        'merchant_bias_score': 'first',
        'cuisine_type': 'first',
        'seating_capacity': 'first',
        'label_kpt': 'mean',
        'rider_wait_imp': 'mean',
        'rush_indicator': 'mean'
    }).rename(columns={

        'label_kpt': 'avg_kpt',
        'rider_wait_imp': 'avg_wait_time',
        'rush_indicator': 'rush_frequency'
    }).reset_index()
    
    merchant_stats.to_json('data/merchant_stats.json', orient='records')
    
    print("\nModels and metrics saved.")

if __name__ == "__main__":
    train_and_evaluate()
