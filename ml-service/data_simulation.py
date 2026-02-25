import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)

def generate_data(num_merchants=10000, num_orders=200000):
    print(f"Generating {num_merchants} merchants and {num_orders} orders...")
    
    # --- Merchant Features ---
    merchant_ids = np.arange(1, num_merchants + 1)
    merchant_types = ['QSR', 'Fine Dining', 'Cafe', 'Bakery', 'Casual Dining']
    
    merchants = pd.DataFrame({
        'merchant_id': merchant_ids,
        'merchant_avg_rating': np.random.uniform(3.0, 5.0, num_merchants),
        'seating_capacity': np.random.randint(10, 100, num_merchants),
        'avg_daily_orders': np.random.randint(20, 150, num_merchants),
        # reliability score indicates how biased they are (1.0 is perfect, lower is noisier)
        'merchant_reliability_score': np.random.uniform(0.5, 1.0, num_merchants),
        'chain_flag': np.random.choice([0, 1], num_merchants, p=[0.7, 0.3]),
        'cuisine_type': np.random.choice(merchant_types, num_merchants),
        # Bias Type: 0: Normal, 1: Mark Early, 2: Mark Late, 3: Mark at Rider Arrival
        'bias_type': np.random.choice([0, 1, 2, 3], num_merchants, p=[0.4, 0.3, 0.2, 0.1])
    })

    # --- Order Features ---
    # Sampling merchant IDs for orders based on their avg daily orders
    order_merchant_ids = np.random.choice(merchant_ids, size=num_orders)
    
    orders = pd.DataFrame({
        'order_id': np.arange(1, num_orders + 1),
        'merchant_id': order_merchant_ids
    })
    
    # Merge merchant features into orders
    orders = orders.merge(merchants, on='merchant_id')
    
    # Time-based features
    base_date = datetime(2024, 1, 1)
    orders['order_time'] = [base_date + timedelta(
        days=np.random.randint(0, 30),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    ) for _ in range(num_orders)]
    
    orders['is_weekend'] = orders['order_time'].dt.dayofweek >= 5
    orders['hour_of_day'] = orders['order_time'].dt.hour
    
    # Order details
    orders['item_count'] = np.random.randint(1, 15, num_orders)
    orders['order_complexity_score'] = orders['item_count'] * np.random.uniform(0.8, 1.5, num_orders)
    
    # Hidden Ground Truth: Actual KPT
    # Base prep time + complexity + load factors
    orders['dine_in_load_estimate'] = np.random.randint(0, 50, num_orders)
    orders['competitor_load_estimate'] = np.random.randint(0, 30, num_orders)
    
    # Cuisine-specific prep speed
    # Bakery/Cafe: Fast (0.4x-0.6x), QSR: Normal (1.0x), Casual/Fine: Slow (1.4x-2.2x)
    cuisine_multipliers = {
        'Bakery': 0.4,
        'Cafe': 0.6,
        'QSR': 1.0,
        'Casual Dining': 1.4,
        'Fine Dining': 2.2
    }
    orders['cuisine_multiplier'] = orders['cuisine_type'].map(cuisine_multipliers)
    
    base_kpt = 8 # mins
    orders['actual_kpt'] = (base_kpt + \
                          orders['order_complexity_score'] * 2 + \
                          orders['dine_in_load_estimate'] * 0.2 + \
                          orders['competitor_load_estimate'] * 0.1) * orders['cuisine_multiplier'] + \
                          np.random.normal(0, 1.5, num_orders)

    
    # Ensure KPT is positive
    orders['actual_kpt'] = orders['actual_kpt'].clip(lower=5)
    
    orders['actual_food_ready_time'] = orders.apply(
        lambda x: x['order_time'] + timedelta(minutes=x['actual_kpt']), axis=1
    )
    
    # Rider arrival time (usually random but somewhat correlated with prep time)
    orders['rider_arrival_time'] = orders.apply(
        lambda x: x['order_time'] + timedelta(minutes=max(5, x['actual_kpt'] + np.random.normal(0, 5))), axis=1
    )
    
    # --- Simulate Merchant Marked FOR (Food Order Ready) ---
    def simulate_marked_for(row):
        actual_ready = row['actual_food_ready_time']
        bias_type = row['bias_type']
        
        if bias_type == 0: # Normal (small random noise)
            offset = np.random.normal(0, 2)
        elif bias_type == 1: # Mark Early (pre-emptive)
            offset = -np.random.uniform(5, 15)
        elif bias_type == 2: # Mark Late (lazy)
            offset = np.random.uniform(5, 10)
        elif bias_type == 3: # Mark at Rider Arrival
            return row['rider_arrival_time'] + timedelta(seconds=np.random.randint(0, 30))
        
        return actual_ready + timedelta(minutes=offset)

    orders['merchant_marked_FOR_time'] = orders.apply(simulate_marked_for, axis=1)
    
    # Cleanup and save
    os.makedirs('data', exist_ok=True)
    
    # Save merchants and orders separately for portability
    merchants.to_csv('data/merchants.csv', index=False)
    orders.to_csv('data/orders.csv', index=False)
    
    print("Data generation complete. Saved to data/merchants.csv and data/orders.csv")

if __name__ == "__main__":
    generate_data()
