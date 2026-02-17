"""
Step 2: Monte Carlo Simulation Engine
Skill Futures: Risk Pricing for AI-Era Editor Education
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)  # For reproducibility

# Configuration
OUTPUT_PATH = "/mnt/kimi/output/"
DESKTOP_PATH = "C:/Users/zizih/Desktop/"

print("=" * 70)
print("STEP 2: MONTE CARLO SIMULATION ENGINE")
print("=" * 70)

# ==========================================
# 1. Load Parameters and Data
# ==========================================
print("\n[1] Loading parameters...")

with open(f"{DESKTOP_PATH}model_parameters.json", 'r') as f:
    params = json.load(f)

df = pd.read_csv(f"{DESKTOP_PATH}processed_data.csv")
df['date'] = pd.to_datetime(df['date'])

print(f"   Historical μ_traditional: {params['mu_traditional']:.2f}%")
print(f"   Historical σ_traditional: {params['sigma_traditional']:.2f}%")
print(f"   Historical μ_ai: {params['mu_ai']:.2f}%")
print(f"   Historical σ_ai: {params['sigma_ai']:.2f}%")

# ==========================================
# 2. Simulation Configuration
# ==========================================
print("\n[2] Setting up simulation parameters...")

# Contract parameters
contract = {
    'guarantee_salary_k': 4500,      # Minimum guaranteed salary (CNY/month)
    'training_cost_c': 10000,        # Cost of retraining program (CNY)
    'guarantee_period_t': 36,        # Guarantee period in months (3 years)
    'waiting_period_w': 3,           # Waiting period before claim (months)
    'discount_rate_r': 0.03,         # Annual discount rate
    'ai_impact_eta': 0.3,            # AI impact decay factor
    'simulation_months': 60,         # Total simulation horizon (5 years)
    'n_simulations': 10000           # Number of Monte Carlo paths
}

print(f"   Guarantee Salary K: {contract['guarantee_salary_k']} CNY")
print(f"   Training Cost C: {contract['training_cost_c']} CNY")
print(f"   Guarantee Period T: {contract['guarantee_period_t']} months")
print(f"   Simulations: {contract['n_simulations']}")

# ==========================================
# 3. AI Shock Projection Model
# ==========================================
print("\n[3] Building AI shock projection model...")

# Extend lambda_t for future periods (2026-2030)
last_lambda = df['lambda_t'].iloc[-1]
last_date = df['date'].iloc[-1]

# Create future dates (60 months from start)
future_dates = pd.date_range(start='2026-01-01', periods=contract['simulation_months'], freq='MS')

# Project lambda_t using mean reversion model
# dλ = θ(μ_λ - λ)dt + σ_λ dW
theta = 0.1  # Mean reversion speed
mu_lambda = params['lambda_mean']
sigma_lambda = params['lambda_std'] / np.sqrt(12)  # Monthly volatility

lambda_projection = np.zeros((contract['n_simulations'], contract['simulation_months']))
lambda_projection[:, 0] = last_lambda

for t in range(1, contract['simulation_months']):
    dW = np.random.normal(0, 1, contract['n_simulations'])
    d_lambda = theta * (mu_lambda - lambda_projection[:, t-1]) + sigma_lambda * dW
    lambda_projection[:, t] = np.maximum(lambda_projection[:, t-1] + d_lambda, 0)

print(f"   Projected λ range: [{lambda_projection.min():.3f}, {lambda_projection.max():.3f}]")
print(f"   Mean projected λ: {lambda_projection.mean():.3f}")

# ==========================================
# 4. Salary Path Simulation (Geometric Brownian Motion with AI Shock)
# ==========================================
print("\n[4] Simulating salary paths...")

# Initial salaries (last observed)
P0_traditional = df['salary_traditional'].iloc[-1]
P0_ai = df['salary_ai'].iloc[-1]

# Monthly parameters
mu_monthly = params['mu_traditional'] / 100 / 12  # Convert to monthly decimal
sigma_monthly = params['sigma_traditional'] / 100 / np.sqrt(12)

# Storage for paths
paths_traditional = np.zeros((contract['n_simulations'], contract['simulation_months']))
paths_ai = np.zeros((contract['n_simulations'], contract['simulation_months']))

paths_traditional[:, 0] = P0_traditional
paths_ai[:, 0] = P0_ai

# Simulate paths
for t in range(1, contract['simulation_months']):
    # Random shocks
    Z_traditional = np.random.normal(0, 1, contract['n_simulations'])
    Z_ai = np.random.normal(0, 1, contract['n_simulations'])
    
    # Correlation between traditional and AI salaries
    correlation = 0.7
    Z_ai = correlation * Z_traditional + np.sqrt(1 - correlation**2) * Z_ai
    
    # Traditional salary: GBM with AI shock decay
    # dP/P = (μ - η*λ)dt + σdW
    drift_t = mu_monthly - contract['ai_impact_eta'] * lambda_projection[:, t] / 12
    paths_traditional[:, t] = paths_traditional[:, t-1] * np.exp(
        (drift_t - 0.5 * sigma_monthly**2) + sigma_monthly * Z_traditional
    )
    
    # AI salary: GBM with premium
    drift_ai = (params['mu_ai'] / 100 / 12) - 0.1 * lambda_projection[:, t] / 12  # Less impacted
    sigma_ai_monthly = params['sigma_ai'] / 100 / np.sqrt(12)
    paths_ai[:, t] = paths_ai[:, t-1] * np.exp(
        (drift_ai - 0.5 * sigma_ai_monthly**2) + sigma_ai_monthly * Z_ai
    )

print(f"   Traditional salary range: [{paths_traditional.min():.0f}, {paths_traditional.max():.0f}]")
print(f"   AI salary range: [{paths_ai.min():.0f}, {paths_ai.max():.0f}]")

# ==========================================
# 5. Calculate Payoffs (Trigger Conditions)
# ==========================================
print("\n[5] Calculating contract payoffs...")

# Discount factor
monthly_discount = np.exp(-contract['discount_rate_r'] / 12)

payoff_results = []

for i in range(contract['n_simulations']):
    path = paths_traditional[i, :]
    triggered = False
    trigger_time = None
    payoff = 0
    
    # Check each month after waiting period
    for t in range(contract['waiting_period_w'], contract['waiting_period_w'] + contract['guarantee_period_t']):
        if t >= len(path):
            break
            
        if path[t] < contract['guarantee_salary_k']:
            triggered = True
            trigger_time = t
            # Discounted payoff
            payoff = contract['training_cost_c'] * (monthly_discount ** t)
            break
    
    payoff_results.append({
        'simulation_id': i,
        'triggered': triggered,
        'trigger_time': trigger_time if triggered else None,
        'payoff': payoff,
        'final_salary': path[-1],
        'min_salary': path.min(),
        'avg_salary': path.mean()
    })

payoff_df = pd.DataFrame(payoff_results)

# ==========================================
# 6. Calculate Key Metrics
# ==========================================
print("\n[6] Calculating contract metrics...")

trigger_prob = payoff_df['triggered'].mean()
expected_payoff = payoff_df['payoff'].mean()
payoff_std = payoff_df['payoff'].std()
value_at_risk_95 = np.percentile(payoff_df['payoff'], 95)
conditional_var_95 = payoff_df[payoff_df['payoff'] > value_at_risk_95]['payoff'].mean()

# Calculate risk loading (20% buffer)
risk_loading = 0.20
total_cost = expected_payoff * (1 + risk_loading)

results = {
    'contract_parameters': contract,
    'trigger_probability': float(trigger_prob),
    'trigger_probability_pct': float(trigger_prob * 100),
    'expected_payoff': float(expected_payoff),
    'payoff_std': float(payoff_std),
    'value_at_risk_95': float(value_at_risk_95),
    'conditional_var_95': float(conditional_var_95),
    'risk_loading_rate': risk_loading,
    'fair_premium': float(expected_payoff),
    'suggested_premium_with_loading': float(total_cost),
    'simulation_timestamp': datetime.now().isoformat()
}

print(f"\n   TRIGGER PROBABILITY: {trigger_prob:.2%}")
print(f"   EXPECTED PAYOFF: {expected_payoff:.2f} CNY")
print(f"   PAYOFF STD: {payoff_std:.2f} CNY")
print(f"   VaR (95%): {value_at_risk_95:.2f} CNY")
print(f"   FAIR PREMIUM: {expected_payoff:.2f} CNY")
print(f"   SUGGESTED PREMIUM (with {risk_loading*100:.0f}% loading): {total_cost:.2f} CNY")

# ==========================================
# 7. Save Simulation Results
# ==========================================
print("\n[7] Saving simulation results...")

# Save payoff results
payoff_file = f"{DESKTOP_PATH}simulation_payoffs.csv"
payoff_df.to_csv(payoff_file, index=False)
print(f"   Saved: {payoff_file}")

# Save sample paths (first 100 for visualization)
sample_paths_file = f"{DESKTOP_PATH}sample_paths.csv"
sample_paths_df = pd.DataFrame({
    'month': np.tile(np.arange(contract['simulation_months']), 100),
    'simulation_id': np.repeat(np.arange(100), contract['simulation_months']),
    'traditional_salary': paths_traditional[:100, :].flatten(),
    'ai_salary': paths_ai[:100, :].flatten(),
    'lambda_shock': lambda_projection[:100, :].flatten()
})
sample_paths_df.to_csv(sample_paths_file, index=False)
print(f"   Saved: {sample_paths_file}")

# Save summary results
results_file = f"{DESKTOP_PATH}simulation_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"   Saved: {results_file}")

# Save all paths (compressed)
paths_file = f"{DESKTOP_PATH}all_paths.npz"
np.savez_compressed(paths_file, 
                   traditional=paths_traditional,
                   ai=paths_ai,
                   lambda_shock=lambda_projection)
print(f"   Saved: {paths_file}")

# ==========================================
# 8. Generate Report
# ==========================================
print("\n" + "=" * 70)
print("MONTE CARLO SIMULATION COMPLETE")
print("=" * 70)

print(f"\n[Key Results for Paper]")
print(f"   Based on {contract['n_simulations']:,} simulations over {contract['simulation_months']} months")
print(f"   Guarantee Salary K = {contract['guarantee_salary_k']} CNY")
print(f"   Training Cost C = {contract['training_cost_c']} CNY")
print(f"   AI Impact Factor η = {contract['ai_impact_eta']}")
print(f"\n   1. Probability of Trigger: {trigger_prob:.2%}")
print(f"   2. Expected Payout: {expected_payoff:.2f} CNY")
print(f"   3. Risk-Adjusted Premium: {total_cost:.2f} CNY")
print(f"\n   Interpretation: Out of 100 students, {trigger_prob*100:.0f} will trigger the guarantee")
print(f"   Average cost per student: {expected_payoff:.0f} CNY")

print(f"\n[Output Files]")
print(f"   1. simulation_payoffs.csv - Individual simulation results")
print(f"   2. sample_paths.csv - First 100 paths for visualization")
print(f"   3. simulation_results.json - Summary statistics")
print(f"   4. all_paths.npz - Full simulation data (compressed)")

print("\n" + "=" * 70)
print("Ready for Step 3: Sensitivity Analysis")
print("=" * 70)