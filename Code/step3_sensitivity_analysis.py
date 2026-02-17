"""
Step 3: Sensitivity Analysis
Skill Futures: Risk Pricing for AI-Era Editor Education
"""

import pandas as pd
import numpy as np
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

DESKTOP_PATH = "C:/Users/zizih/Desktop/"

print("=" * 70)
print("STEP 3: SENSITIVITY ANALYSIS")
print("=" * 70)

# ==========================================
# 1. Load Base Parameters
# ==========================================
print("\n[1] Loading base parameters...")

with open(f"{DESKTOP_PATH}model_parameters.json", 'r') as f:
    params = json.load(f)

df = pd.read_csv(f"{DESKTOP_PATH}processed_data.csv")

# Base contract parameters
base_contract = {
    'guarantee_salary_k': 4500,
    'training_cost_c': 10000,
    'guarantee_period_t': 36,
    'waiting_period_w': 3,
    'discount_rate_r': 0.03,
    'ai_impact_eta': 0.3,
    'simulation_months': 60,
    'n_simulations': 2000  # Reduced for speed in sensitivity analysis
}

P0_traditional = df['salary_traditional'].iloc[-1]
P0_ai = df['salary_ai'].iloc[-1]
last_lambda = df['lambda_t'].iloc[-1]

mu_monthly = params['mu_traditional'] / 100 / 12
sigma_monthly = params['sigma_traditional'] / 100 / np.sqrt(12)

# ==========================================
# 2. Define Parameter Grids
# ==========================================
print("\n[2] Setting up parameter grids...")

# Parameter ranges to test
k_values = [3500, 4000, 4500, 5000, 5500]  # Guarantee salary
eta_values = [0.1, 0.2, 0.3, 0.4, 0.5]     # AI impact factor
c_values = [5000, 7500, 10000, 12500, 15000]  # Training cost

print(f"   K values: {k_values}")
print(f"   η values: {eta_values}")
print(f"   C values: {c_values}")
print(f"   Total scenarios: {len(k_values) * len(eta_values) * len(c_values)}")

# ==========================================
# 3. Simulation Function
# ==========================================
print("\n[3] Running sensitivity simulations...")

def run_simulation(contract_params, params, P0_traditional, last_lambda):
    """Run a single Monte Carlo simulation with given parameters"""
    n_sim = contract_params['n_simulations']
    months = contract_params['simulation_months']
    
    # Project lambda
    theta = 0.1
    mu_lambda = params['lambda_mean']
    sigma_lambda = params['lambda_std'] / np.sqrt(12)
    
    lambda_proj = np.zeros((n_sim, months))
    lambda_proj[:, 0] = last_lambda
    
    for t in range(1, months):
        dW = np.random.normal(0, 1, n_sim)
        d_lambda = theta * (mu_lambda - lambda_proj[:, t-1]) + sigma_lambda * dW
        lambda_proj[:, t] = np.maximum(lambda_proj[:, t-1] + d_lambda, 0)
    
    # Simulate salary paths
    paths = np.zeros((n_sim, months))
    paths[:, 0] = P0_traditional
    
    for t in range(1, months):
        Z = np.random.normal(0, 1, n_sim)
        drift = mu_monthly - contract_params['ai_impact_eta'] * lambda_proj[:, t] / 12
        paths[:, t] = paths[:, t-1] * np.exp(
            (drift - 0.5 * sigma_monthly**2) + sigma_monthly * Z
        )
    
    # Calculate payoffs
    monthly_discount = np.exp(-contract_params['discount_rate_r'] / 12)
    payoffs = []
    
    for i in range(n_sim):
        triggered = False
        payoff = 0
        for t in range(contract_params['waiting_period_w'], 
                      contract_params['waiting_period_w'] + contract_params['guarantee_period_t']):
            if t >= months:
                break
            if paths[i, t] < contract_params['guarantee_salary_k']:
                triggered = True
                payoff = contract_params['training_cost_c'] * (monthly_discount ** t)
                break
        
        payoffs.append(payoff)
    
    payoffs = np.array(payoffs)
    trigger_prob = np.mean(payoffs > 0)
    expected_payoff = np.mean(payoffs)
    
    return {
        'trigger_probability': trigger_prob,
        'expected_payoff': expected_payoff,
        'fair_premium': expected_payoff,
        'premium_with_loading': expected_payoff * 1.2
    }

# ==========================================
# 4. Run All Scenarios
# ==========================================
results_list = []
total_scenarios = len(k_values) * len(eta_values) * len(c_values)
counter = 0

for k, eta, c in product(k_values, eta_values, c_values):
    counter += 1
    if counter % 10 == 0:
        print(f"   Progress: {counter}/{total_scenarios} scenarios...")
    
    contract = base_contract.copy()
    contract['guarantee_salary_k'] = k
    contract['ai_impact_eta'] = eta
    contract['training_cost_c'] = c
    
    result = run_simulation(contract, params, P0_traditional, last_lambda)
    
    results_list.append({
        'K': k,
        'eta': eta,
        'C': c,
        'trigger_probability': result['trigger_probability'],
        'expected_payoff': result['expected_payoff'],
        'fair_premium': result['fair_premium'],
        'premium_with_loading': result['premium_with_loading']
    })

sensitivity_df = pd.DataFrame(results_list)

# ==========================================
# 5. Generate Analysis Tables
# ==========================================
print("\n[4] Generating analysis tables...")

# Table 1: Varying K (fixed eta=0.3, C=10000)
table_k = sensitivity_df[(sensitivity_df['eta'] == 0.3) & (sensitivity_df['C'] == 10000)][
    ['K', 'trigger_probability', 'expected_payoff', 'premium_with_loading']
].sort_values('K')

# Table 2: Varying eta (fixed K=4500, C=10000)
table_eta = sensitivity_df[(sensitivity_df['K'] == 4500) & (sensitivity_df['C'] == 10000)][
    ['eta', 'trigger_probability', 'expected_payoff', 'premium_with_loading']
].sort_values('eta')

# Table 3: Varying C (fixed K=4500, eta=0.3)
table_c = sensitivity_df[(sensitivity_df['K'] == 4500) & (sensitivity_df['eta'] == 0.3)][
    ['C', 'trigger_probability', 'expected_payoff', 'premium_with_loading']
].sort_values('C')

print("\n   Table 1: Sensitivity to Guarantee Salary K (η=0.3, C=10,000)")
print(table_k.to_string(index=False, float_format='%.2f'))

print("\n   Table 2: Sensitivity to AI Impact Factor η (K=4,500, C=10,000)")
print(table_eta.to_string(index=False, float_format='%.2f'))

print("\n   Table 3: Sensitivity to Training Cost C (K=4,500, η=0.3)")
print(table_c.to_string(index=False, float_format='%.2f'))

# ==========================================
# 6. Calculate Elasticities
# ==========================================
print("\n[5] Calculating elasticities...")

# Elasticity of premium with respect to K
base_row = sensitivity_df[(sensitivity_df['K'] == 4500) & 
                          (sensitivity_df['eta'] == 0.3) & 
                          (sensitivity_df['C'] == 10000)].iloc[0]

k_high = sensitivity_df[(sensitivity_df['K'] == 5500) & 
                        (sensitivity_df['eta'] == 0.3) & 
                        (sensitivity_df['C'] == 10000)].iloc[0]

k_low = sensitivity_df[(sensitivity_df['K'] == 3500) & 
                       (sensitivity_df['eta'] == 0.3) & 
                       (sensitivity_df['C'] == 10000)].iloc[0]

elasticity_k = ((k_high['premium_with_loading'] - k_low['premium_with_loading']) / k_low['premium_with_loading']) / \
               ((k_high['K'] - k_low['K']) / k_low['K'])

eta_high = sensitivity_df[(sensitivity_df['K'] == 4500) & 
                          (sensitivity_df['eta'] == 0.5) & 
                          (sensitivity_df['C'] == 10000)].iloc[0]

eta_low = sensitivity_df[(sensitivity_df['K'] == 4500) & 
                         (sensitivity_df['eta'] == 0.1) & 
                         (sensitivity_df['C'] == 10000)].iloc[0]

elasticity_eta = ((eta_high['premium_with_loading'] - eta_low['premium_with_loading']) / eta_low['premium_with_loading']) / \
                 ((eta_high['eta'] - eta_low['eta']) / eta_low['eta'])

print(f"   Elasticity of Premium w.r.t K: {elasticity_k:.3f}")
print(f"   Elasticity of Premium w.r.t η: {elasticity_eta:.3f}")

# ==========================================
# 7. Find Optimal K for Target Trigger Probability
# ==========================================
print("\n[6] Finding optimal K for target trigger probabilities...")

target_probs = [0.10, 0.25, 0.50]
optimal_ks = {}

for target in target_probs:
    # Find K that gives closest trigger probability
    subset = sensitivity_df[(sensitivity_df['eta'] == 0.3) & (sensitivity_df['C'] == 10000)]
    closest_idx = (subset['trigger_probability'] - target).abs().idxmin()
    optimal_k = subset.loc[closest_idx, 'K']
    actual_prob = subset.loc[closest_idx, 'trigger_probability']
    premium = subset.loc[closest_idx, 'premium_with_loading']
    optimal_ks[target] = {'K': int(optimal_k), 'actual_prob': float(actual_prob), 'premium': float(premium)}
    print(f"   Target {target:.0%}: K={optimal_k}, Actual={actual_prob:.1%}, Premium={premium:.0f} CNY")

# ==========================================
# 8. Save Results
# ==========================================
print("\n[7] Saving sensitivity analysis results...")

sensitivity_file = f"{DESKTOP_PATH}sensitivity_analysis.csv"
sensitivity_df.to_csv(sensitivity_file, index=False)
print(f"   Saved: {sensitivity_file}")

# Convert numpy/pandas types to Python native types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    else:
        return obj

# Save summary with type conversion
summary = {
    'base_case': {
        'K': int(base_row['K']),
        'eta': float(base_row['eta']),
        'C': int(base_row['C']),
        'trigger_probability': float(base_row['trigger_probability']),
        'premium': float(base_row['premium_with_loading'])
    },
    'elasticities': {
        'wrt_K': float(elasticity_k),
        'wrt_eta': float(elasticity_eta)
    },
    'optimal_K_for_targets': optimal_ks,
    'tables': {
        'varying_K': [{k: convert_to_serializable(v) for k, v in row.items()} for row in table_k.to_dict('records')],
        'varying_eta': [{k: convert_to_serializable(v) for k, v in row.items()} for row in table_eta.to_dict('records')],
        'varying_C': [{k: convert_to_serializable(v) for k, v in row.items()} for row in table_c.to_dict('records')]
    }
}

summary_file = f"{DESKTOP_PATH}sensitivity_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"   Saved: {summary_file}")

# ==========================================
# 9. Generate Report
# ==========================================
print("\n" + "=" * 70)
print("SENSITIVITY ANALYSIS COMPLETE")
print("=" * 70)

print(f"\n[Key Findings for Paper]")
print(f"   1. Base Case (K=4500, η=0.3, C=10000):")
print(f"      - Trigger Probability: {base_row['trigger_probability']:.1%}")
print(f"      - Risk-Adjusted Premium: {base_row['premium_with_loading']:.0f} CNY")

print(f"\n   2. Parameter Sensitivity:")
print(f"      - Premium elasticity w.r.t K: {elasticity_k:.2f}")
print(f"        (K每增加1%，保费变化{elasticity_k:.2f}%)")
print(f"      - Premium elasticity w.r.t η: {elasticity_eta:.2f}")
print(f"        (η每增加1%，保费变化{elasticity_eta:.2f}%)")

print(f"\n   3. Recommended Contract Designs:")
for target, info in optimal_ks.items():
    print(f"      - {target:.0%}触发概率: K={info['K']}, 保费={info['premium']:.0f} CNY")

print(f"\n[Output Files]")
print(f"   1. sensitivity_analysis.csv - All {total_scenarios} scenarios")
print(f"   2. sensitivity_summary.json - Key findings and tables")

print("\n" + "=" * 70)
print("Ready for Step 4: Contract Pricing & Visualization")
print("=" * 70)