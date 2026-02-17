"""
Step 4: Contract Pricing and Risk Analysis
Skill Futures: Risk Pricing for AI-Era Editor Education
This script performs detailed pricing analysis and risk metrics calculation
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DESKTOP_PATH = "C:/Users/zizih/Desktop/"
OUTPUT_FILE = DESKTOP_PATH + "contract_pricing_results.json"

print("=" * 70)
print("STEP 4: CONTRACT PRICING AND RISK ANALYSIS")
print("=" * 70)

# ==========================================
# 1. Load Previous Results
# ==========================================
print("\n[1] Loading previous results...")

# Load sensitivity analysis
sensitivity_df = pd.read_csv(f"{DESKTOP_PATH}sensitivity_analysis.csv")
print(f"   Loaded {len(sensitivity_df)} sensitivity scenarios")

# Load simulation payoffs
payoff_df = pd.read_csv(f"{DESKTOP_PATH}simulation_payoffs.csv")
print(f"   Loaded {len(payoff_df)} simulation paths")

# Load model parameters
with open(f"{DESKTOP_PATH}model_parameters.json", 'r') as f:
    params = json.load(f)
print(f"   Model parameters loaded")

# Load processed data
processed_df = pd.read_csv(f"{DESKTOP_PATH}processed_data.csv")

# ==========================================
# 2. Calculate Detailed Risk Metrics
# ==========================================
print("\n[2] Calculating detailed risk metrics...")

# Filter base case (K=4500, eta=0.3, C=10000)
base_case = payoff_df.copy()

# Calculate various risk metrics
risk_metrics = {
    # Basic probability metrics
    'trigger_probability': float(base_case['triggered'].mean()),
    'expected_payoff': float(base_case['payoff'].mean()),
    'median_payoff': float(base_case['payoff'].median()),
    'payoff_std': float(base_case['payoff'].std()),
    'payoff_var': float(base_case['payoff'].var()),
    
    # Value at Risk metrics
    'var_90': float(np.percentile(base_case['payoff'], 90)),
    'var_95': float(np.percentile(base_case['payoff'], 95)),
    'var_99': float(np.percentile(base_case['payoff'], 99)),
    
    # Conditional Value at Risk (Expected Shortfall)
    'cvar_90': float(base_case[base_case['payoff'] > np.percentile(base_case['payoff'], 90)]['payoff'].mean()),
    'cvar_95': float(base_case[base_case['payoff'] > np.percentile(base_case['payoff'], 95)]['payoff'].mean()),
    'cvar_99': float(base_case[base_case['payoff'] > np.percentile(base_case['payoff'], 99)]['payoff'].mean()),
    
    # Maximum loss metrics
    'max_payoff': float(base_case['payoff'].max()),
    'min_payoff': float(base_case['payoff'].min()),
    
    # Trigger time distribution
    'mean_trigger_time': float(base_case[base_case['triggered']]['trigger_time'].mean()),
    'median_trigger_time': float(base_case[base_case['triggered']]['trigger_time'].median()),
    'trigger_time_std': float(base_case[base_case['triggered']]['trigger_time'].std()),
    
    # Final salary distribution for triggered paths
    'mean_final_salary_triggered': float(base_case[base_case['triggered']]['final_salary'].mean()),
    'mean_min_salary_triggered': float(base_case[base_case['triggered']]['min_salary'].mean()),
}

# Calculate confidence intervals (95% CI)
n_bootstrap = 1000
bootstrap_means = []
for i in range(n_bootstrap):
    sample = base_case['payoff'].sample(frac=1.0, replace=True)
    bootstrap_means.append(sample.mean())

risk_metrics['expected_payoff_ci_lower'] = float(np.percentile(bootstrap_means, 2.5))
risk_metrics['expected_payoff_ci_upper'] = float(np.percentile(bootstrap_means, 97.5))

print(f"   Trigger Probability: {risk_metrics['trigger_probability']:.2%}")
print(f"   Expected Payoff: {risk_metrics['expected_payoff']:.2f} CNY")
print(f"   95% CI: [{risk_metrics['expected_payoff_ci_lower']:.2f}, {risk_metrics['expected_payoff_ci_upper']:.2f}]")
print(f"   VaR (95%): {risk_metrics['var_95']:.2f} CNY")
print(f"   CVaR (95%): {risk_metrics['cvar_95']:.2f} CNY")

# ==========================================
# 3. Pricing with Different Risk Loadings
# ==========================================
print("\n[3] Calculating prices with different risk loadings...")

base_expected = risk_metrics['expected_payoff']
risk_loadings = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

pricing_schemes = []
for loading in risk_loadings:
    price = base_expected * (1 + loading)
    pricing_schemes.append({
        'risk_loading': loading,
        'price': float(price),
        'price_increase': float(price - base_expected),
        'price_increase_pct': float(loading * 100)
    })

print("\n   Pricing Schemes:")
for scheme in pricing_schemes:
    print(f"      Loading {scheme['risk_loading']*100:.0f}%: {scheme['price']:.0f} CNY (+{scheme['price_increase_pct']:.0f}%)")

# ==========================================
# 4. Calculate Fair Premium for Different Contract Designs
# ==========================================
print("\n[4] Calculating fair premiums for different contract designs...")

# Design 1: Low-risk (low K, low premium)
design_low = sensitivity_df[(sensitivity_df['K'] == 3500) & 
                            (sensitivity_df['eta'] == 0.3) & 
                            (sensitivity_df['C'] == 10000)].iloc[0]

# Design 2: Balanced (base case)
design_balanced = sensitivity_df[(sensitivity_df['K'] == 4500) & 
                                 (sensitivity_df['eta'] == 0.3) & 
                                 (sensitivity_df['C'] == 10000)].iloc[0]

# Design 3: High-protection (high K, high premium)
design_high = sensitivity_df[(sensitivity_df['K'] == 5500) & 
                             (sensitivity_df['eta'] == 0.3) & 
                             (sensitivity_df['C'] == 10000)].iloc[0]

contract_designs = [
    {
        'design_name': 'Low-Risk (Budget)',
        'K': int(design_low['K']),
        'eta': float(design_low['eta']),
        'trigger_prob': float(design_low['trigger_probability']),
        'fair_premium': float(design_low['expected_payoff']),
        'risk_adjusted_premium': float(design_low['premium_with_loading']),
        'description': 'Low guarantee, suitable for risk-seeking students'
    },
    {
        'design_name': 'Balanced (Recommended)',
        'K': int(design_balanced['K']),
        'eta': float(design_balanced['eta']),
        'trigger_prob': float(design_balanced['trigger_probability']),
        'fair_premium': float(design_balanced['expected_payoff']),
        'risk_adjusted_premium': float(design_balanced['premium_with_loading']),
        'description': 'Moderate guarantee, balanced risk and cost'
    },
    {
        'design_name': 'High-Protection (Premium)',
        'K': int(design_high['K']),
        'eta': float(design_high['eta']),
        'trigger_prob': float(design_high['trigger_probability']),
        'fair_premium': float(design_high['expected_payoff']),
        'risk_adjusted_premium': float(design_high['premium_with_loading']),
        'description': 'High guarantee, maximum protection'
    }
]

print("\n   Contract Designs Comparison:")
for design in contract_designs:
    print(f"      {design['design_name']}:")
    print(f"         K={design['K']} CNY, P(trigger)={design['trigger_prob']:.1%}")
    print(f"         Premium: {design['risk_adjusted_premium']:.0f} CNY")

# ==========================================
# 5. Break-even Analysis
# ==========================================
print("\n[5] Performing break-even analysis...")

# Calculate the minimum K that makes the contract viable
def find_break_even_K(target_prob=0.5):
    subset = sensitivity_df[(sensitivity_df['eta'] == 0.3) & 
                            (sensitivity_df['C'] == 10000)].copy()
    subset['prob_diff'] = np.abs(subset['trigger_probability'] - target_prob)
    best = subset.loc[subset['prob_diff'].idxmin()]
    return best

break_even_50 = find_break_even_K(0.5)
break_even_75 = find_break_even_K(0.75)

break_even_results = {
    '50%_trigger': {
        'K': float(break_even_50['K']),
        'actual_prob': float(break_even_50['trigger_probability']),
        'premium': float(break_even_50['premium_with_loading'])
    },
    '75%_trigger': {
        'K': float(break_even_75['K']),
        'actual_prob': float(break_even_75['trigger_probability']),
        'premium': float(break_even_75['premium_with_loading'])
    }
}

print(f"   For 50% trigger probability:")
print(f"      Required K = {break_even_50['K']:.0f} CNY")
print(f"      Actual probability = {break_even_50['trigger_probability']:.1%}")
print(f"      Premium = {break_even_50['premium_with_loading']:.0f} CNY")

# ==========================================
# 6. Calculate Sensitivity Elasticities Matrix
# ==========================================
print("\n[6] Calculating complete elasticity matrix...")

elasticity_matrix = {}
for eta in [0.1, 0.2, 0.3, 0.4, 0.5]:
    eta_results = {}
    for c in [5000, 7500, 10000, 12500, 15000]:
        subset = sensitivity_df[(sensitivity_df['eta'] == eta) & 
                                (sensitivity_df['C'] == c)]
        if len(subset) >= 2:
            k_low = subset[subset['K'] == 3500].iloc[0]
            k_high = subset[subset['K'] == 5500].iloc[0]
            
            # Calculate elasticity
            pct_change_K = (5500 - 3500) / 3500
            pct_change_premium = (k_high['premium_with_loading'] - k_low['premium_with_loading']) / k_low['premium_with_loading']
            elasticity = pct_change_premium / pct_change_K
            
            eta_results[str(c)] = float(elasticity)
    
    elasticity_matrix[str(eta)] = eta_results

print("   Elasticity Matrix (premium elasticity w.r.t K):")
for eta, c_results in elasticity_matrix.items():
    print(f"      η={eta}: {c_results}")

# ==========================================
# 7. Save All Results
# ==========================================
print("\n[7] Saving all pricing results...")

# Compile all results
final_results = {
    'risk_metrics': risk_metrics,
    'pricing_schemes': pricing_schemes,
    'contract_designs': contract_designs,
    'break_even_analysis': break_even_results,
    'elasticity_matrix': elasticity_matrix,
    'parameters': {
        'base_K': 4500,
        'base_eta': 0.3,
        'base_C': 10000,
        'discount_rate': 0.03,
        'simulation_months': 60,
        'n_simulations': len(payoff_df)
    }
}

# Save to JSON
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"   Saved: {OUTPUT_FILE}")

# ==========================================
# 8. Generate Summary Table for Paper
# ==========================================
print("\n[8] Generating summary table for paper...")

summary_table = pd.DataFrame({
    'Metric': [
        'Trigger Probability',
        'Expected Payout (Fair Premium)',
        '95% Confidence Interval (Expected)',
        'Value at Risk (95%)',
        'Conditional VaR (95%)',
        'Mean Trigger Time (months)',
        'Median Trigger Time (months)',
        'Max Single Payout',
        'Min Single Payout',
        'Recommended Premium (20% loading)',
        'Premium Elasticity w.r.t K',
        'Premium Elasticity w.r.t η'
    ],
    'Value': [
        f"{risk_metrics['trigger_probability']:.1%}",
        f"{risk_metrics['expected_payoff']:.0f} CNY",
        f"[{risk_metrics['expected_payoff_ci_lower']:.0f}, {risk_metrics['expected_payoff_ci_upper']:.0f}] CNY",
        f"{risk_metrics['var_95']:.0f} CNY",
        f"{risk_metrics['cvar_95']:.0f} CNY",
        f"{risk_metrics['mean_trigger_time']:.1f}",
        f"{risk_metrics['median_trigger_time']:.1f}",
        f"{risk_metrics['max_payoff']:.0f} CNY",
        f"{risk_metrics['min_payoff']:.0f} CNY",
        f"{risk_metrics['expected_payoff'] * 1.2:.0f} CNY",
        f"{elasticity_matrix['0.3']['10000']:.2f}",
        f"{0.152:.2f}"  # From step 3
    ]
})

# Save summary table
summary_file = DESKTOP_PATH + "pricing_summary_table.csv"
summary_table.to_csv(summary_file, index=False)
print(f"   Saved: {summary_file}")

print("\n" + summary_table.to_string(index=False))

# ==========================================
# 9. Generate Contract Design Recommendations
# ==========================================
print("\n[9] Generating contract design recommendations...")

recommendations = pd.DataFrame({
    'Student Type': ['Risk-Averse', 'Risk-Neutral', 'Risk-Seeking'],
    'Recommended K (CNY)': [5500, 4500, 3500],
    'Trigger Probability': ['~100%', '97.6%', '26.2%'],
    'Premium (CNY)': [11910, 11242, 2896],
    'Rationale': [
        'Maximum protection against AI risk, suitable for students with high loan burden',
        'Balanced protection at reasonable cost, recommended for most students',
        'Low upfront cost, bet on own ability to beat the market'
    ]
})

recommendations_file = DESKTOP_PATH + "contract_recommendations.csv"
recommendations.to_csv(recommendations_file, index=False)
print(f"   Saved: {recommendations_file}")

print("\n" + "=" * 70)
print("CONTRACT PRICING ANALYSIS COMPLETE")
print("=" * 70)

print(f"\n[Output Files Created]")
print(f"   1. contract_pricing_results.json - All pricing and risk metrics")
print(f"   2. pricing_summary_table.csv - Summary table for paper")
print(f"   3. contract_recommendations.csv - Student-segment recommendations")

print("\n" + "=" * 70)
print("Ready for Step 5: Final Visualization")
print("=" * 70)