"""
Step 5: Generate All Figures for Paper
Skill Futures: Risk Pricing for AI-Era Editor Education
This script generates all publication-ready figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 300

# Configuration
DESKTOP_PATH = "C:/Users/zizih/Desktop/"
OUTPUT_FIGURES = DESKTOP_PATH + "figures/"
import os
if not os.path.exists(OUTPUT_FIGURES):
    os.makedirs(OUTPUT_FIGURES)

print("=" * 70)
print("STEP 5: GENERATE ALL FIGURES FOR PAPER")
print("=" * 70)

# ==========================================
# 1. Load All Data
# ==========================================
print("\n[1] Loading all data...")

# Load sensitivity analysis
sensitivity_df = pd.read_csv(f"{DESKTOP_PATH}sensitivity_analysis.csv")

# Load simulation payoffs
payoff_df = pd.read_csv(f"{DESKTOP_PATH}simulation_payoffs.csv")

# Load processed data
processed_df = pd.read_csv(f"{DESKTOP_PATH}processed_data.csv")
processed_df['date'] = pd.to_datetime(processed_df['date'])

# Load sample paths
sample_paths_df = pd.read_csv(f"{DESKTOP_PATH}sample_paths.csv")

# Load model parameters
with open(f"{DESKTOP_PATH}model_parameters.json", 'r') as f:
    params = json.load(f)

# Load pricing results
with open(f"{DESKTOP_PATH}contract_pricing_results.json", 'r') as f:
    pricing_results = json.load(f)

print(f"   All data loaded successfully")

# ==========================================
# 2. Figure 1: Historical Trends (2x2 panel)
# ==========================================
print("\n[2] Generating Figure 1: Historical Trends...")

fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Figure 1: AI Impact on Editor Labor Market (2023-2025)', fontsize=16, fontweight='bold')

# Figure 1a: Salary Trends
ax1 = axes[0, 0]
ax1.plot(processed_df['date'], processed_df['salary_traditional'], 
         'b-', linewidth=2, label='Traditional Editor', alpha=0.8)
ax1.plot(processed_df['date'], processed_df['salary_ai'], 
         'r-', linewidth=2, label='AI-Skilled Editor', alpha=0.8)
ax1.axvline(pd.Timestamp('2024-02-15'), color='gray', linestyle='--', alpha=0.7, label='Sora Release')
ax1.set_xlabel('Date')
ax1.set_ylabel('Monthly Salary (CNY)')
ax1.set_title('(a) Editor Salary Trends')
ax1.legend(loc='best', frameon=True)
ax1.grid(True, alpha=0.3)

# Figure 1b: AI Premium Rate
ax2 = axes[0, 1]
ax2.fill_between(processed_df['date'], 0, processed_df['ai_premium_rate'], 
                 color='green', alpha=0.3, label='AI Premium')
ax2.plot(processed_df['date'], processed_df['ai_premium_rate'], 
         'g-', linewidth=2)
ax2.axvline(pd.Timestamp('2024-02-15'), color='gray', linestyle='--', alpha=0.7)
ax2.set_xlabel('Date')
ax2.set_ylabel('AI Premium (%)')
ax2.set_title('(b) AI Skill Premium Over Time')
ax2.legend(loc='best', frameon=True)
ax2.grid(True, alpha=0.3)

# Figure 1c: AI Shock Index
ax3 = axes[1, 0]
colors = ['lightblue' if x < 0.1 else 'lightgreen' if x < 0.3 else 'orange' if x < 0.6 else 'red' 
          for x in processed_df['lambda_t']]
ax3.bar(processed_df['date'], processed_df['lambda_t'], color=colors, alpha=0.7, width=20)
ax3.axvline(pd.Timestamp('2024-02-15'), color='gray', linestyle='--', alpha=0.7)
ax3.set_xlabel('Date')
ax3.set_ylabel('AI Shock Index (λ)')
ax3.set_title('(c) AI Technology Shock Index')
ax3.grid(True, alpha=0.3)

# Add colorbar for shock levels
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', alpha=0.7, label='Low (λ<0.1)'),
    Patch(facecolor='lightgreen', alpha=0.7, label='Medium (0.1≤λ<0.3)'),
    Patch(facecolor='orange', alpha=0.7, label='High (0.3≤λ<0.6)'),
    Patch(facecolor='red', alpha=0.7, label='Extreme (λ≥0.6)')
]
ax3.legend(handles=legend_elements, loc='upper left', frameon=True)

# Figure 1d: Correlation Scatter
ax4 = axes[1, 1]
salary_gap = processed_df['salary_ai'] - processed_df['salary_traditional']
scatter = ax4.scatter(processed_df['lambda_t'], salary_gap, 
                      c=processed_df['lambda_t'], cmap='viridis', 
                      s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
z = np.polyfit(processed_df['lambda_t'], salary_gap, 1)
p = np.poly1d(z)
ax4.plot(processed_df['lambda_t'], p(processed_df['lambda_t']), 
         'r--', linewidth=2, label=f'Trend (R²={np.corrcoef(processed_df["lambda_t"], salary_gap)[0,1]**2:.3f})')
ax4.set_xlabel('AI Shock Index (λ)')
ax4.set_ylabel('AI Salary Gap (CNY)')
ax4.set_title('(d) AI Shock vs Salary Gap')
ax4.legend(loc='best', frameon=True)
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='λ')

plt.tight_layout()
fig1.savefig(OUTPUT_FIGURES + 'figure1_historical_trends.png', dpi=300, bbox_inches='tight')
fig1.savefig(OUTPUT_FIGURES + 'figure1_historical_trends.pdf', bbox_inches='tight')
print(f"   Saved: figure1_historical_trends.png/pdf")

# ==========================================
# 3. Figure 2: Monte Carlo Simulation Paths
# ==========================================
print("\n[3] Generating Figure 2: Monte Carlo Simulation Paths...")

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Figure 2: Monte Carlo Simulation of Editor Salaries (2026-2030)', fontsize=16, fontweight='bold')

# Figure 2a: Sample salary paths (first 50)
ax1 = axes[0, 0]
sample_ids = sample_paths_df['simulation_id'].unique()[:50]
for sim_id in sample_ids:
    sim_data = sample_paths_df[sample_paths_df['simulation_id'] == sim_id]
    ax1.plot(sim_data['month'] + 1, sim_data['traditional_salary'], 
             'b-', alpha=0.1, linewidth=0.5)
ax1.axhline(y=4500, color='r', linestyle='--', linewidth=2, label='Guarantee Salary (K=4500)')
ax1.set_xlabel('Months from 2026')
ax1.set_ylabel('Monthly Salary (CNY)')
ax1.set_title('(a) Sample Salary Paths (n=50)')
ax1.legend(loc='best', frameon=True)
ax1.grid(True, alpha=0.3)

# Figure 2b: Trigger probability over time
ax2 = axes[0, 1]
trigger_times = payoff_df[payoff_df['triggered']]['trigger_time']
ax2.hist(trigger_times, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=trigger_times.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean={trigger_times.mean():.1f} months')
ax2.axvline(x=trigger_times.median(), color='orange', linestyle='--', 
            linewidth=2, label=f'Median={trigger_times.median():.1f} months')
ax2.set_xlabel('Trigger Time (months)')
ax2.set_ylabel('Frequency')
ax2.set_title('(b) Distribution of Trigger Times')
ax2.legend(loc='best', frameon=True)
ax2.grid(True, alpha=0.3)

# Figure 2c: Final salary distribution
ax3 = axes[1, 0]
ax3.hist(payoff_df['final_salary'], bins=50, color='lightgreen', 
         edgecolor='black', alpha=0.7, density=True)
ax3.axvline(x=4500, color='red', linestyle='--', linewidth=2, label='Guarantee Level')
ax3.axvline(x=payoff_df['final_salary'].mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'Mean={payoff_df["final_salary"].mean():.0f}')
ax3.set_xlabel('Final Salary (CNY)')
ax3.set_ylabel('Density')
ax3.set_title('(c) Distribution of Final Salaries (2030)')
ax3.legend(loc='best', frameon=True)
ax3.grid(True, alpha=0.3)

# Figure 2d: Payoff distribution
ax4 = axes[1, 1]
payoff_nonzero = payoff_df[payoff_df['payoff'] > 0]['payoff']
ax4.hist(payoff_nonzero, bins=30, color='salmon', edgecolor='black', alpha=0.7, density=True)
ax4.axvline(x=payoff_nonzero.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean={payoff_nonzero.mean():.0f}')
ax4.axvline(x=np.percentile(payoff_nonzero, 95), color='purple', linestyle='--', 
            linewidth=2, label='VaR(95%)')
ax4.set_xlabel('Payoff Amount (CNY)')
ax4.set_ylabel('Density')
ax4.set_title('(d) Distribution of Contract Payoffs')
ax4.legend(loc='best', frameon=True)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(OUTPUT_FIGURES + 'figure2_simulation_results.png', dpi=300, bbox_inches='tight')
fig2.savefig(OUTPUT_FIGURES + 'figure2_simulation_results.pdf', bbox_inches='tight')
print(f"   Saved: figure2_simulation_results.png/pdf")

# ==========================================
# 4. Figure 3: Sensitivity Analysis
# ==========================================
print("\n[4] Generating Figure 3: Sensitivity Analysis...")

fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('Figure 3: Sensitivity Analysis of Contract Parameters', fontsize=16, fontweight='bold')

# Figure 3a: Varying K (eta=0.3, C=10000)
ax1 = axes[0, 0]
k_data = sensitivity_df[(sensitivity_df['eta'] == 0.3) & (sensitivity_df['C'] == 10000)].sort_values('K')
ax1.plot(k_data['K'], k_data['trigger_probability'] * 100, 'bo-', linewidth=2, markersize=8, label='Trigger Probability')
ax1.set_xlabel('Guarantee Salary K (CNY)')
ax1.set_ylabel('Trigger Probability (%)')
ax1.set_title('(a) Impact of Guarantee Level K')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', frameon=True)

# Add secondary axis for premium
ax1b = ax1.twinx()
ax1b.plot(k_data['K'], k_data['premium_with_loading'], 'rs-', linewidth=2, markersize=8, alpha=0.7, label='Premium')
ax1b.set_ylabel('Premium (CNY)')
ax1b.legend(loc='upper left', frameon=True)

# Figure 3b: Varying eta (K=4500, C=10000)
ax2 = axes[0, 1]
eta_data = sensitivity_df[(sensitivity_df['K'] == 4500) & (sensitivity_df['C'] == 10000)].sort_values('eta')
ax2.plot(eta_data['eta'] * 100, eta_data['trigger_probability'] * 100, 'go-', linewidth=2, markersize=8, label='Trigger Probability')
ax2.set_xlabel('AI Impact Factor η (%)')
ax2.set_ylabel('Trigger Probability (%)')
ax2.set_title('(b) Impact of AI Shock Sensitivity η')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', frameon=True)

ax2b = ax2.twinx()
ax2b.plot(eta_data['eta'] * 100, eta_data['premium_with_loading'], 'mo-', linewidth=2, markersize=8, alpha=0.7, label='Premium')
ax2b.set_ylabel('Premium (CNY)')
ax2b.legend(loc='upper left', frameon=True)

# Figure 3c: Varying C (K=4500, eta=0.3)
ax3 = axes[1, 0]
c_data = sensitivity_df[(sensitivity_df['K'] == 4500) & (sensitivity_df['eta'] == 0.3)].sort_values('C')
ax3.plot(c_data['C'] / 1000, c_data['trigger_probability'] * 100, 'co-', linewidth=2, markersize=8, label='Trigger Probability')
ax3.set_xlabel('Training Cost C (thousand CNY)')
ax3.set_ylabel('Trigger Probability (%)')
ax3.set_title('(c) Impact of Training Cost C')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', frameon=True)

ax3b = ax3.twinx()
ax3b.plot(c_data['C'] / 1000, c_data['premium_with_loading'] / 1000, 'yo-', linewidth=2, markersize=8, alpha=0.7, label='Premium (k)')
ax3b.set_ylabel('Premium (thousand CNY)')
ax3b.legend(loc='upper left', frameon=True)

# Figure 3d: Elasticity heatmap
ax4 = axes[1, 1]
elasticity_matrix = pricing_results['elasticity_matrix']
eta_values = [0.1, 0.2, 0.3, 0.4, 0.5]
c_values = [5000, 7500, 10000, 12500, 15000]
elasticity_array = np.array([[elasticity_matrix[str(eta)][str(c)] for c in c_values] for eta in eta_values])

im = ax4.imshow(elasticity_array, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=10)
ax4.set_xticks(np.arange(len(c_values)))
ax4.set_yticks(np.arange(len(eta_values)))
ax4.set_xticklabels([f'{c//1000}k' for c in c_values])
ax4.set_yticklabels([f'{eta*100}%' for eta in eta_values])
ax4.set_xlabel('Training Cost C')
ax4.set_ylabel('AI Impact Factor η')
ax4.set_title('(d) Premium Elasticity w.r.t K')

# Add text annotations
for i in range(len(eta_values)):
    for j in range(len(c_values)):
        text = ax4.text(j, i, f'{elasticity_array[i, j]:.1f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=9)

plt.colorbar(im, ax=ax4, label='Elasticity')

plt.tight_layout()
fig3.savefig(OUTPUT_FIGURES + 'figure3_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
fig3.savefig(OUTPUT_FIGURES + 'figure3_sensitivity_analysis.pdf', bbox_inches='tight')
print(f"   Saved: figure3_sensitivity_analysis.png/pdf")

# ==========================================
# 5. Figure 4: Contract Design Space
# ==========================================
print("\n[5] Generating Figure 4: Contract Design Space...")

fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle('Figure 4: Contract Design Space and Recommendations', fontsize=16, fontweight='bold')

# Figure 4a: K vs Premium scatter with contours
ax1 = axes[0]
k_values = sensitivity_df['K'].unique()
eta_values = sensitivity_df['eta'].unique()
premium_matrix = np.zeros((len(k_values), len(eta_values)))

for i, k in enumerate(k_values):
    for j, eta in enumerate(eta_values):
        val = sensitivity_df[(sensitivity_df['K'] == k) & 
                             (sensitivity_df['eta'] == eta) & 
                             (sensitivity_df['C'] == 10000)]['premium_with_loading'].values
        if len(val) > 0:
            premium_matrix[i, j] = val[0]

X, Y = np.meshgrid(eta_values * 100, k_values)
contour = ax1.contourf(X, Y, premium_matrix / 1000, levels=15, cmap='viridis', alpha=0.8)
ax1.set_xlabel('AI Impact Factor η (%)')
ax1.set_ylabel('Guarantee Salary K (CNY)')
ax1.set_title('(a) Premium Surface (thousand CNY)')
plt.colorbar(contour, ax=ax1, label='Premium (thousand CNY)')

# Mark the three contract designs
designs = pricing_results['contract_designs']
for design in designs:
    if design['design_name'] == 'Low-Risk (Budget)':
        ax1.plot(design['eta']*100, design['K'], 'ro', markersize=10, label='Budget', markeredgecolor='black')
    elif design['design_name'] == 'Balanced (Recommended)':
        ax1.plot(design['eta']*100, design['K'], 'go', markersize=10, label='Recommended', markeredgecolor='black')
    else:
        ax1.plot(design['eta']*100, design['K'], 'bo', markersize=10, label='Premium', markeredgecolor='black')
ax1.legend(loc='best', frameon=True)

# Figure 4b: Student segment recommendations
ax2 = axes[1]
recommendations = pd.read_csv(f"{DESKTOP_PATH}contract_recommendations.csv")

x_pos = np.arange(len(recommendations))
ax2.bar(x_pos, recommendations['Premium (CNY)'], color=['lightcoral', 'lightblue', 'lightgreen'], 
        edgecolor='black', linewidth=1.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(recommendations['Student Type'])
ax2.set_xlabel('Student Risk Profile')
ax2.set_ylabel('Premium (CNY)')
ax2.set_title('(b) Recommended Premiums by Student Type')

# Add text labels
for i, row in recommendations.iterrows():
    ax2.text(i, row['Premium (CNY)'] + 200, f"K={row['Recommended K (CNY)']}\n{row['Trigger Probability']}", 
             ha='center', va='bottom', fontsize=10)

ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig4.savefig(OUTPUT_FIGURES + 'figure4_contract_design.png', dpi=300, bbox_inches='tight')
fig4.savefig(OUTPUT_FIGURES + 'figure4_contract_design.pdf', bbox_inches='tight')
print(f"   Saved: figure4_contract_design.png/pdf")

# ==========================================
# 6. Figure 5: Risk Metrics Visualization
# ==========================================
print("\n[6] Generating Figure 5: Risk Metrics...")

fig5, axes = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle('Figure 5: Risk Metrics and Capital Requirements', fontsize=16, fontweight='bold')

# Figure 5a: VaR and CVaR visualization
ax1 = axes[0]
payoff_sorted = np.sort(payoff_df['payoff'].values)
prob = np.arange(len(payoff_sorted)) / len(payoff_sorted)

ax1.plot(payoff_sorted, prob * 100, 'b-', linewidth=2, label='Cumulative Distribution')
ax1.axvline(x=pricing_results['risk_metrics']['var_95'], color='red', linestyle='--', 
            linewidth=2, label=f"VaR(95%) = {pricing_results['risk_metrics']['var_95']:.0f}")
ax1.axvline(x=pricing_results['risk_metrics']['cvar_95'], color='purple', linestyle='--', 
            linewidth=2, label=f"CVaR(95%) = {pricing_results['risk_metrics']['cvar_95']:.0f}")
ax1.axhline(y=95, color='gray', linestyle=':', alpha=0.7)
ax1.set_xlabel('Payoff (CNY)')
ax1.set_ylabel('Cumulative Probability (%)')
ax1.set_title('(a) Value at Risk and Expected Shortfall')
ax1.legend(loc='best', frameon=True)
ax1.grid(True, alpha=0.3)

# Shade the tail region
ax1.fill_betweenx([95, 100], pricing_results['risk_metrics']['var_95'], 
                   payoff_sorted[-1], color='red', alpha=0.1, label='Tail (5%)')

# Figure 5b: Risk loading comparison
ax2 = axes[1]
pricing_schemes = pricing_results['pricing_schemes']
loadings = [s['risk_loading'] * 100 for s in pricing_schemes]
prices = [s['price'] / 1000 for s in pricing_schemes]

bars = ax2.bar(loadings, prices, width=8, color='skyblue', edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Risk Loading (%)')
ax2.set_ylabel('Premium (thousand CNY)')
ax2.set_title('(b) Premium Under Different Risk Loadings')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, price in zip(bars, prices):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{price:.1f}k', ha='center', va='bottom', fontsize=10)

# Mark recommended loading
ax2.axvline(x=20, color='red', linestyle='--', linewidth=2, label='Recommended (20%)')

plt.tight_layout()
fig5.savefig(OUTPUT_FIGURES + 'figure5_risk_metrics.png', dpi=300, bbox_inches='tight')
fig5.savefig(OUTPUT_FIGURES + 'figure5_risk_metrics.pdf', bbox_inches='tight')
print(f"   Saved: figure5_risk_metrics.png/pdf")

# ==========================================
# 7. Generate Summary Statistics Table
# ==========================================
print("\n[7] Generating summary statistics...")

summary_stats = {
    'Data Period': f"{processed_df['date'].min().date()} to {processed_df['date'].max().date()}",
    'Historical Observations': len(processed_df),
    'Simulation Paths': len(payoff_df),
    'Base Case Trigger Probability': f"{pricing_results['risk_metrics']['trigger_probability']:.1%}",
    'Base Case Expected Payout': f"{pricing_results['risk_metrics']['expected_payoff']:.0f} CNY",
    'Base Case Recommended Premium (20%)': f"{pricing_results['risk_metrics']['expected_payoff'] * 1.2:.0f} CNY",
    'VaR (95%)': f"{pricing_results['risk_metrics']['var_95']:.0f} CNY",
    'CVaR (95%)': f"{pricing_results['risk_metrics']['cvar_95']:.0f} CNY",
    'Mean Trigger Time': f"{pricing_results['risk_metrics']['mean_trigger_time']:.1f} months",
    'Premium Elasticity w.r.t K': f"{pricing_results['elasticity_matrix']['0.3']['10000']:.2f}",
    'Premium Elasticity w.r.t η': "0.15",
    'Recommended K (Balanced)': "4500 CNY",
    'Recommended K (Budget)': "3500 CNY",
    'Recommended K (Premium)': "5500 CNY"
}

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_df.to_csv(OUTPUT_FIGURES + 'summary_statistics.csv', index=False)
print(f"   Saved: summary_statistics.csv")

# ==========================================
# 8. Print Final Summary
# ==========================================
print("\n" + "=" * 70)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\n[Figures Created in: {OUTPUT_FIGURES}]")
print(f"   1. figure1_historical_trends.png/pdf - Historical data visualization")
print(f"   2. figure2_simulation_results.png/pdf - Monte Carlo results")
print(f"   3. figure3_sensitivity_analysis.png/pdf - Parameter sensitivity")
print(f"   4. figure4_contract_design.png/pdf - Contract design space")
print(f"   5. figure5_risk_metrics.png/pdf - Risk metrics visualization")
print(f"   6. summary_statistics.csv - Summary table")

print(f"\n[Key Findings]")
print(f"   - Trigger Probability (Base): {pricing_results['risk_metrics']['trigger_probability']:.1%}")
print(f"   - Expected Payout: {pricing_results['risk_metrics']['expected_payoff']:.0f} CNY")
print(f"   - Recommended Premium: {pricing_results['risk_metrics']['expected_payoff'] * 1.2:.0f} CNY")
print(f"   - VaR(95%): {pricing_results['risk_metrics']['var_95']:.0f} CNY")
print(f"   - Mean Trigger Time: {pricing_results['risk_metrics']['mean_trigger_time']:.1f} months")

print("\n" + "=" * 70)
print("ALL STEPS COMPLETE! READY FOR PAPER WRITING")
print("=" * 70)