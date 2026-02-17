"""
Step 1: Data Loading and Preprocessing
Skill Futures: Risk Pricing for AI-Era Editor Education
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')
# Configuration
import os
# 获取桌面路径
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
INPUT_PATH = DESKTOP_PATH + "/"  # 或者把CSV文件放在桌面
OUTPUT_PATH = DESKTOP_PATH + "/"
print("=" * 70)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("=" * 70)

# ==========================================
# 1. Load AI Shock Index Data
# ==========================================
print("\n[1] Loading AI shock index data...")

lambda_df = pd.read_csv(f"{INPUT_PATH}AI冲击指数_lambda_t.csv")
lambda_df['date'] = pd.to_datetime(lambda_df['year_month'])
lambda_df = lambda_df.sort_values('date').reset_index(drop=True)

print(f"   Records: {len(lambda_df)} months")
print(f"   Date range: {lambda_df['date'].min().date()} to {lambda_df['date'].max().date()}")
print(f"   Columns: {list(lambda_df.columns)}")

# Verify key columns exist
required_cols = ['lambda_t', 'total_lambda']
for col in required_cols:
    if col not in lambda_df.columns:
        print(f"   WARNING: Column '{col}' not found!")

# Use total_lambda if lambda_t is not available
if 'lambda_t' not in lambda_df.columns and 'total_lambda' in lambda_df.columns:
    lambda_df['lambda_t'] = lambda_df['total_lambda']
    print("   Using 'total_lambda' as 'lambda_t'")

# ==========================================
# 2. Load Salary Data
# ==========================================
print("\n[2] Loading salary data...")

salary_df = pd.read_csv(f"{INPUT_PATH}论文完整数据_2023_2025.csv")
salary_df['date'] = pd.to_datetime(salary_df['year_month'])
salary_df = salary_df.sort_values('date').reset_index(drop=True)

print(f"   Records: {len(salary_df)} months")
print(f"   Columns: {list(salary_df.columns)}")

# Standardize column names
column_mapping = {
    '薪资_传统剪辑': 'salary_traditional',
    '薪资_AI剪辑': 'salary_ai',
    'lambda_t': 'lambda_t_salary'
}

for old_name, new_name in column_mapping.items():
    if old_name in salary_df.columns:
        salary_df[new_name] = salary_df[old_name]

# ==========================================
# 3. Calculate Derived Variables
# ==========================================
print("\n[3] Calculating derived variables...")

# 3.1 AI Premium Rate
salary_df['ai_premium_rate'] = (salary_df['salary_ai'] / salary_df['salary_traditional'] - 1) * 100

# 3.2 Salary Growth Rates
salary_df['growth_traditional'] = salary_df['salary_traditional'].pct_change() * 100
salary_df['growth_ai'] = salary_df['salary_ai'].pct_change() * 100

# 3.3 Cumulative AI Shock (for dynamic K calculation)
lambda_df['cumulative_lambda'] = lambda_df['lambda_t'].cumsum()
lambda_df['lambda_ma3'] = lambda_df['lambda_t'].rolling(window=3, min_periods=1).mean()

# 3.4 Shock intensity categories
lambda_df['shock_level'] = pd.cut(
    lambda_df['lambda_t'], 
    bins=[0, 0.1, 0.3, 0.6, 1.0], 
    labels=['Low', 'Medium', 'High', 'Extreme']
)

print(f"   AI premium rate range: {salary_df['ai_premium_rate'].min():.1f}% to {salary_df['ai_premium_rate'].max():.1f}%")
print(f"   Average traditional salary: {salary_df['salary_traditional'].mean():.0f} CNY")
print(f"   Average AI salary: {salary_df['salary_ai'].mean():.0f} CNY")

# ==========================================
# 4. Merge Datasets
# ==========================================
print("\n[4] Merging datasets...")

merged_df = pd.merge(
    salary_df[['date', 'year_month', 'salary_traditional', 'salary_ai', 
               'ai_premium_rate', 'growth_traditional', 'growth_ai']],
    lambda_df[['date', 'lambda_t', 'cumulative_lambda', 'lambda_ma3', 'shock_level']],
    on='date',
    how='inner'
)

print(f"   Merged records: {len(merged_df)}")
print(f"   Final columns: {list(merged_df.columns)}")

# ==========================================
# 5. Calculate Model Parameters
# ==========================================
print("\n[5] Estimating model parameters...")

params = {}

# 5.1 Salary statistics
params['salary_mean_traditional'] = float(salary_df['salary_traditional'].mean())
params['salary_mean_ai'] = float(salary_df['salary_ai'].mean())
params['salary_std_traditional'] = float(salary_df['salary_traditional'].std())
params['salary_std_ai'] = float(salary_df['salary_ai'].std())

# 5.2 Growth rates (annualized)
params['mu_traditional'] = float(salary_df['growth_traditional'].mean() * 12)  # Annualized %
params['mu_ai'] = float(salary_df['growth_ai'].mean() * 12)
params['sigma_traditional'] = float(salary_df['growth_traditional'].std() * np.sqrt(12))
params['sigma_ai'] = float(salary_df['growth_ai'].std() * np.sqrt(12))

# 5.3 AI shock parameters
params['lambda_mean'] = float(lambda_df['lambda_t'].mean())
params['lambda_max'] = float(lambda_df['lambda_t'].max())
params['lambda_std'] = float(lambda_df['lambda_t'].std())

# 5.4 Correlation between shock and salary gap
salary_gap = salary_df['salary_ai'] - salary_df['salary_traditional']
params['corr_shock_gap'] = float(np.corrcoef(lambda_df['lambda_t'], salary_gap)[0, 1])

print(f"   Traditional salary μ={params['mu_traditional']:.2f}%, σ={params['sigma_traditional']:.2f}%")
print(f"   AI salary μ={params['mu_ai']:.2f}%, σ={params['sigma_ai']:.2f}%")
print(f"   Correlation(λ, salary gap)={params['corr_shock_gap']:.3f}")

# ==========================================
# 6. Save Processed Data
# ==========================================
print("\n[6] Saving processed data...")

# Save merged dataset
merged_file = f"{OUTPUT_PATH}processed_data.csv"
merged_df.to_csv(merged_file, index=False)
print(f"   Saved: {merged_file}")

# Save parameters for later use
params_file = f"{OUTPUT_PATH}model_parameters.json"
with open(params_file, 'w') as f:
    json.dump(params, f, indent=2)
print(f"   Saved: {params_file}")

# Save summary statistics
summary_stats = {
    'data_period': f"{merged_df['date'].min().date()} to {merged_df['date'].max().date()}",
    'observations': len(merged_df),
    'lambda_t_range': [float(merged_df['lambda_t'].min()), float(merged_df['lambda_t'].max())],
    'salary_traditional_range': [float(merged_df['salary_traditional'].min()), 
                                  float(merged_df['salary_traditional'].max())],
    'salary_ai_range': [float(merged_df['salary_ai'].min()), 
                        float(merged_df['salary_ai'].max())],
    'ai_premium_range': [float(merged_df['ai_premium_rate'].min()), 
                         float(merged_df['ai_premium_rate'].max())]
}

summary_file = f"{OUTPUT_PATH}data_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"   Saved: {summary_file}")

# ==========================================
# 7. Generate Preview Report
# ==========================================
print("\n" + "=" * 70)
print("DATA PREPROCESSING COMPLETE")
print("=" * 70)

print(f"\n[Summary for Paper]")
print(f"   Analysis Period: {summary_stats['data_period']}")
print(f"   Total Observations: {summary_stats['observations']} months")
print(f"   AI Shock Index (λ_t): {summary_stats['lambda_t_range'][0]:.3f} - {summary_stats['lambda_t_range'][1]:.3f}")
print(f"   Traditional Editor Salary: {summary_stats['salary_traditional_range'][0]:.0f} - {summary_stats['salary_traditional_range'][1]:.0f} CNY")
print(f"   AI Editor Salary: {summary_stats['salary_ai_range'][0]:.0f} - {summary_stats['salary_ai_range'][1]:.0f} CNY")
print(f"   AI Premium Range: {summary_stats['ai_premium_range'][0]:.1f}% - {summary_stats['ai_premium_range'][1]:.1f}%")

print(f"\n[Output Files]")
print(f"   1. processed_data.csv - Cleaned dataset for analysis")
print(f"   2. model_parameters.json - Estimated parameters for simulation")
print(f"   3. data_summary.json - Summary statistics")

print("\n" + "=" * 70)
print("Ready for Step 2: Monte Carlo Simulation Setup")
print("=" * 70)