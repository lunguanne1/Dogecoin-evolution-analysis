import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# === 1. Setări inițiale ===
ticker = 'DOGE-USD'
start = "2024-09-01"
end = "2025-03-01"

# === 2. Descarcă datele DOGE ===
doge_data = yf.download(ticker, start=start, end=end)[['Close']].dropna()
doge_data.index = pd.to_datetime(doge_data.index)

# === 3. Calculează randamente lunare ===
monthly_returns = doge_data['Close'].resample('ME').last().pct_change().dropna()
monthly_returns_df = pd.DataFrame(monthly_returns)
monthly_returns_df.columns = ['Month_Rtn']

# === 4. Simulează factori Fama-French lunari ===
np.random.seed(42)
ff3_factors = pd.DataFrame(index=monthly_returns_df.index)
ff3_factors['Mkt-RF'] = np.random.normal(0.02, 0.05, len(ff3_factors))
ff3_factors['SMB'] = np.random.normal(0.01, 0.03, len(ff3_factors))
ff3_factors['HML'] = np.random.normal(0.005, 0.02, len(ff3_factors))
ff3_factors['RF'] = 0.001  

# === 5. Combină factori și randamente DOGE ===
ff_data = pd.merge(ff3_factors, monthly_returns_df, left_index=True, right_index=True)

# === 6. CAPM ===
rf = ff_data['RF'].mean()
market_premium = ff_data['Mkt-RF'].mean()
X_capm = sm.add_constant(ff_data['Mkt-RF'])
y_capm = ff_data['Month_Rtn'] - ff_data['RF']

capm_model = sm.OLS(y_capm, X_capm).fit()
print("=== CAPM Summary ===")
print(capm_model.summary())

intercept_capm, beta_capm = capm_model.params
expected_return_capm = rf + beta_capm * market_premium
print(f"\nCAPM Monthly Expected Return: {expected_return_capm:.4f}")
print(f"CAPM Yearly Expected Return: {expected_return_capm * 12:.4f}")

# === 7. Fama-French 3-Factor Model ===
X_ff3 = ff_data[['Mkt-RF', 'SMB', 'HML']]
X_ff3 = sm.add_constant(X_ff3)
y_ff3 = ff_data['Month_Rtn'] - ff_data['RF']

ff3_model = sm.OLS(y_ff3, X_ff3).fit()
print("\n=== Fama-French 3-Factor Model Summary ===")
print(ff3_model.summary())

intercept_ff3, b1, b2, b3 = ff3_model.params

market_premium = ff_data['Mkt-RF'].mean()
size_premium = ff_data['SMB'].mean()
value_premium = ff_data['HML'].mean()

expected_monthly_return_ff3 = rf + b1 * market_premium + b2 * size_premium + b3 * value_premium
expected_yearly_return_ff3 = expected_monthly_return_ff3 * 12

print(f"\nFama-French Monthly Expected Return: {expected_monthly_return_ff3:.4f}")
print(f"Fama-French Yearly Expected Return: {expected_yearly_return_ff3:.4f}")

# === 8. Plot Randamente reale vs estimate ===
ff_data['CAPM_Pred'] = capm_model.predict(X_capm)
ff_data['FF3_Pred'] = ff3_model.predict(X_ff3)

plt.figure(figsize=(10, 5))
plt.plot(ff_data.index, y_capm, label='Randament Real DOGE', marker='o')
plt.title("Randament Real DOGE")
plt.xlabel("Data")
plt.ylabel("Randament Excedentar (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(ff_data.index, ff_data['CAPM_Pred'], label='CAPM Estimat', linestyle='--')
plt.title("Randament Estimat - Model CAPM")
plt.xlabel("Data")
plt.ylabel("Randament Estimat (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(ff_data.index, ff_data['FF3_Pred'], label='Fama-French Estimat', linestyle=':')
plt.title("Randament Estimat - Model Fama-French 3 Factori")
plt.xlabel("Data")
plt.ylabel("Randament Estimat (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
