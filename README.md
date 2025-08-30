# Dogecoin Financial Analysis

This repository contains a university project for the **Financial Mathematics** course, focusing on the analysis of **Dogecoin (DOGE)** over a 6-month period.  
The main objective is to evaluate whether traditional financial models, such as **CAPM (Capital Asset Pricing Model)** and the **Fama–French 3-Factor Model**, can explain the behavior of Dogecoin returns.

## Repository structure
- `Proiect Matematici financiare.pptx` – project presentation (in Romanian)  
- `Proiect Matematici financiare.py` – Python script for data collection, analysis, and visualization  
- `DOGE_EUR_6months.csv` – dataset with historical Dogecoin prices (EUR)  

## Methodology
- Data collected from **Yahoo Finance**  
- Computation of monthly returns  
- Application of CAPM and Fama–French 3-Factor regression models  
- Comparison of estimated vs. realized returns  

## Findings
- **CAPM** provides a preliminary estimate of risk and return, but with low explanatory power.  
- **Fama–French 3-Factor** slightly improves the fit, yet factors such as SMB and HML are not significant for cryptocurrencies.  
- Overall, Dogecoin’s dynamics are only partially captured by traditional models, as volatility is strongly influenced by market sentiment and social media.  

## How to run
Install required libraries:
```bash
pip install pandas yfinance statsmodels numpy matplotlib
