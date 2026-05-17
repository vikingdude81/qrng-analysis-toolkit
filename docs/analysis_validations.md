# Analysis Validations

## p-hacking risks
- `chaos_analysis.py`: Exploratory Lyapunov exponent testing without predefined hypotheses
- `consciousness_metrics.py`: Unrestricted metric comparisons without hypothesis specification

## Multiple comparisons
- Both files perform uncorrected multiple hypothesis tests (e.g., Lyapunov exponents, brain region metrics)
- **Critical risk**: No false discovery rate (FDR) correction applied

## Non-stationarity
- `chaos_analysis.py`: Time series assumed stationary without validation (e.g., ADF test missing)
- `consciousness_metrics.py`: Neuroimaging data (EEG/fMRI) lacks non-stationarity checks

## Recommendations
1. Implement FDR correction for all multiple comparisons
2. Add stationarity validation (ADF test) for time series in `chaos_analysis.py`
3. Use wavelet transforms for non-stationary neuroimaging data in `consciousness_metrics.py`
4. Restrict hypothesis testing to pre-specified questions only
