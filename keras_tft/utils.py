import pandas as pd
import matplotlib.pyplot as plt
import holidays
import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_timeseries(
    file_path: str,
    forecast_horizon: int,
    is_holiday: bool = False,
    country: str = None,
    is_weekend: bool = False,
    include_friday_in_weekend: bool = False,
    static_covariates: Optional[List[str]] = None,
    cutoff_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Preprocess time series data for the TFT model.

    This function loads data, handles static covariates, performs feature engineering 
    (holidays, weekends), scales the target variable, and splits the data into 
    history, prediction input, and test sets.

    Args:
        file_path (str): Path to the data file (.csv or .xlsx).
        forecast_horizon (int): Number of steps to forecast into the future.
        is_holiday (bool, optional): Whether to add a holiday indicator. Defaults to False.
        country (str, optional): Country code or name for holidays (e.g., 'Greece', 'US'). 
                                 Required if is_holiday is True. Defaults to None.
        is_weekend (bool, optional): Whether to add a weekend indicator. Defaults to False.
        include_friday_in_weekend (bool, optional): If True, Friday is considered a weekend. Defaults to False.
        static_covariates (List[str], optional): List of column names to use as static covariates. 
                                                 These will be preserved in the output DataFrames.
        cutoff_date (str, optional): Date string to split history and future data. 
                                     If None, uses the last `forecast_horizon` steps as test.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]: 
            - 'history' dataframe: Data up to the cutoff point.
            - 'pred_input' dataframe: History + Future (for creating windows).
            - 'test_df' dataframe: Actual future values (scaled) for evaluation.
            - scaler: Fitted MinMaxScaler for the target column.
    """
    # 1. Load Data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

    # 2. Identify Columns
    # Assumption: 1st column is Date/Timestamp, 2nd is Target (y) if not named explicitly
    # We rename them to standard 'timestamp' and 'y'
    cols = df.columns.tolist()
    
    # Try to find date column
    date_col = None
    for c in cols:
        if 'date' in c.lower() or 'time' in c.lower():
            date_col = c
            break
    if not date_col:
        date_col = cols[0] # Fallback to first column
        
    # Try to find target column
    target_col = None
    # If we found date_col, assume next one is target if not specified?
    # The prompt implies generic, but let's assume the remaining one or 'y' / 'target'
    # Exclude static_covariates from possible targets if provided
    possible_targets = [c for c in cols if c != date_col and c not in (static_covariates or [])]
    
    # Heuristics for target column name
    common_target_names = ['y', 'target', 'sales', 'traffic', 'demand', 'attendance', 'weekly_sales']
    
    # 1. Check for exact match in common names
    for name in common_target_names:
        for col in possible_targets:
            if col.lower() == name:
                target_col = col
                break
        if target_col:
            break
            
    # 2. If not found, check if only one possible target remains
    if not target_col:
        if len(possible_targets) == 1:
            target_col = possible_targets[0]
        else:
            # 3. Fallback: Pick the first possible target that is NOT the date column (already filtered)
            # and preferably not 'id' or 'store' if possible, but we filtered static_covariates.
            # If we have multiple, we default to the first one.
            if possible_targets:
                target_col = possible_targets[0]
            else:
                # If absolutely no candidates, fallback to 2nd column if it's not date_col, else 1st
                # This is a last resort and might still fail if everything is filtered out.
                if len(cols) > 1 and cols[1] != date_col:
                    target_col = cols[1]
                elif cols[0] != date_col:
                    target_col = cols[0]
                else:
                    raise ValueError("Could not identify target column. Please rename target to 'y' or 'target'.")

    df = df.rename(columns={date_col: 'timestamp', target_col: 'y'})
    
    # Ensure y is float
    df['y'] = df['y'].astype(float)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split into train and test if cutoff_date is provided
    if cutoff_date:
        train_df = df[df['timestamp'] <= cutoff_date].copy()
        test_df = df[df['timestamp'] > cutoff_date].copy()
    else:
        # If no cutoff, use all data for fitting (not recommended for strict evaluation)
        train_df = df.copy()
        test_df = pd.DataFrame() # Empty
    
    # Scale target column using training stats
    scaler = MinMaxScaler()
    # Fit on training data
    scaler.fit(train_df[['y']])
    
    # Transform both
    df['y'] = scaler.transform(df[['y']])
    # We also need to update train_df/test_df if we use them later, but we are modifying df in place/reassigning
    # Actually, we should be careful. We just updated df['y'].
    # If we need to scale other covariates, we should do it similarly.
    
    # Infer Frequency
    freq = pd.infer_freq(df['timestamp'])
    if not freq:
        # Simple heuristic if infer_freq fails (e.g. missing data)
        diff = df['timestamp'].diff().mode()[0]
        if diff == pd.Timedelta(days=1):
            freq = 'D'
        elif diff == pd.Timedelta(hours=1):
            freq = 'H'
        # Add more heuristics if needed, or default to 'D'
        else:
            freq = 'D' 

    # 3. Infer Covariates & ID Column
    # Identify ID column
    if static_covariates:
        # If multiple static covariates, we can use them all as features.
        # But we need ONE id_column for TFT tracking/plotting usually?
        # Or we just use the first one as ID.
        # Let's assume the first one is the primary ID.
        primary_id = static_covariates[0]
        
        # Verify all exist
        missing = [c for c in static_covariates if c not in df.columns]
        if missing:
             # Check if renamed
             if len(missing) == 1 and missing[0] == primary_id:
                 if primary_id == date_col:
                      df['id_column'] = df['timestamp'].astype(str)
                 elif primary_id == target_col:
                      df['id_column'] = df['y'].astype(str)
                 else:
                      raise ValueError(f"Static covariate '{primary_id}' not found.")
             else:
                  raise ValueError(f"Static covariates {missing} not found.")
        
        if 'id_column' not in df.columns:
            if primary_id in df.columns:
                # Rename primary_id to id_column to avoid duplication
                # But we must ensure we don't lose the original name if it's used elsewhere?
                # The user wants to avoid creating 'Store' if it's not required.
                # If we rename, 'Store' is gone.
                # But 'Store' is in static_covariates list.
                # If we rename, we should update static_covariates?
                # Actually, if we rename, we should just use 'id_column' as the static feature name.
                # But the user passes static_cov_cols=['id_column'] in the notebook.
                # So renaming is correct.
                df = df.rename(columns={primary_id: 'id_column'})
                # Ensure it is string
                df['id_column'] = df['id_column'].astype(str)
            else:
                # Should have been handled above
                pass
                
        # Ensure all static covariates are strings or categorical?
        # If they are used as static features in the model, they should be preserved.
        # We don't rename them away, just ensure id_column exists.
        
    elif 'id_column' not in df.columns:
        df['id_column'] = "0" # Default ID as string
        
    # Identify other covariates (exclude timestamp, y, id_column)
    # We assume remaining columns are covariates
    exclude_cols = ['timestamp', 'y', 'id_column']
    potential_covariates = [c for c in df.columns if c not in exclude_cols]
    
    continuous_covariates = []
    categorical_covariates = [] # Including one-hot encoded
    
    for col in potential_covariates:
        # Check if column is one-hot encoded (binary 0/1)
        if df[col].dropna().isin([0, 1]).all():
             categorical_covariates.append(col)
        # Check if numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
             continuous_covariates.append(col)
        else:
             # Treat non-numeric as categorical (though TFT might need encoding, we leave as is for now)
             categorical_covariates.append(col)

    # Normalize continuous covariates using training stats
    # We need to fit a scaler for EACH covariate or use one scaler for all if they are similar?
    # Usually separate scalers or one StandardScaler for the matrix.
    # But we are returning only ONE scaler (for y?).
    # The user said "Validation/testing data should be normalized with the training data stats (mean and std)!"
    # And "return data, pred_input, scaler".
    # If we scale covariates, we should probably keep their scalers or just apply the transformation.
    # Since we only return one scaler, I assume it's primarily for the target 'y' inverse transform.
    # For covariates, we just transform them in the dataframe and don't return their scalers (unless requested).
    
    for col in continuous_covariates:
        # Fit on training part
        if cutoff_date:
            train_vals = df[df['timestamp'] <= cutoff_date][col]
        else:
            train_vals = df[col]
            
        mean = train_vals.mean()
        std = train_vals.std()
        
        if std != 0:
            df[col] = (df[col] - mean) / std
            
    # 4. Time Covariates
    time_covariates = []
    
    if is_holiday:
        if not country:
            raise ValueError("Country must be specified when is_holiday is True.")
        try:
            # Use getattr to fetch country class dynamically or country_holidays if available
            if hasattr(holidays, 'country_holidays'):
                country_holidays = holidays.country_holidays(country)
            else:
                # Fallback for older versions if country_holidays not present (though dir showed it is)
                country_holidays = getattr(holidays, country)()
        except Exception as e:
             # Fallback or re-raise if country is invalid
             raise ValueError(f"Invalid country '{country}' for holidays: {e}")
             
        df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x in country_holidays else 0)
        time_covariates.append('is_holiday')
        
    if is_weekend:
        weekend_days = [4, 5, 6] if include_friday_in_weekend else [5, 6]
        df['is_weekend'] = df['timestamp'].dt.dayofweek.isin(weekend_days).astype(int)
        time_covariates.append('is_weekend')

    # Cyclic Encodings for Hour
    # Check if frequency is hourly
    # Handle 'h' or 'H' or '1H' etc.
    if freq and (freq == 'H' or freq == 'h' or 'H' in str(freq) or 'h' in str(freq)):
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        time_covariates.extend(['hour_sin', 'hour_cos'])

    # 5. Future Dataframe (Known Covariates)
    # We use the data after cutoff_date as the future known covariates
    
    if not cutoff_date:
        raise ValueError("cutoff_date is required to split data into history and future for known covariates.")
        
    # Select final columns
    final_cols = ['id_column', 'timestamp', 'y'] + continuous_covariates + categorical_covariates + time_covariates
    df_final = df[final_cols].copy()
    
    # Split into history and future
    data = df_final[df_final['timestamp'] <= cutoff_date].copy()
    future_data = df_final[df_final['timestamp'] > cutoff_date].copy()
    
    if future_data.empty:
        raise ValueError("No data found after cutoff_date. Cannot create future inputs.")
        
    # Use all available future data (known covariates)
    # future_data = future_data.iloc[:forecast_horizon].copy() # Removed slicing
    
    # Mask target in future (avoid leakage)
    future_data['y'] = 0
    
    # Create pred_input
    pred_input = pd.concat([data, future_data], axis=0, ignore_index=True)
    
    # Return data (train), pred_input (train + masked test), test_df (actual test with y), and scaler
    # We need to recreate test_df from df_final split because future_data was modified (y=0)
    # Actually, we can just use the slice from df_final again
    test_df = df_final[df_final['timestamp'] > cutoff_date].copy()
    # Apply scaling to test_df y as well? 
    # Yes, df was scaled in place at line 121: df['y'] = scaler.transform(df[['y']])
    # But wait, df_final was created from df at line 234.
    # df['y'] was scaled at line 121.
    # So df_final has scaled y.
    # So test_df has scaled y.
    
    return data, pred_input, test_df, scaler


def plot_probabilistic_forecast(
    history_df: pd.DataFrame, 
    forecast_df: pd.DataFrame,
    target_col: str = 'y',
    timeseries_name: str = 'Time Series',
    time_col: str = 'timestamp',
    history_length: int = 60,
    scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None,
    actual_df: Optional[pd.DataFrame] = None,
    id_col_name: str = 'ID'
):
    """
    Plot the probabilistic forecast along with historical data and optional actuals.
    
    Supports panel data (multiple series) by creating subplots for each ID.
    If a scaler is provided, the historical data, forecast, and actuals are inverse transformed 
    to the original scale for visualization.

    Args:
        history_df (pd.DataFrame): Historical dataframe containing the target column.
        forecast_df (pd.DataFrame): Forecast dataframe with 'q10', 'q50', 'q90' columns.
        target_col (str, optional): Name of target column. Defaults to 'y'.
        timeseries_name (str, optional): Name of the time series for the title. Defaults to 'Time Series'.
        time_col (str, optional): Name of the time column. Defaults to 'timestamp'.
        history_length (int, optional): Number of historical steps to plot. Defaults to 60.
        scaler (Union[MinMaxScaler, StandardScaler], optional): Scaler used to scale the target column. 
                                                                If provided, data will be inverse transformed.
        actual_df (pd.DataFrame, optional): Dataframe containing actual values for the forecast period.
        id_col_name (str, optional): Name of the ID column to display in the title. Defaults to 'ID'.
    """
    
    # Check for multiple IDs
    id_col = 'id_column'
    unique_ids = ['Default']
    
    if id_col in forecast_df.columns:
        unique_ids = forecast_df[id_col].unique()
    
    num_series = len(unique_ids)
    
    # Setup subplots
    cols = 1
    rows = num_series
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows), sharex=False)
    if rows == 1:
        axes = [axes]
    
    # Inverse transform logic (apply once if possible, or per series?)
    # Scaler is usually global for the target.
    # We should be careful not to inverse transform multiple times if we modify the DFs in place.
    # Let's work with copies inside the loop or inverse transform everything first.
    
    # Better to inverse transform everything first if scaler is provided
    history_df_plot = history_df.copy()
    forecast_df_plot = forecast_df.copy()
    actual_df_plot = actual_df.copy() if actual_df is not None else None
    
    if scaler:
        # Inverse transform target column in history
        history_df_plot[target_col] = scaler.inverse_transform(history_df_plot[[target_col]]).flatten()
        
        # Inverse transform forecast columns
        for q in ['q10', 'q50', 'q90']:
            if q in forecast_df_plot.columns:
                forecast_df_plot[q] = scaler.inverse_transform(forecast_df_plot[[q]]).flatten()
                
        # Inverse transform actuals
        if actual_df_plot is not None:
             actual_df_plot[target_col] = scaler.inverse_transform(actual_df_plot[[target_col]]).flatten()

    for i, series_id in enumerate(unique_ids):
        ax = axes[i]
        
        # Filter data for this series
        if num_series > 1:
            series_history = history_df_plot[history_df_plot[id_col] == series_id].copy()
            series_forecast = forecast_df_plot[forecast_df_plot[id_col] == series_id].copy()
            series_actual = actual_df_plot[actual_df_plot[id_col] == series_id].copy() if actual_df_plot is not None else None
        else:
            series_history = history_df_plot.copy()
            series_forecast = forecast_df_plot.copy()
            series_actual = actual_df_plot.copy() if actual_df_plot is not None else None
            
        # Slice history length
        if len(series_history) > history_length:
            series_history = series_history.iloc[-history_length:]
            
        if len(series_history) == 0:
            print(f"Warning: No history data for ID {series_id}")
            continue

        # Calculate forecast horizon for this series
        forecast_horizon = len(series_forecast)
        if forecast_horizon == 0:
            print(f"Warning: No forecast data for series {series_id}")
            continue

        # Prepare Time Axis
        # Ensure time column is datetime
        if time_col in series_history.columns:
            series_history[time_col] = pd.to_datetime(series_history[time_col])
            x_history = series_history[time_col]
            last_date = series_history[time_col].iloc[-1]
            freq = pd.infer_freq(x_history)
        else:
            # Fallback
            series_history['timestamp'] = pd.to_datetime(series_history['timestamp'])
            x_history = series_history['timestamp']
            last_date = series_history['timestamp'].iloc[-1]
            freq = pd.infer_freq(x_history)
            
        # Infer frequency unit for title
        freq_unit = "steps"
        if freq:
            freq_upper = freq.upper()
            if 'H' in freq_upper: freq_unit = "hours"
            elif 'D' in freq_upper: freq_unit = "days"
            elif 'W' in freq_upper: freq_unit = "weeks"
            elif 'M' in freq_upper: freq_unit = "months"
        else:
             # Fallback heuristic
            if len(x_history) > 1:
                diff = x_history.diff().mode()[0]
                if diff >= pd.Timedelta(days=28): freq_unit = "months"
                elif diff >= pd.Timedelta(days=7): freq_unit = "weeks"
                elif diff >= pd.Timedelta(days=1): freq_unit = "days"
                elif diff >= pd.Timedelta(hours=1): freq_unit = "hours"
            freq = 'D' # Default

        if freq:
            try:
                offset = pd.tseries.frequencies.to_offset(freq)
                start_date = last_date + offset
            except:
                 start_date = last_date + pd.Timedelta(days=1)
        else:
            start_date = last_date + pd.Timedelta(days=1)
            
        # Prepare Forecast Dates
        # Check if actuals have dates to align with
        x_forecast = None
        if series_actual is not None and time_col in series_actual.columns:
            if len(series_actual) >= forecast_horizon:
                 x_forecast = pd.to_datetime(series_actual[time_col].iloc[:forecast_horizon])
                 series_actual = series_actual.iloc[:forecast_horizon] # Slice actuals too
        
        if x_forecast is None:
             future_dates = pd.date_range(
                start=start_date,
                periods=forecast_horizon,
                freq=freq
            )
             x_forecast = future_dates

        # Plot History
        y_history = series_history[target_col]
        ax.plot(x_history, y_history, label='History', color='black', alpha=0.6)
        
        # Connector line
        first_forecast_date = x_forecast.iloc[0] if hasattr(x_forecast, 'iloc') else x_forecast[0]
        ax.plot(
            [x_history.iloc[-1], first_forecast_date], 
            [y_history.iloc[-1], series_forecast["q50"].iloc[0]], 
            color="#1f77b4", linestyle="--", alpha=0.5
        )
        
        # Forecast Median
        ax.plot(x_forecast, series_forecast["q50"], label="Forecast (Median)", color="#1f77b4", linewidth=2)

        # Interval
        if "q10" in series_forecast.columns and "q90" in series_forecast.columns:
            ax.fill_between(
                x_forecast, series_forecast["q10"], series_forecast["q90"], 
                color="#1f77b4", alpha=0.2, label="Confidence (q10-q90)"
            )
            
        # Actuals
        if series_actual is not None:
            y_actual_vals = series_actual[target_col]
            if len(y_actual_vals) == len(x_forecast):
                 ax.plot(x_forecast, y_actual_vals, label="Actual", color="green", linestyle="--", alpha=0.8)
            else:
                 ax.plot(y_actual_vals.values, label="Actual", color="green", linestyle="--", alpha=0.8)

        # Title and Labels
        series_title = f"{timeseries_name} ({id_col_name}: {series_id})" if num_series > 1 else timeseries_name
        ax.set_title(f"{series_title} Forecast: Next {forecast_horizon} {freq_unit}")
        ax.set_xlabel("Time")
        ax.set_ylabel(target_col)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance_results):
    """
    Plot feature importance for past, future, and optionally static inputs in the TFT model.
    
    Visualizes the attention weights or variable selection weights as horizontal bar charts.

    Args:
        importance_results (Tuple[pd.DataFrame, ...]): Tuple containing (past_imp, future_imp) 
                                                       or (past_imp, future_imp, static_imp).
    """
    
    if len(importance_results) == 3:
        past_imp, future_imp, static_imp = importance_results
    else:
        past_imp, future_imp = importance_results
        static_imp = None

    cols = 3 if static_imp is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 5))
    
    # Ensure axes is iterable
    if cols == 1: axes = [axes]
    
    # Plot Past
    axes[0].barh(past_imp["Feature"], past_imp["Importance"], color="#1f77b4")
    axes[0].set_title("Past Input Importance (Encoder)")
    axes[0].set_xlabel("Attention Weight (0-1)")
    axes[0].invert_yaxis() # Highest importance on top

    # Plot Future
    axes[1].barh(future_imp["Feature"], future_imp["Importance"], color="#ff7f0e")
    axes[1].set_title("Future Input Importance (Decoder)")
    axes[1].set_xlabel("Attention Weight (0-1)")
    axes[1].invert_yaxis()
    
    # Plot Static
    if static_imp is not None:
        axes[2].barh(static_imp["Feature"], static_imp["Importance"], color="#2ca02c")
        axes[2].set_title("Static Input Importance")
        axes[2].set_xlabel("Selection Weight (0-1)")
        axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.show()