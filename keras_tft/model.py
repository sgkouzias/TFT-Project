import keras
from keras import layers, ops
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .layers import GatedResidualNetwork, MultivariateVariableSelection, GatedLinearUnit, StaticVariableSelection, InterpretableMultiHeadAttention
from .loss import QuantileLoss

class TFTForecaster:
    """
    Temporal Fusion Transformer (TFT) Forecaster.

    This class implements the TFT architecture for time series forecasting, supporting
    multi-horizon forecasting, static covariates, and interpretable attention mechanisms.
    It wraps the Keras model building, training, and prediction logic.

    Attributes:
        input_len (int): Length of the input sequence (lookback window).
        output_len (int): Length of the output sequence (forecast horizon).
        quantiles (List[float]): List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
        hidden_dim (int): Hidden dimension size for internal layers.
        dropout_rate (float): Dropout rate for regularization.
        num_heads (int): Number of attention heads.
        optimizer_name (str): Name of the optimizer to use.
        learning_rate (float): Learning rate for the optimizer.
        num_past_features (int): Number of past-observed features.
        num_future_features (int): Number of known future features.
        num_static_features (int): Number of static features.
        past_categorical_dict (Dict[int, int]): Dictionary mapping past feature indices to vocab sizes.
        future_categorical_dict (Dict[int, int]): Dictionary mapping future feature indices to vocab sizes.
        static_categorical_dict (Dict[int, int]): Dictionary mapping static feature indices to vocab sizes.
        model (keras.Model): The underlying Keras model.
    """
    def __init__(
        self, 
        input_chunk_length: int, 
        output_chunk_length: int, 
        quantiles: List[float] = [0.1, 0.5, 0.9], 
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        num_heads: int = 4,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        # Feature configs required for building at init
        num_past_features: int = 0,
        num_future_features: int = 0,
        num_static_features: int = 0,
        past_categorical_dict: Dict[int, int] = {}, # {idx: vocab_size}
        future_categorical_dict: Dict[int, int] = {},
        static_categorical_dict: Dict[int, int] = {}
    ):
        """
        Initialize the TFTForecaster.

        Args:
            input_chunk_length (int): Number of past time steps to use as input.
            output_chunk_length (int): Number of future time steps to predict.
            quantiles (List[float], optional): Quantiles to predict. Defaults to [0.1, 0.5, 0.9].
            hidden_dim (int, optional): Hidden dimension size. Defaults to 128.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            optimizer (str, optional): Optimizer name. Defaults to "adam".
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            num_past_features (int, optional): Number of past features. Defaults to 0.
            num_future_features (int, optional): Number of future features. Defaults to 0.
            num_static_features (int, optional): Number of static features. Defaults to 0.
            past_categorical_dict (Dict[int, int], optional): Map of past categorical feature indices to vocab sizes. Defaults to {}.
            future_categorical_dict (Dict[int, int], optional): Map of future categorical feature indices to vocab sizes. Defaults to {}.
            static_categorical_dict (Dict[int, int], optional): Map of static categorical feature indices to vocab sizes. Defaults to {}.
        """
        self.input_len = input_chunk_length
        self.output_len = output_chunk_length
        self.quantiles = quantiles
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        
        # Feature configs
        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.num_static_features = num_static_features
        self.past_categorical_dict = past_categorical_dict
        self.future_categorical_dict = future_categorical_dict
        self.static_categorical_dict = static_categorical_dict
        
        self.model = None
        self.explain_model = None
        self.scalers: Dict[str, Tuple[float, float]] = {}
        self.target_col: Optional[str] = None
        self.past_cov_cols: List[str] = []
        self.future_cov_cols: List[str] = []
        self.static_cov_cols: List[str] = []
        self.feature_cols: List[str] = []
        
        # Build model immediately if feature counts are provided (and > 0 implies intent, though 0 is valid for static)
        # We assume if the user initializes with these, they want it built.
        # If they are 0, we might build a degenerate model or wait?
        # The prompt says "The model should be built upon initialization".
        # So we build it.
        self._build_and_compile_model()

    def _build_and_compile_model(self):
        if self.model is not None:
             return

        num_past_features = self.num_past_features
        num_future_features = self.num_future_features
        num_static_features = self.num_static_features
        
        # --- Inputs ---
        input_past = keras.Input(shape=(self.input_len, num_past_features), name="past_input")
        input_future = keras.Input(shape=(self.output_len, num_future_features), name="future_input")
        
        inputs = [input_past, input_future]
        
        # --- Static Covariate Encoders ---
        if num_static_features > 0:
            input_static = keras.Input(shape=(num_static_features,), name="static_input")
            inputs.append(input_static)
            
            # Static VSN (2D)
            static_embedding, static_weights = StaticVariableSelection(
                num_static_features, self.hidden_dim, self.dropout_rate, name="vsn_static",
                categorical_indices=list(self.static_categorical_dict.keys()),
                vocab_sizes=list(self.static_categorical_dict.values())
            )(input_static)
            
            # Create 4 context vectors
            c_s = layers.Dense(self.hidden_dim, name="static_selection")(static_embedding)
            c_e = layers.Dense(self.hidden_dim, name="static_enrich")(static_embedding)
            c_h = layers.Dense(self.hidden_dim, name="static_h")(static_embedding)
            c_c = layers.Dense(self.hidden_dim, name="static_c")(static_embedding)
            
        else:
            # Zero context if no static features
            # GlobalAveragePooling1D: (Batch, Time, Feat) -> (Batch, Feat)
            dummy = layers.GlobalAveragePooling1D()(input_past) 
            static_embedding = layers.Dense(self.hidden_dim, kernel_initializer='zeros', bias_initializer='zeros')(dummy)
            # Ensure it's zero
            static_embedding = layers.Lambda(lambda x: x * 0)(static_embedding)
            
            c_s = c_e = c_h = c_c = static_embedding
            static_weights = None # No static weights

        # 1. Variable Selection Networks
        
        # Past VSN
        # Use c_s for context
        if num_past_features > 0:
            x_past, past_weights = MultivariateVariableSelection(
                num_past_features, self.hidden_dim, self.dropout_rate, name="vsn_past",
                categorical_indices=list(self.past_categorical_dict.keys()),
                vocab_sizes=list(self.past_categorical_dict.values())
            )([input_past, c_s])
        else:
            # Dummy zero output
            x_past = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], self.input_len, self.hidden_dim)))(input_past)
            past_weights = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], 0)))(input_past)
        
        # Future VSN
        # Use c_s for context
        if num_future_features > 0:
            x_fut, future_weights = MultivariateVariableSelection(
                num_future_features, self.hidden_dim, self.dropout_rate, name="vsn_future",
                categorical_indices=list(self.future_categorical_dict.keys()),
                vocab_sizes=list(self.future_categorical_dict.values())
            )([input_future, c_s])
        else:
            # Dummy zero output
            x_fut = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], self.output_len, self.hidden_dim)))(input_future)
            future_weights = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], 0)))(input_future)

        # 2. LSTM Encoder-Decoder (Seq2Seq)
        lstm_layer = layers.LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        
        # Initialize LSTM state with c_h, c_c

        # Static Enrichment for Past (use c_e)
        x_past = GatedResidualNetwork(self.hidden_dim, self.dropout_rate, name="grn_enrich_past")([x_past, c_e])
        
        # Run LSTM on Past
        encoder_out, state_h, state_c = lstm_layer(x_past, initial_state=[c_h, c_c])
        
        # Post-LSTM Gate (GLU) + Add + Norm
        encoder_out = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(encoder_out)
        encoder_out = layers.LayerNormalization()(encoder_out + x_past) # Residual from x_past

        # Static Enrichment for Future (use c_e)
        x_fut = GatedResidualNetwork(self.hidden_dim, self.dropout_rate, name="grn_enrich_fut")([x_fut, c_e])

        # Run LSTM on Future
        # We initialize with encoder state
        decoder_out, _, _ = lstm_layer(x_fut, initial_state=[state_h, state_c])
        
        # Post-LSTM Gate + Add + Norm
        decoder_out = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(decoder_out)
        decoder_out = layers.LayerNormalization()(decoder_out + x_fut)

        # 3. Multi-Head Attention (Interpretable)
        # Returns (output, weights) if return_attention_scores=True
        attn_layer = InterpretableMultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.hidden_dim, dropout=self.dropout_rate
        )
        attn_out, attn_weights = attn_layer(
            query=decoder_out, value=encoder_out, key=encoder_out, return_attention_scores=True
        )
        
        # Gating for Attention output
        attn_out = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(attn_out)
        attn_out = layers.LayerNormalization()(attn_out + decoder_out) # Residual from decoder_out
        
        # 4. Position-wise Feed Forward (GRN)
        output_grn = GatedResidualNetwork(self.hidden_dim, self.dropout_rate, name="grn_output")
        outputs = output_grn(attn_out)
        
        # Final Gate + Add + Norm
        outputs = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(outputs)
        outputs = layers.LayerNormalization()(outputs + attn_out)

        # 5. Output Head (Quantiles)
        output_dim = len(self.quantiles)
        predictions = layers.Dense(output_dim)(outputs) 
        
        self.model = keras.Model(inputs=inputs, outputs=predictions, name="TemporalFusionTransformer")
        
        # --- Explainability Model ---
        # Outputs: past_weights, future_weights, static_weights (if any), attention_scores
        explain_outputs = {
            "past_weights": past_weights,
            "future_weights": future_weights,
            "attention_scores": attn_weights
        }
        if static_weights is not None:
            explain_outputs['static_weights'] = static_weights
            
        self.explain_model = keras.Model(inputs=inputs, outputs=explain_outputs)
        
        # Configure Optimizer with clipnorm
        if self.optimizer_name.lower() == "adam":
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        elif self.optimizer_name.lower() == "rmsprop":
            opt = keras.optimizers.RMSprop(learning_rate=self.learning_rate, clipnorm=1.0)
        elif self.optimizer_name.lower() == "sgd":
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate, clipnorm=1.0)
        else:
            try:
                 opt_cls = getattr(keras.optimizers, self.optimizer_name)
                 opt = opt_cls(learning_rate=self.learning_rate, clipnorm=1.0)
            except:
                 print(f"Warning: Could not instantiate optimizer '{self.optimizer_name}' with learning_rate. Using default.")
                 opt = self.optimizer_name

        self.model.compile(optimizer=opt, loss=QuantileLoss(self.quantiles))
        # self.explain_model is already set? No, we need to set it.
        self.explain_model = keras.Model(inputs=inputs, outputs=explain_outputs)


    def get_feature_importance(self, df: pd.DataFrame):
        """
        Extract feature importance using the explainability model.

        Calculates the average attention weights for past, future, and static features
        over a sample of the provided DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to calculate importance from.

        Returns:
            Tuple[pd.DataFrame, ...]: DataFrames containing feature importance scores for 
            past, future, and (optionally) static features.
        
        Raises:
            ValueError: If the model is not fitted or explainability model is unavailable.
        """
        if self.model is None or not hasattr(self, 'explain_model'):
            raise ValueError("Model not fitted or explainability unavailable.")
        
        # Prepare data (limit to 100 samples)
        matrix_df = self._scale_matrix(df, fit=False)
        matrix_vals = matrix_df.values
        
        col_to_idx = {name: i for i, name in enumerate(matrix_df.columns)}
        past_cols = [self.target_col] + self.past_cov_cols + self.future_cov_cols
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols
        
        past_idxs = [col_to_idx[c] for c in past_cols]
        fut_idxs = [col_to_idx[c] for c in fut_cols]
        static_idxs = [col_to_idx[c] for c in static_cols] if static_cols else []
        
        X_past, X_fut, X_static = [], [], []
        num_total = len(matrix_df) - self.input_len - self.output_len
        if num_total <= 0:
             raise ValueError("Dataframe too short for importance calculation.")
             
        num_samples = min(100, num_total)
        step_size = max(1, num_total // num_samples)
        
        for i in range(0, num_total, step_size):
            if len(X_past) >= num_samples: break
            X_past.append(matrix_vals[i:i+self.input_len, past_idxs])
            X_fut.append(matrix_vals[i+self.input_len:i+self.input_len+self.output_len, fut_idxs])
            if static_idxs:
                X_static.append(matrix_vals[i, static_idxs])
        
        inputs = [np.array(X_past), np.array(X_fut)]
        if static_idxs:
            inputs.append(np.array(X_static))
        
        # Get weights from explainability model
        outputs = self.explain_model.predict(inputs, verbose=0)
        
        # Average and format
        # outputs['past_weights'] shape: (Batch, Features) or (Batch, Time, Features)?
        # In MultivariateVariableSelection, we return `weights` which is (Batch, Features) because of temporal averaging.
        # So np.mean(axis=0) is correct.
        
        past_imp = pd.DataFrame({
            "Feature": past_cols, 
            "Importance": np.mean(outputs['past_weights'], axis=0)
        }).sort_values("Importance", ascending=False)
        
        fut_imp = pd.DataFrame({
            "Feature": fut_cols,
            "Importance": np.mean(outputs['future_weights'], axis=0)
        }).sort_values("Importance", ascending=False)
        
        if 'static_weights' in outputs and outputs['static_weights'] is not None:
            static_imp = pd.DataFrame({
                "Feature": static_cols,
                "Importance": np.mean(outputs['static_weights'], axis=0)
            }).sort_values("Importance", ascending=False)
            return past_imp, fut_imp, static_imp
        
        return past_imp, fut_imp

    def _scale_matrix(self, df: pd.DataFrame, fit: bool = False, categorical_cols: set = None) -> pd.DataFrame:
        """
        Scales the input DataFrame using standard scaling (mean/std).
        
        Args:
            df (pd.DataFrame): Input DataFrame to scale.
            fit (bool, optional): Whether to fit the scalers on this data. Defaults to False.
            categorical_cols (set, optional): Set of column names to exclude from scaling. Defaults to None.

        Returns:
            pd.DataFrame: Scaled DataFrame.

        Raises:
            ValueError: If a column is not found in fitted scalers (when fit=False) or if a categorical column is not numeric.
        """
        if categorical_cols is None: categorical_cols = set()
        
        data_dict = {}
        for col in self.feature_cols:
            if col in categorical_cols:
                # Do not scale, keep as is (assuming already numeric/encoded)
                # Ensure it's numeric for numpy array conversion later
                # If it's string, we might need LabelEncoder, but for now assume pre-encoded or numeric ID
                try:
                    val = df[col].values.astype(float)
                except ValueError:
                    # If string, try to encode? Or just raise error?
                    # For now, let's assume user provides numeric IDs as per instructions
                    # But if they provide strings, we could map them?
                    # Let's just try to keep as object if conversion fails, but downstream expects float/int array
                    # Actually, TFT expects integer inputs for embeddings.
                    # So we should cast to int?
                    # But _scale_matrix returns a single DF, usually float.
                    # If we mix types, matrix_vals will be object.
                    # This might break numpy slicing if not careful.
                    # Let's assume they are numeric IDs.
                    raise ValueError(f"Categorical column {col} must be numeric (integer IDs).")
                
                data_dict[col] = val
                # We don't add to self.scalers
            else:
                val = df[col].values.astype(float)
                if fit:
                    mean, std = np.mean(val), np.std(val)
                    if std == 0: std = 1e-7
                    self.scalers[col] = (mean, std)
                
                if col not in self.scalers:
                     # If not in scalers (e.g. fit=False and not seen before), raise error
                     raise ValueError(f"Column {col} not found in fitted scalers.")
                    
                mean, std = self.scalers[col]
                data_dict[col] = (val - mean) / (std + 1e-7)
            
        return pd.DataFrame(data_dict, index=df.index)

    def fit(self, df, target_col, past_cov_cols=None, future_cov_cols=None, static_cov_cols=None, exogenous=None,
            epochs=10, batch_size=32, verbose=1,
            validation_split=0.0,
            use_lr_schedule=True, 
            use_early_stopping=False, early_stopping_patience=10):
        """
        Train the TFT model.

        Args:
            df (pd.DataFrame): Training data containing target, covariates, and static features.
            target_col (str): Name of the target column.
            past_cov_cols (List[str], optional): List of past-observed covariate column names. Defaults to None.
            future_cov_cols (List[str], optional): List of known future covariate column names. Defaults to None.
            static_cov_cols (List[str], optional): List of static covariate column names. Defaults to None.
            exogenous (List[str], optional): Alias for future_cov_cols. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 32.
            verbose (int, optional): Verbosity mode. Defaults to 1.
            validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.0.
            use_lr_schedule (bool, optional): Whether to use ReduceLROnPlateau callback. Defaults to True.
            use_early_stopping (bool, optional): Whether to use EarlyStopping callback. Defaults to False.
            early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
        """
        
        self.target_col = target_col
        
        if exogenous is not None:
            # If exogenous is provided, we treat them as future covariates (known in future)
            # unless specified otherwise. This simplifies the API.
            self.future_cov_cols = exogenous
            self.past_cov_cols = [] # Implicitly included in past input via future covs
        else:
            self.past_cov_cols = past_cov_cols if past_cov_cols else []
            self.future_cov_cols = future_cov_cols if future_cov_cols else []
            
        self.static_cov_cols = static_cov_cols if static_cov_cols else []
        
        # All columns needed
        self.feature_cols = list(set([target_col] + self.past_cov_cols + self.future_cov_cols + self.static_cov_cols))
        
        # Identify categorical columns to skip scaling
        # We map indices from *categorical_dict to column names
        # Need to determine past_cols and fut_cols first to map indices
        # Past Input: Target + Past Covs + Future Covs (as observed in past)
        # Use dict.fromkeys to preserve order and remove duplicates
        past_cols = list(dict.fromkeys([target_col] + self.past_cov_cols + self.future_cov_cols))
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols
        
        categorical_cols = set()
        
        # Static
        for idx in self.static_categorical_dict.keys():
            if idx < len(static_cols):
                categorical_cols.add(static_cols[idx])
                
        # Past
        for idx in self.past_categorical_dict.keys():
            if idx < len(past_cols):
                categorical_cols.add(past_cols[idx])
                
        # Future
        for idx in self.future_categorical_dict.keys():
            if idx < len(fut_cols):
                categorical_cols.add(fut_cols[idx])

        # Scale
        matrix_df = self._scale_matrix(df, fit=True, categorical_cols=categorical_cols)
        
        # Create Windows
        X_past_list, X_fut_list, X_static_list, y_list = [], [], [], []
        
        matrix_vals = matrix_df.values
        col_to_idx = {name: i for i, name in enumerate(matrix_df.columns)}
        
        past_idxs = [col_to_idx[c] for c in past_cols]
        fut_idxs = [col_to_idx[c] for c in fut_cols]
        static_idxs = [col_to_idx[c] for c in static_cols]
        target_idx = col_to_idx[target_col]
        
        # Helper to create windows from a single series array
        # Pre-allocate arrays
        # We need to count total samples first to pre-allocate
        total_samples = 0
        
        if self.static_cov_cols:
             grouped = matrix_df.groupby(self.static_cov_cols)
             groups = [group_df.values for _, group_df in grouped]
        else:
             groups = [matrix_vals]
             
        for series_vals in groups:
            n = len(series_vals)
            if n >= self.input_len + self.output_len:
                total_samples += n - (self.input_len + self.output_len) + 1
                
        if total_samples == 0:
             raise ValueError("No valid windows created. Data might be too short.")

        X_past = np.empty((total_samples, self.input_len, len(past_cols)))
        X_fut = np.empty((total_samples, self.output_len, len(fut_cols)))
        X_static = np.empty((total_samples, len(static_cols))) if static_cols else None
        y = np.empty((total_samples, self.output_len))
        
        idx = 0
        for series_vals in groups:
            n = len(series_vals)
            if n < self.input_len + self.output_len:
                continue
                
            num_windows = n - (self.input_len + self.output_len) + 1
            
            # Vectorized window creation (stride_tricks could be faster but complex)
            # Simple loop is faster than append
            for i in range(num_windows):
                X_past[idx] = series_vals[i : i+self.input_len, past_idxs]
                X_fut[idx] = series_vals[i+self.input_len : i+self.input_len+self.output_len, fut_idxs]
                if static_idxs:
                    X_static[idx] = series_vals[i, static_idxs]
                y[idx] = series_vals[i+self.input_len : i+self.input_len+self.output_len, target_idx]
                idx += 1
                
        y = y[..., np.newaxis] # Expand for quantiles
    
        # Check if we need to rebuild due to feature count mismatch
        current_past = self.num_past_features
        current_future = self.num_future_features
        current_static = self.num_static_features
        
        new_past = len(past_cols)
        new_future = len(fut_cols)
        new_static = len(static_cols)
        
        if (current_past != new_past or 
            current_future != new_future or 
            current_static != new_static):

            if verbose > 0:
                print(f"WARNING: Feature counts changed. Rebuilding model. "
                      f"Past: {current_past}->{new_past}, Future: {current_future}->{new_future}, Static: {current_static}->{new_static}. "
                      f"This will reset model weights.")
            
            self.num_past_features = new_past
            self.num_future_features = new_future
            self.num_static_features = new_static
            
            self.model = None
            self.explain_model = None
            
            self._build_and_compile_model()
        
        if self.model is None:
             # If model was not built at init (e.g. 0 features passed), try to build now?
             # But we enforced init args.
             # If user passed 0 at init but now has data, we might need to rebuild or error.
             # Let's allow rebuilding if not built or if built with 0 features and now we have more?
             # Simpler: If model is None, build it.
             self.num_past_features = len(past_cols)
             self.num_future_features = len(fut_cols)
             self.num_static_features = len(static_cols)
             self._build_and_compile_model()
            
        if verbose > 0:
            print(f"Training on {len(X_past)} samples.\npast-covariates: {len(past_cols)} | future-covariates: {len(fut_cols)} | static-covariates: {len(static_cols)}")
        
        inputs = [X_past, X_fut]
        if static_cols:
            inputs.append(X_static)
        
        # Callbacks
        callbacks = []
        if use_lr_schedule:
            lr_schedule = keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=verbose
            )
            callbacks.append(lr_schedule)
        
        if use_early_stopping:
            monitor = 'val_loss' if validation_split > 0 else 'loss'
            early_stop = keras.callbacks.EarlyStopping(
                monitor=monitor, patience=early_stopping_patience, restore_best_weights=True, verbose=verbose
            )
            callbacks.append(early_stop)
            
        self.model.fit(inputs, y, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                       validation_split=validation_split, callbacks=callbacks)

    def summary(self):
        """
        Prints the summary of the underlying Keras model.
        
        If the model has not been built yet, prints a message indicating so.
        """
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call fit() first.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate probabilistic forecasts for the given DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing historical data and future covariates.
                               Must include the target column (for history) and all covariate columns.
                               For future covariates, values must be provided for the forecast horizon.

        Returns:
            pd.DataFrame: DataFrame containing the forecasts (quantiles) and the timestamp index.
                          Columns: 'q10', 'q50', 'q90' (depending on quantiles), and 'id_column' (if panel data).
        
        Raises:
            ValueError: If the model is not fitted or if input data is insufficient.
        """
        if self.model is None: raise ValueError("Model not fitted.")
        
        # Re-derive categorical cols (should refactor this into a method)
        past_cols = list(dict.fromkeys([self.target_col] + self.past_cov_cols + self.future_cov_cols))
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols
        
        categorical_cols = set()
        for idx in self.static_categorical_dict.keys():
            if idx < len(static_cols): categorical_cols.add(static_cols[idx])
        for idx in self.past_categorical_dict.keys():
            if idx < len(past_cols): categorical_cols.add(past_cols[idx])
        for idx in self.future_categorical_dict.keys():
            if idx < len(fut_cols): categorical_cols.add(fut_cols[idx])
                
        # Scale
        matrix_df = self._scale_matrix(df, fit=False, categorical_cols=categorical_cols)
        matrix_vals = matrix_df.values
        col_to_idx = {name: i for i, name in enumerate(matrix_df.columns)}
        
        past_idxs = [col_to_idx[c] for c in past_cols]
        fut_idxs = [col_to_idx[c] for c in fut_cols]
        static_idxs = [col_to_idx[c] for c in static_cols]
        
        # Helper to predict for a single sequence
        def predict_single(seq_vals):
            total_required = self.input_len + self.output_len
            if len(seq_vals) < total_required:
                # Pad with zeros or error?
                # Error is safer.
                raise ValueError(f"Sequence length {len(seq_vals)} < required {total_required}")
                
            # 1. Past: The data BEFORE the forecast horizon
            past_seq = seq_vals[-(self.input_len + self.output_len) : -self.output_len, past_idxs]
            
            # 2. Future: The data DURING the forecast horizon
            fut_seq = seq_vals[-self.output_len:, fut_idxs]
            
            # 3. Static: Take from the last row
            if static_idxs:
                static_seq = seq_vals[-1, static_idxs]
            else:
                static_seq = np.zeros(0)
            
            # Add batch dim
            past_seq = past_seq[np.newaxis, ...]
            fut_seq = fut_seq[np.newaxis, ...]
            
            inputs = [past_seq, fut_seq]
            if static_cols:
                static_seq = static_seq[np.newaxis, ...]
                inputs.append(static_seq)
            
            pred_scaled = self.model.predict(inputs, verbose=0)
            
            # Inverse Scale
            mean, std = self.scalers[self.target_col]
            pred_actual = (pred_scaled * (std + 1e-7)) + mean
            
            return pred_actual[0] # (Output_Len, Quantiles)

        results_list = []
        
        if self.static_cov_cols:
            # Group by original DF to preserve ID types/values
            grouped = df.groupby(self.static_cov_cols)
            for group_name, group_df in grouped:
                if len(group_df) < self.input_len + self.output_len:
                    print(f"Warning: Skipping group {group_name} - insufficient data ({len(group_df)} < {self.input_len + self.output_len})")
                    continue

                # Get scaled values using the index
                scaled_group_vals = matrix_df.loc[group_df.index].values
                pred = predict_single(scaled_group_vals)
                
                # Create result dict for this group
                res = {}
                for i, q in enumerate(self.quantiles):
                    res[f"q{int(q*100)}"] = pred[:, i]
                
                res_df = pd.DataFrame(res)
                # Add Timestamp Index (corresponding to the future horizon)
                res_df.index = group_df.index[-self.output_len:]
                
                # Add ID columns
                if isinstance(group_name, tuple):
                    for idx, col in enumerate(self.static_cov_cols):
                        res_df[col] = group_name[idx]
                else:
                    res_df[self.static_cov_cols[0]] = group_name
                    
                results_list.append(res_df)
                
            final_df = pd.concat(results_list, ignore_index=True)
            return final_df
            
        else:
            # Single Series
            pred = predict_single(matrix_vals)
            results = {}
            for i, q in enumerate(self.quantiles):
                results[f"q{int(q*100)}"] = pred[:, i]
            
            res_df = pd.DataFrame(results)
            res_df.index = df.index[-self.output_len:]
            return res_df
