import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import torch
from time import time
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import signal
import os
from datetime import datetime
import json
from scipy import stats
import matplotlib.patches as mpatches
from sklearn.feature_selection import VarianceThreshold

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Dataset processing timeout")

class ASKLLM2:
    def __init__(self, use_gpu=False, n_trials=30, timeout_per_dataset=3600):  # 1 hour timeout per dataset
        self.results = []
        self.datasets_info = []
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.n_trials = n_trials
        self.best_params = {}
        self.timeout_per_dataset = timeout_per_dataset
        self.models = {
            'rf': RandomForestClassifier,
            'xgb': XGBClassifier,
            'lgb': LGBMClassifier,
            'cat': CatBoostClassifier,
            'svm': SVC
        }
        
        # Create directory for partial results
        self.results_dir = 'partial_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load partial results if they exist
        self.load_partial_results()
        
    def load_partial_results(self):
        """Loads previously saved partial results"""
        try:
            if os.path.exists(f'{self.results_dir}/results.json'):
                with open(f'{self.results_dir}/results.json', 'r') as f:
                    self.results = json.load(f)
                print(f"Loaded {len(self.results)} partial results")
        except Exception as e:
            print(f"Error loading partial results: {str(e)}")

    def save_partial_results(self):
        """Saves partial results"""
        try:
            with open(f'{self.results_dir}/results.json', 'w') as f:
                json.dump(self.results, f)
            print(f"Partial results saved ({len(self.results)} results)")
        except Exception as e:
            print(f"Error saving partial results: {str(e)}")
    
    def detect_column_types(self, X):
        """Identifies numeric and categorical columns"""
        # Convert to DataFrame if it's not
        X = pd.DataFrame(X)
        
        # Convert all categorical columns to string
        cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns
        for col in cat_cols:
            X[col] = X[col].astype(str)
        
        # Identify numeric columns (after conversion)
        numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        categorical_cols = X.select_dtypes(include=['object', 'string']).columns
        
        # If no columns are found, try converting all to numeric
        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            try:
                X = X.astype(float)
                numeric_cols = X.columns
            except:
                try:
                    X = X.astype(str)
                    categorical_cols = X.columns
                except:
                    pass
        
        return numeric_cols, categorical_cols
    
    def preprocess_data(self, X, y):
        """Advanced data preprocessing with robust handling for categorical columns"""
        # Ensure X is a DataFrame
        X = pd.DataFrame(X)
        
        print(f"\nPreprocessing dataset with {X.shape[1]} original features")
        
        # Convert the target to string if it's categorical
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.astype(str)
        y = np.array(y).astype(str)
        
        # Identify column types
        numeric_cols, categorical_cols = self.detect_column_types(X)
        
        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        # Check if there are columns to process
        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            raise ValueError("No valid columns found for processing")
        
        transformers = []
        
        # Preprocessing for numeric columns
        if len(numeric_cols) > 0:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            transformers.append(('num', numeric_transformer, numeric_cols))
        
        # Preprocessing for categorical columns
        if len(categorical_cols) > 0:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            transformers.append(('cat', categorical_transformer, categorical_cols))
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough')  # Keep unspecified columns
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        
        # Create feature names for processed columns
        feature_names = []
        
        # Add names for numeric features
        if len(numeric_cols) > 0:
            feature_names.extend(numeric_cols)
        
        # Add names for one-hot encoded categorical features
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[list(categorical_cols).index(col)]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        # Add names for remaining features
        if preprocessor.remainder == 'passthrough':
            remaining_cols = [col for col in X.columns if col not in numeric_cols and col not in categorical_cols]
            feature_names.extend(remaining_cols)
        
        # Convert to DataFrame with feature names
        X_processed = pd.DataFrame(X_processed, columns=feature_names)
        
        # Apply additional normalization for SVM
        if self.models.get('svm') is not None:
            # Apply MinMaxScaler to ensure all values are between 0 and 1
            scaler = MinMaxScaler()
            X_processed = pd.DataFrame(
                scaler.fit_transform(X_processed),
                columns=X_processed.columns
            )
            
            # Remove features with zero or very low variance
            selector = VarianceThreshold(threshold=1e-5)
            X_processed = pd.DataFrame(
                selector.fit_transform(X_processed),
                columns=X_processed.columns[selector.get_support()]
            )
            
            print(f"Features after low variance removal: {X_processed.shape[1]}")
        
        print(f"Features after preprocessing: {X_processed.shape[1]}")
        
        # Check if we still have features
        if X_processed.shape[1] == 0:
            raise ValueError("All features were lost during preprocessing")
        
        # Label encoding
        le = LabelEncoder()
        y_processed = le.fit_transform(y)
        
        print(f"Unique classes in target: {np.unique(y_processed)}")
            
        return X_processed, y_processed
    
    def get_model_params(self, trial, model_name):
        """Defines the hyperparameter search space for each model"""
        if model_name == 'rf':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),  # Increased range
                'max_depth': trial.suggest_int('max_depth', 3, 50),  # Increased range
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),  # Increased range
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        elif model_name == 'xgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),  # Increased range
                'max_depth': trial.suggest_int('max_depth', 3, 20),  # Increased range
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),  # Reduced minimum
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Reduced minimum
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Reduced minimum
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # Increased range
                'gamma': trial.suggest_float('gamma', 0, 10)  # Increased range
            }
        elif model_name == 'lgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),  # Increased range
                'max_depth': trial.suggest_int('max_depth', 3, 30),  # Increased range
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),  # Reduced minimum
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),  # Increased range
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Reduced minimum
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Reduced minimum
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),  # Increased range
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 20.0, log=True),  # Increased range
                'min_split_gain': trial.suggest_float('min_split_gain', 1e-3, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 20.0, log=True),  # Increased range
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),  # Increased range
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),  # Reduced minimum
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),  # Reduced minimum
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),  # Increased range
                'verbose': -1
            }
        elif model_name == 'cat':
            # First choose the bootstrap type
            bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
            
            # Base parameters with more conservative ranges
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),  # Reduced max range
                'depth': trial.suggest_int('depth', 3, 10),  # Reduced max range
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Increased minimum
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),  # Reduced max range
                'bootstrap_type': bootstrap_type,
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),  # Reduced max range
                'od_type': 'Iter',
                'od_wait': trial.suggest_int('od_wait', 10, 50),  # Reduced max range
                'thread_count': -1,  # Use all available cores
                'task_type': 'CPU',  # Force CPU usage
                'boosting_type': 'Plain',  # Faster boosting type
                'grow_policy': 'SymmetricTree',  # Faster growth policy
                'max_bin': 200,  # Reduce number of bins for higher speed
                'verbose': False
            }
            
            # Add bagging_temperature only if bootstrap_type is Bayesian
            if bootstrap_type == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)  # Reduced max range
            
            return params
        elif model_name == 'svm':
            # Adjusting SVM parameters for better convergence
            return {
                'C': trial.suggest_float('C', 1e-3, 1e1, log=True),  # More conservative range
                'gamma': trial.suggest_float('gamma', 1e-3, 1e1, log=True),  # More conservative range
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'cache_size': 2000,  # Increased for better performance
                'probability': True,
                'max_iter': 20000,  # Increased to give more chance for convergence
                'tol': 1e-2,  # Increased for faster convergence
                'class_weight': 'balanced',  # To handle imbalanced classes
                'shrinking': True,  # Activate shrinking for faster convergence
                'decision_function_shape': 'ovr'  # Use one-vs-rest for better performance
            }
        
    def optimize_hyperparameters(self, X, y, model_name, n_splits=10):
        """Optimizes hyperparameters using Optuna with cross-validation"""
        print(f"\n🔍 Optimizing hyperparameters for {model_name.upper()}...")
        start_time = time()
        
        def train_with_timeout(model, X_train, y_train, timeout=180):  # 3 minutes per fold
            def train():
                if model_name == 'cat':
                    model.fit(X_train, y_train, verbose=True)
                else:
                    model.fit(X_train, y_train)
            
            # Configure timeout for training
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                train()
                signal.alarm(0)  # Disable timeout
                return True
            except TimeoutError:
                print(f"⚠️ Timeout in training after {timeout} seconds")
                signal.alarm(0)  # Disable timeout
                return False
            except Exception as e:
                print(f"Error in training: {str(e)}")
                signal.alarm(0)  # Disable timeout
                return False
        
        def objective(trial):
            trial_start = time()
            print(f"\nStarting trial {trial.number} for {model_name.upper()}")
            
            # Get parameters for the specific model
            params = self.get_model_params(trial, model_name)
            print(f"Parameters of trial {trial.number}: {params}")
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                fold_start = time()
                print(f"  Fold {fold}/{n_splits} - Trial {trial.number}")
                
                # Use iloc for position-based indexing
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y[val_idx]
                
                # Train model with timeout
                model = self.models[model_name](**params, random_state=42)
                if not train_with_timeout(model, X_train_fold, y_train_fold):
                    print(f"  ⚠️ Skipping fold {fold} due to timeout")
                    continue
                
                # Evaluate
                y_pred = model.predict(X_val_fold)
                score = f1_score(y_val_fold, y_pred, average='weighted')
                cv_scores.append(score)
            
                fold_time = time() - fold_start
                print(f"  Fold {fold} completed in {fold_time:.2f}s - Score: {score:.4f}")
            
            if not cv_scores:  # If all folds failed
                return float('-inf')
            
            trial_time = time() - trial_start
            mean_score = np.mean(cv_scores)
            print(f"Trial {trial.number} completed in {trial_time:.2f}s - Mean score: {mean_score:.4f}")
            
            return mean_score
        
        # Create Optuna study with timeout
        study = optuna.create_study(direction='maximize')
        
        # Configure timeout for optimization
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3600)  # 1 hour timeout for optimization
        
        try:
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        except TimeoutError:
            print(f"\n⚠️ Timeout in hyperparameter optimization for {model_name.upper()} after 1 hour")
            if len(study.trials) > 0:
                print("Using the best parameters found so far")
            else:
                print("No trial completed. Using default parameters.")
                return self.get_model_params(optuna.trial.Trial(None, None), model_name)
        finally:
            signal.alarm(0)  # Disable timeout
        
        total_time = time() - start_time
        print(f"\n✅ Optimization of {model_name.upper()} completed in {total_time:.2f}s")
        print(f"Best hyperparameters found:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
        
        return study.best_params

    def evaluate_model(self, X_train, X_test, y_train, y_test, model_name, best_params):
        """Evaluates the model with the best hyperparameters"""
        print(f"\nTraining {model_name.upper()} with best hyperparameters...")
        start_time = time()
        
        # Configure model with best hyperparameters
        if model_name == 'svm':
            best_params['probability'] = True
        model = self.models[model_name](**best_params, random_state=42)
        
        # Cross-validation for robust evaluation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = {
            'accuracy': [],
            'auc_ovo': [],
            'cross_entropy': [],
            'time': []
        }
        
        print("\nPerforming cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            print(f"\nFold {fold}/5")
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train[val_idx]
            
            fold_start = time()
            
            # Train model
            if model_name == 'cat':
                model.fit(X_train_fold, y_train_fold, verbose=False)
            else:
                model.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            y_pred = model.predict(X_val_fold)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            
            # Calculate AUROC OVO
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val_fold)
                cv_scores['auc_ovo'].append(roc_auc_score(y_val_fold, y_proba, multi_class='ovo', average='weighted'))
                cv_scores['cross_entropy'].append(log_loss(y_val_fold, y_proba))
            else:
                cv_scores['auc_ovo'].append(np.nan)
                cv_scores['cross_entropy'].append(np.nan)
            
            fold_time = time() - fold_start
            cv_scores['time'].append(fold_time)
            
            # Format output string
            print(f"Fold {fold} - Accuracy: {cv_scores['accuracy'][-1]:.4f}, "
                  f"AUROC OVO: {cv_scores['auc_ovo'][-1]:.4f if not np.isnan(cv_scores['auc_ovo'][-1]) else 'N/A'}, "
                  f"CE: {cv_scores['cross_entropy'][-1]:.4f if not np.isnan(cv_scores['cross_entropy'][-1]) else 'N/A'}, "
                  f"Time: {fold_time:.2f}s")
        
        # Train final model with all training data
        final_start = time()
        if model_name == 'cat':
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
        train_time = time() - final_start
        
        # Evaluation on the training set
        train_start = time()
        y_train_pred = model.predict(X_train)
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)
            train_auc_ovo = roc_auc_score(y_train, y_train_proba, multi_class='ovo', average='weighted')
            train_ce = log_loss(y_train, y_train_proba)
        else:
            train_auc_ovo = np.nan
            train_ce = np.nan
        train_predict_time = time() - train_start
        
        # Evaluation on the test set
        test_start = time()
        y_test_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)
            test_auc_ovo = roc_auc_score(y_test, y_test_proba, multi_class='ovo', average='weighted')
            test_ce = log_loss(y_test, y_test_proba)
        else:
            test_auc_ovo = np.nan
            test_ce = np.nan
        test_predict_time = time() - test_start
        
        # Final metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        total_time = train_time + train_predict_time + test_predict_time
        
        # Print results
        print("\nCross-Validation Results:")
        print(f"Accuracy: {np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}")
        print(f"AUROC OVO: {np.mean(cv_scores['auc_ovo']):.4f} ± {np.std(cv_scores['auc_ovo']):.4f}")
        print(f"Cross-Entropy: {np.mean(cv_scores['cross_entropy']):.4f} ± {np.std(cv_scores['cross_entropy']):.4f}")
        print(f"Average time: {np.mean(cv_scores['time']):.2f}s ± {np.std(cv_scores['time']):.2f}s")
        
        print("\nResults on Training Set:")
        print(f"Accuracy: {train_acc:.4f}")
        print(f"AUROC OVO: {train_auc_ovo:.4f if not np.isnan(train_auc_ovo) else 'N/A'}")
        print(f"Cross-Entropy: {train_ce:.4f if not np.isnan(train_ce) else 'N/A'}")
        print(f"Time (train + predict): {train_time + train_predict_time:.2f}s")
        
        print("\nResults on Test Set:")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"AUROC OVO: {test_auc_ovo:.4f if not np.isnan(test_auc_ovo) else 'N/A'}")
        print(f"Cross-Entropy: {test_ce:.4f if not np.isnan(test_ce) else 'N/A'}")
        print(f"Time (predict): {test_predict_time:.2f}s")
        
        print(f"\nTotal time (tune + train + predict): {total_time:.2f}s")
        
        return {
            'train_accuracy': train_acc,
            'train_auc_ovo': train_auc_ovo,
            'train_cross_entropy': train_ce,
            'test_accuracy': test_acc,
            'test_auc_ovo': test_auc_ovo,
            'test_cross_entropy': test_ce,
            'time': total_time,
            'cv_accuracy_mean': np.mean(cv_scores['accuracy']),
            'cv_accuracy_std': np.std(cv_scores['accuracy']),
            'cv_auc_ovo_mean': np.mean(cv_scores['auc_ovo']),
            'cv_auc_ovo_std': np.std(cv_scores['auc_ovo']),
            'cv_ce_mean': np.mean(cv_scores['cross_entropy']),
            'cv_ce_std': np.std(cv_scores['cross_entropy']),
            'cv_time_mean': np.mean(cv_scores['time']),
            'cv_time_std': np.std(cv_scores['time'])
        }

    def perform_demsar_test(self, results_df, metric='accuracy'):
        """Performs Demšar's statistical test without dependency on Orange"""
        # Mapping metrics to column names
        metric_to_column = {
            'accuracy': 'test_accuracy',
            'auc_ovo': 'test_auc_ovo',
            'cross_entropy': 'test_cross_entropy',
            'time': 'time'
        }
        
        # Use the correct column name
        column_name = metric_to_column.get(metric, metric)
        
        # Prepare data for the test
        datasets = results_df['dataset_id'].unique()
        models = results_df['model'].unique()
        
        # Create results matrix
        results_matrix = np.zeros((len(datasets), len(models)))
        for i, dataset in enumerate(datasets):
            for j, model in enumerate(models):
                dataset_model_results = results_df[
                    (results_df['dataset_id'] == dataset) & 
                    (results_df['model'] == model)
                ]
                if not dataset_model_results.empty:
                    results_matrix[i, j] = dataset_model_results[column_name].iloc[0]
        
        # Calculate ranks
        ranks = np.zeros_like(results_matrix)
        for i in range(len(datasets)):
            ranks[i] = stats.rankdata(results_matrix[i])
        
        # Calculate mean ranks
        mean_ranks = np.mean(ranks, axis=0)
        
        # Perform Friedman test
        friedman_stat, p_value = stats.friedmanchisquare(*[ranks[:, i] for i in range(len(models))])
        
        # Calculate CD (Critical Difference)
        n_datasets = len(datasets)
        n_models = len(models)
        
        # CD formula for the Nemenyi test
        q_alpha = 2.343  # critical value for alpha=0.05
        cd = q_alpha * np.sqrt((n_models * (n_models + 1)) / (6 * n_datasets))
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        # Sort models by mean rank
        sorted_idx = np.argsort(mean_ranks)
        sorted_ranks = mean_ranks[sorted_idx]
        sorted_models = np.array(models)[sorted_idx]
        
        # Plot ranks
        y_pos = np.arange(len(models))
        plt.barh(y_pos, sorted_ranks, align='center')
        plt.yticks(y_pos, sorted_models)
        
        # Add CD line
        plt.axvline(x=min(sorted_ranks) + cd, color='r', linestyle='--', label=f'CD = {cd:.3f}')
        
        # Add horizontal lines for not significantly different groups
        current_group = 0
        for i in range(len(sorted_ranks)):
            if i == 0 or sorted_ranks[i] - sorted_ranks[i-1] > cd:
                current_group += 1
            plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
        
        plt.xlabel('Mean Rank')
        plt.title(f"Demšar's Test - {metric.upper()}\n"
                 f'Friedman p-value: {p_value:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'demsar_test_{metric}.png')
        plt.close()
        
        # Print results
        print(f"\nResults of Demšar's Test for {metric.upper()}:")
        print(f"Friedman statistic: {friedman_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Critical Difference (CD): {cd:.4f}")
        print("\nMean ranks:")
        for model, rank in zip(models, mean_ranks):
            print(f"{model}: {rank:.4f}")
        
        # Identify groups of not significantly different models
        print("\nGroups of not significantly different models:")
        groups = []
        current_group = []
        for i in range(len(sorted_ranks)):
            if i == 0 or sorted_ranks[i] - sorted_ranks[i-1] <= cd:
                current_group.append(sorted_models[i])
            else:
                groups.append(current_group)
                current_group = [sorted_models[i]]
        if current_group:
            groups.append(current_group)
        
        for i, group in enumerate(groups, 1):
            print(f"Group {i}: {', '.join(group)}")
        
        return {
            'friedman_stat': friedman_stat,
            'p_value': p_value,
            'cd': cd,
            'mean_ranks': dict(zip(models, mean_ranks)),
            'groups': groups
        }

    def run_experiments(self):
        """Runs experiments on the specified datasets"""
        dataset_ids = [23381, 1063, 6332, 40994, 1510, 1480, 11, 29, 15, 188, 1464, 37, 469, 458, 54, 50, 307, 31, 1494, 1468, 40966, 1068, 1462, 1049, 23, 1050, 1501, 40975, 40982]
        
        # Filter already processed datasets
        processed_ids = set(r['dataset_id'] for r in self.results)
        remaining_ids = [did for did in dataset_ids if did not in processed_ids]
        
        print(f"Already processed datasets: {len(processed_ids)}")
        print(f"Remaining datasets: {len(remaining_ids)}")
        
        for dataset_id in tqdm(remaining_ids, desc="Evaluating datasets"):
            try:
                print(f"\n{'='*50}")
                print(f"Processing dataset ID: {dataset_id}")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}")
                
                # Configure timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_per_dataset)
                
                # Load and preprocess data
                data = openml.datasets.get_dataset(dataset_id)
                X, y, _, _ = data.get_data(target=data.default_target_attribute)
                X_processed, y_processed = self.preprocess_data(X, y)
                
                # First split: separate test set (20%) that will never be used until the final evaluation
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
                )
                
                print(f"\nInitial data split:")
                print(f"Train+validation set (80%): {X_train_val.shape[0]} samples")
                print(f"Test set (20%): {X_test.shape[0]} samples")
                print("NOTE: The test set will ONLY be used for final evaluation")
                
                # Evaluate each model
                for model_name in self.models.keys():
                    try:
                        print(f"\n{'='*30}")
                        print(f"Model: {model_name.upper()}")
                        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"{'='*30}")
                        
                        # Optimize hyperparameters using ONLY the train+validation set
                        best_params = self.optimize_hyperparameters(X_train_val, y_train_val, model_name)
                        self.best_params[f"{dataset_id}_{model_name}"] = best_params
                        
                        # Train final model using the ENTIRE train+validation set
                        print("\nTraining final model with all train+validation data...")
                        if model_name == 'svm':
                            best_params['probability'] = True
                        model = self.models[model_name](**best_params, random_state=42)
                        
                        train_start = time()
                        if model_name == 'cat':
                            model.fit(X_train_val, y_train_val, verbose=False)
                        else:
                            model.fit(X_train_val, y_train_val)
                        train_time = time() - train_start
                        
                        # Evaluate on the train+validation set
                        train_val_start = time()
                        y_train_val_pred = model.predict(X_train_val)
                        if hasattr(model, 'predict_proba'):
                            y_train_val_proba = model.predict_proba(X_train_val)
                            # Calculate AUC OVO for multiple classes
                            n_classes = len(np.unique(y_train_val))
                            if n_classes > 2:
                                train_val_auc_ovo = roc_auc_score(y_train_val, y_train_val_proba, multi_class='ovo', average='weighted')
                            else:
                                train_val_auc_ovo = roc_auc_score(y_train_val, y_train_val_proba[:, 1])
                            train_val_ce = log_loss(y_train_val, y_train_val_proba)
                        else:
                            train_val_auc_ovo = np.nan
                            train_val_ce = np.nan
                        train_val_predict_time = time() - train_val_start
                        
                        # Evaluate on the test set (used for the first time)
                        test_start = time()
                        y_test_pred = model.predict(X_test)
                        if hasattr(model, 'predict_proba'):
                            y_test_proba = model.predict_proba(X_test)
                            # Calculate AUC OVO for multiple classes
                            if n_classes > 2:
                                test_auc_ovo = roc_auc_score(y_test, y_test_proba, multi_class='ovo', average='weighted')
                            else:
                                test_auc_ovo = roc_auc_score(y_test, y_test_proba[:, 1])
                            test_ce = log_loss(y_test, y_test_proba)
                        else:
                            test_auc_ovo = np.nan
                            test_ce = np.nan
                        test_predict_time = time() - test_start
                        
                        # Calculate metrics
                        train_val_acc = accuracy_score(y_train_val, y_train_val_pred)
                        test_acc = accuracy_score(y_test, y_test_pred)
                        total_time = train_time + train_val_predict_time + test_predict_time
                        
                        # Print results
                        print("\nResults on Train+Validation Set (80%):")
                        print(f"Accuracy: {train_val_acc:.4f}")
                        print(f"AUROC OVO: {train_val_auc_ovo:.4f}" if not np.isnan(train_val_auc_ovo) else "AUROC OVO: N/A")
                        print(f"Cross-Entropy: {train_val_ce:.4f}" if not np.isnan(train_val_ce) else "Cross-Entropy: N/A")
                        print(f"Time (train + predict): {train_time + train_val_predict_time:.2f}s")
                        
                        print("\nResults on Test Set (20%):")
                        print(f"Accuracy: {test_acc:.4f}")
                        print(f"AUROC OVO: {test_auc_ovo:.4f}" if not np.isnan(test_auc_ovo) else "AUROC OVO: N/A")
                        print(f"Cross-Entropy: {test_ce:.4f}" if not np.isnan(test_ce) else "Cross-Entropy: N/A")
                        print(f"Time (predict): {test_predict_time:.2f}s")
                        
                        print(f"\nTotal time (tune + train + predict): {total_time:.2f}s")
                        
                        # Store results
                        result = {
                            'dataset_id': dataset_id,
                            'dataset_name': data.name,
                            'model': model_name,
                            'instances': data.qualities['NumberOfInstances'],
                            'features': data.qualities['NumberOfFeatures'],
                            'processed_features': X_processed.shape[1],
                            'n_classes': n_classes,
                            'train_val_accuracy': train_val_acc,
                            'train_val_auc_ovo': train_val_auc_ovo,
                            'train_val_cross_entropy': train_val_ce,
                            'test_accuracy': test_acc,
                            'test_auc_ovo': test_auc_ovo,
                            'test_cross_entropy': test_ce,
                            'time': total_time
                        }
                        
                        # Add hyperparameters to the result
                        result.update(best_params)
                        
                        self.results.append(result)
                        
                        # Save partial results after each model
                        self.save_partial_results()
                        
                    except Exception as e:
                        print(f"\nError in model {model_name} for dataset {dataset_id}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                self.datasets_info.append({
                    'id': dataset_id,
                    'name': data.name,
                    'instances': data.qualities['NumberOfInstances'],
                    'features': data.qualities['NumberOfFeatures']
                })
                
                # Disable timeout
                signal.alarm(0)
                
            except TimeoutError:
                print(f"\nTimeout on dataset {dataset_id} after {self.timeout_per_dataset} seconds")
                signal.alarm(0)
                continue
            except Exception as e:
                print(f"\nError on dataset {dataset_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                signal.alarm(0)
                continue
                
        return pd.DataFrame(self.results)
    
    def get_results(self):
        """Returns results as a DataFrame"""
        return pd.DataFrame(self.results)

# Running the experiment
if __name__ == "__main__":
    # Configure to use GPU if available
    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {'Yes' if use_gpu else 'No'}")
    
    askllm = ASKLLM2(use_gpu=use_gpu, n_trials=30, timeout_per_dataset=3600)
    results_df = askllm.run_experiments()
    
    # Save results
    results_df.to_csv('askllm2_results.csv', index=False)
    
    # Perform Demšar's test for each metric
    metrics = ['accuracy', 'auc_ovo', 'cross_entropy', 'time']
    demsar_results = {}
    
    for metric in metrics:
        print(f"\n{'='*50}")
        print(f"Demšar's Test for {metric.upper()}")
        print(f"{'='*50}")
        demsar_results[metric] = askllm.perform_demsar_test(results_df, metric)
    
    # Save Demšar's test results
    with open('demsar_results.json', 'w') as f:
        json.dump(demsar_results, f, indent=4)
    
    # Display summary of results by model
    print("\nSummary of Results by Model:")
    for model in results_df['model'].unique():
        print(f"\n{model.upper()}:")
        model_results = results_df[results_df['model'] == model]
        print(model_results[['dataset_name', 'accuracy', 'auc_ovo', 'cross_entropy', 'time']])
        
        print("\nMeans:")
        print(model_results[['accuracy', 'auc_ovo', 'cross_entropy', 'time']].mean())
        
        print("\nCross-Validation Means:")
        cv_metrics = ['cv_accuracy_mean', 'cv_auc_ovo_mean', 'cv_ce_mean', 'cv_time_mean']
        cv_stds = ['cv_accuracy_std', 'cv_auc_ovo_std', 'cv_ce_std', 'cv_time_std']
        
        for metric, std in zip(cv_metrics, cv_stds):
            mean = model_results[metric].mean()
            std_mean = model_results[std].mean()
            print(f"{metric}: {mean:.4f} ± {std_mean:.4f}")
