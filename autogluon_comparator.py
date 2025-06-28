import optuna
import torch
import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report, log_loss, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from time import time
import os
import warnings
from tabulate import tabulate
import json
from autogluon.tabular import TabularPredictor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import StratifiedKFold

# Global configurations
TEST_SIZE = 0.3  # Changed to 30% for testing
RANDOM_SEED = 42
N_TRIALS = 30  # Number of trials for hyperparameter optimization
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=UserWarning)

# Configure AutoGluon to avoid GUI issues
os.environ['AUTOGLUON_DISABLE_GUI'] = '1'
os.environ['AUTOGLUON_DISABLE_MPL'] = '1'

# Dictionaries to store results
results_comparison = {}

def load_and_preprocess_data(dataset_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[str]]:
    """Loads and preprocesses data from OpenML"""
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    print(f"\nOriginal dataset:")
    print(f"Shape: {X.shape}")
    print(f"Classes: {np.unique(y, return_counts=True)}")
    
    # Check for constant features
    constant_features = [col for col in X.columns if X[col].nunique() == 1]
    if constant_features:
        print(f"\n⚠️ Constant features found: {constant_features}")
        X = X.drop(columns=constant_features)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(exclude=['number']).columns
    
    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Complete preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after preprocessing
    feature_names = []
    # Add names of numeric features
    feature_names.extend(numeric_features)
    # Add names of categorical features after one-hot encoding
    for cat_feature in categorical_features:
        categories = preprocessor.named_transformers_['cat'].named_steps['encoder'].categories_[list(categorical_features).index(cat_feature)]
        feature_names.extend([f"{cat_feature}_{cat}" for cat in categories])
    
    # Encode classes
    label_encoder = LabelEncoder()
    y_processed = label_encoder.fit_transform(y)
    
    # Initial split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed,
        y_processed,
        test_size=TEST_SIZE,
        stratify=y_processed,
        random_state=RANDOM_SEED
    )
    
    print(f"\n✅ Dataset {dataset_id} processed and split:")
    print(f"Train+Validation: {X_train_val.shape}, Test: {X_test.shape}")
    print(f"Class distribution (train+val): {np.bincount(y_train_val)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")
    print(f"Number of features: {len(feature_names)}")
    
    # Check for features with zero variance after preprocessing
    zero_var_features = []
    for i, feature in enumerate(feature_names):
        if np.var(X_train_val[:, i]) == 0:
            zero_var_features.append(feature)
    
    if zero_var_features:
        print(f"\n⚠️ Zero variance features found: {zero_var_features}")
    
    return X_train_val, y_train_val, X_test, y_test, len(label_encoder.classes_), feature_names

def optimize_autogluon_hyperparameters(X_train_val: np.ndarray, y_train_val: np.ndarray, num_classes: int, feature_names: List[str]) -> Dict[str, Any]:
    """Optimizes AutoGluon hyperparameters using Optuna with cross-validation"""
    print("\n🔍 Optimizing AutoGluon hyperparameters...")
    
    # Create DataFrame for AutoGluon with feature names
    train_data = pd.DataFrame(X_train_val, columns=feature_names)
    train_data['target'] = y_train_val
    
    # Configure cross-validation
    n_splits = 10  # Number of folds for cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    def objective(trial):
        # Define search space for general parameters
        time_limit = trial.suggest_int('time_limit', 60, 300)
        num_bag_folds = trial.suggest_int('num_bag_folds', 3, 7)
        num_stack_levels = trial.suggest_int('num_stack_levels', 0, 2)
        
        # Define search space for specific model hyperparameters
        gbm_params = {
            'num_boost_round': trial.suggest_int('gbm_num_boost_round', 100, 1000),
            'learning_rate': trial.suggest_float('gbm_learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('gbm_num_leaves', 20, 100),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        }
        
        cat_params = {
            'iterations': trial.suggest_int('cat_iterations', 100, 1000),
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.001, 0.1, log=True),
            'depth': trial.suggest_int('cat_depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        }
        
        nn_params = {
            'num_epochs': trial.suggest_int('nn_epochs', 10, 50),
            'learning_rate': trial.suggest_float('nn_learning_rate', 0.0001, 0.01, log=True),
            'dropout_prob': trial.suggest_float('nn_dropout', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        }
        
        # Configure model hyperparameters
        model_hyperparameters = {
            'GBM': [gbm_params],
            'CAT': [cat_params],
            'NN_TORCH': [nn_params]
        }
        
        # List to store scores from each fold
        fold_scores = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data, train_data['target']), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data for this fold
            fold_train = train_data.iloc[train_idx]
            fold_val = train_data.iloc[val_idx]
            
            # Configure and train the model
            predictor = TabularPredictor(
                label='target',
                problem_type='multiclass' if num_classes > 2 else 'binary',
                eval_metric='f1_macro'
            )
            
            # Train model with the current parameters
            predictor.fit(
                fold_train,
                tuning_data=fold_val,
                use_bag_holdout=True,
                hyperparameters=model_hyperparameters,
                time_limit=time_limit//n_splits,  # Divide time among folds
                num_bag_folds=num_bag_folds,
                num_stack_levels=num_stack_levels,
                verbosity=2
            )
            
            # Evaluate on the validation set
            val_score = predictor.evaluate(fold_val)['f1_macro']
            fold_scores.append(val_score)
            
            print(f"Fold {fold} Score: {val_score:.4f}")
        
        # Calculate mean of scores
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"\nMean of scores: {mean_score:.4f} ± {std_score:.4f}")
        
        # Return all parameters for later use
        return mean_score, {
            'time_limit': time_limit,
            'num_bag_folds': num_bag_folds,
            'num_stack_levels': num_stack_levels,
            'hyperparameters': model_hyperparameters,
            'cv_scores': fold_scores,
            'cv_mean': mean_score,
            'cv_std': std_score
        }
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    
    def objective_wrapper(trial):
        score, params = objective(trial)
        trial.set_user_attr('params', params)
        return score
    
    study.optimize(objective_wrapper, n_trials=N_TRIALS)
    
    # Get parameters from the best trial
    best_params = study.best_trial.user_attrs['params']
    
    print(f"\n✅ Best AutoGluon hyperparameters:")
    print(f"F1-Score: {study.best_value:.4f}")
    print("Parameters:", best_params)
    
    return best_params

def calculate_metrics(y_true, y_pred, y_pred_proba, num_classes):
    """Calculates all necessary metrics for evaluation"""
    metrics = {}
    
    # Convert to numpy arrays if necessary
    if isinstance(y_pred_proba, pd.DataFrame):
        y_pred_proba = y_pred_proba.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    # Mean Accuracy
    metrics['accuracy'] = (y_pred == y_true).mean()
    
    # Mean Cross-Entropy
    if num_classes > 2:
        # For multiclass, we need to ensure y_pred_proba has the correct shape
        if y_pred_proba.shape[1] != num_classes:
            y_pred_proba = np.eye(num_classes)[y_pred_proba]
    metrics['cross_entropy'] = log_loss(y_true, y_pred_proba)
    
    # Mean AUC OVO (One-vs-One)
    if num_classes > 2:
        # For multiclass, we calculate AUC OVO
        auc_scores = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                # Filter only samples of classes i and j
                mask = np.isin(y_true, [i, j])
                if np.sum(mask) > 0:
                    y_true_ij = y_true[mask]
                    y_pred_proba_ij = y_pred_proba[mask]
                    # Calculate AUC for the class pair
                    try:
                        auc = roc_auc_score(y_true_ij, y_pred_proba_ij[:, i])
                        auc_scores.append(auc)
                    except:
                        continue
        metrics['auc_ovo'] = np.mean(auc_scores) if auc_scores else 0.0
    else:
        # For binary, AUC is straightforward
        try:
            # If y_pred_proba has only one column, assume it's the probability of the positive class
            if y_pred_proba.shape[1] == 1:
                metrics['auc_ovo'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['auc_ovo'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            print(f"⚠️ Error calculating AUC: {str(e)}")
            metrics['auc_ovo'] = 0.0
    
    return metrics

def train_autogluon_with_params(X_train_val: np.ndarray, y_train_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                               num_classes: int, params: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
    """Trains and evaluates the AutoGluon model with the best hyperparameters"""
    print("\n🚀 Training AutoGluon with best hyperparameters...")
    start_time = time()
    
    try:
        # Create DataFrames for AutoGluon with feature names
        train_data = pd.DataFrame(X_train_val, columns=feature_names)
        train_data['target'] = y_train_val
        
        test_data = pd.DataFrame(X_test, columns=feature_names)
        test_data['target'] = y_test
        
        # Check for zero variance features
        zero_var_features = train_data.columns[train_data.var() == 0].tolist()
        if zero_var_features:
            print(f"\n⚠️ Removing zero variance features: {zero_var_features}")
            train_data = train_data.drop(columns=zero_var_features)
            test_data = test_data.drop(columns=zero_var_features)
            feature_names = [f for f in feature_names if f not in zero_var_features]
        
        # Configure and train the model
        predictor = TabularPredictor(
            label='target',
            problem_type='multiclass' if num_classes > 2 else 'binary',
            eval_metric='f1_macro'
        )
        
        # Extract parameters
        time_limit = params['time_limit']
        num_bag_folds = params['num_bag_folds']
        num_stack_levels = params['num_stack_levels']
        model_hyperparameters = params['hyperparameters']
        
        # Train final model with the ENTIRE training set
        predictor.fit(
            train_data,
            hyperparameters=model_hyperparameters,
            time_limit=time_limit,
            num_bag_folds=num_bag_folds,
            num_stack_levels=num_stack_levels,
            verbosity=2
        )
        
        # Evaluate on the test set
        test_score = predictor.evaluate(test_data)
        
        # Get predictions and probabilities
        y_pred = predictor.predict(test_data)
        y_pred_proba = predictor.predict_proba(test_data)
        
        # Calculate all required metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba, num_classes)
        
        # Calculate total time (tune + train + predict)
        total_time = time() - start_time
        
        # Print detailed metrics
        print("\nEvaluation metrics:")
        print(f"Mean Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean AUC OVO: {metrics['auc_ovo']:.4f}")
        print(f"Mean Cross-Entropy: {metrics['cross_entropy']:.4f}")
        print(f"Total time: {total_time:.2f}s")
        
        if num_classes > 2:
            print("\nConfusion matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            
            print("\nClassification report:")
            print(classification_report(y_test, y_pred))
        
        # Get feature importance
        feature_importance = predictor.feature_importance(test_data)
        if feature_importance is not None:
            importance_dict = {}
            if 'importance' in feature_importance:
                for feature, importance in feature_importance['importance'].items():
                    try:
                        importance_dict[feature] = float(importance)
                    except (ValueError, TypeError):
                        print(f"⚠️ Could not convert importance of feature {feature}: {importance}")
                        continue
        
        return {
            'accuracy': metrics['accuracy'],
            'auc_ovo': metrics['auc_ovo'],
            'cross_entropy': metrics['cross_entropy'],
            'time': total_time,
            'feature_importance': importance_dict if feature_importance is not None else None
        }
    
    except Exception as e:
        print(f"❌ Error in AutoGluon: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(dataset_id: int, results: dict):
    """Plot of AutoGluon results"""
    if results is None:
        print("No valid results to plot")
        return
    
    # Create directory to save plots
    os.makedirs("autogluon_plots", exist_ok=True)
    
    # Plot metrics
    metrics = ['accuracy', 'auc_ovo', 'cross_entropy']
    values = [results[metric] for metric in metrics]
    labels = ['Accuracy', 'AUC OVO', 'Cross-Entropy']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values)
    plt.title(f'AutoGluon Metrics - Dataset {dataset_id}', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'autogluon_plots/metrics_{dataset_id}.png', dpi=300)
    plt.close()
    
    # Plot time
    plt.figure(figsize=(8, 6))
    plt.bar(['Total Time'], [results['time']])
    plt.title(f'Execution Time - Dataset {dataset_id}', fontsize=14)
    plt.ylabel('Time (s)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.text(0, results['time'], f'{results["time"]:.2f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'autogluon_plots/time_{dataset_id}.png', dpi=300)
    plt.close()
    
    # Plot feature importance
    if 'feature_importance' in results and results['feature_importance'] is not None:
        feature_importance = pd.Series(results['feature_importance'])
        plt.figure(figsize=(12, 6))
        feature_importance.sort_values(ascending=True).plot(kind='barh')
        plt.title(f'Feature Importance - Dataset {dataset_id}', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'autogluon_plots/feature_importance_{dataset_id}.png', dpi=300)
        plt.close()

def generate_summary_table(results: dict):
    """Generates a summary table of the results"""
    summary_data = []
    
    for dataset_id, metrics in results.items():
        if metrics is not None:
            summary_data.append({
                'Dataset': dataset_id,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'AUC OVO': f"{metrics['auc_ovo']:.4f}",
                'Cross-Entropy': f"{metrics['cross_entropy']:.4f}",
                'Time (s)': f"{metrics['time']:.2f}"
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    df.to_csv('autogluon_results_summary.csv', index=False)
    
    # Return formatted table
    return tabulate(df, headers='keys', tablefmt='grid', showindex=False)

def save_detailed_results(results: dict):
    """Saves detailed results to a file"""
    try:
        with open('autogluon_detailed_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        return True
    except Exception as e:
        print(f"\n⚠️ Error saving results: {str(e)}")
        return False

def run_autogluon_analysis(dataset_id: int):
    """Runs a complete analysis of AutoGluon on a dataset"""
    print(f"\n{'='*50}")
    print(f"Analyzing dataset {dataset_id} with AutoGluon")
    
    try:
        # Load and preprocess data
        X_train_val, y_train_val, X_test, y_test, num_classes, feature_names = load_and_preprocess_data(dataset_id)
        
        # Optimize hyperparameters
        best_params = optimize_autogluon_hyperparameters(X_train_val, y_train_val, num_classes, feature_names)
        
        # Train model with best parameters
        results = train_autogluon_with_params(X_train_val, y_train_val, X_test, y_test, num_classes, best_params, feature_names)
        
        if results is not None:
            results_comparison[dataset_id] = {
                'accuracy': float(results['accuracy']),
                'auc_ovo': float(results['auc_ovo']),
                'cross_entropy': float(results['cross_entropy']),
                'time': float(results['time']),
                'feature_importance': results['feature_importance'] if 'feature_importance' in results else None
            }
            print(f"✅ Dataset {dataset_id} - Accuracy: {results['accuracy']:.4f}, AUC OVO: {results['auc_ovo']:.4f}, Cross-Entropy: {results['cross_entropy']:.4f}, Time: {results['time']:.2f}s")
        else:
            results_comparison[dataset_id] = None
            print(f"❌ Dataset {dataset_id} returned None")
        
        # Plot results
        plot_results(dataset_id, results_comparison[dataset_id])
        
        # Save results
        save_detailed_results(results_comparison)
        
        return results_comparison[dataset_id]
    
    except Exception as e:
        print(f"❌ Error processing dataset {dataset_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # List of datasets to analyze
    dataset_ids = [23381, 1063, 6332, 40994, 1510, 1480, 11, 29, 15, 188, 1464, 37, 469, 458, 54, 50, 307, 31, 1494, 1468, 40966, 1068, 1462, 1049, 23, 1050, 1501, 40975, 40982]
    
    # Analyze each dataset
    for dataset_id in dataset_ids:
        run_autogluon_analysis(dataset_id)
    
    # Final summary
    print("\n🎉 Analysis complete! Summary:")
    
    # Generate summary table
    summary_table = generate_summary_table(results_comparison)
    print(summary_table)
    
    print("\n📊 Plots saved in the 'autogluon_plots' folder")
    print("📝 Detailed results saved in 'autogluon_detailed_results.json'")
    print("📋 Final summary saved in 'autogluon_results_summary.csv'")
