import optuna
import torch
import numpy as np
import openml
import lightning.pytorch as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score, log_loss
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from typing import List, Tuple, Dict, Any
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from time import time
from sklearn.metrics.pairwise import pairwise_distances
from tabulate import tabulate
import json
from lit_saint import Saint, SaintDatamodule, SaintTrainer
from omegaconf import OmegaConf
from saint_config_patch import SaintConfig
import torch.nn.functional as F
from scipy import stats
from scipy.stats import friedmanchisquare, norm
from scipy.stats import rankdata

# Global configurations
NUM_FOLDS = 10
TEST_SIZE = 0.3
RANDOM_SEED = 42
N_TRIALS = 30 # Number of trials for hyperparameter optimization
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=UserWarning)

# Dictionaries to store results
best_params = {}
results_comparison = {}

class LitSAINT(pl.LightningModule):
    """PyTorch Lightning wrapper for the SAINT model"""
    
    def __init__(self, input_dim: int, num_classes: int, dim: int = 32,
                 depth: int = 1, heads: int = 4, dropout: float = 0.4, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        if dim % heads != 0:
            raise ValueError(f"embed_dim ({dim}) must be divisible by num_heads ({heads})")
        
        self.input_proj = torch.nn.Linear(input_dim, dim)
        
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dropout=dropout,
                batch_first=True,
                dim_feedforward=dim*2
            ),
            num_layers=depth
        )
        
        self.classifier = torch.nn.Linear(dim, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        return self.classifier(x[:, 0, :])

    def _calculate_metrics(self, y_hat, y):
        """Calculates common metrics for training, validation, and testing"""
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        y_pred = y_hat.argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='macro')
        return loss, acc, f1

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, acc, f1 = self._calculate_metrics(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, acc, f1 = self._calculate_metrics(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, acc, f1 = self._calculate_metrics(y_hat, y)
        
        # Log metrics
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'test_loss': loss, 'test_acc': acc, 'test_f1': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_f1'
            }
        }

def load_and_preprocess_data(dataset_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Loads and preprocesses data from OpenML, splitting 80% for training and 20% for testing"""
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(exclude=['number']).columns
    
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
    
    # Encode classes
    label_encoder = LabelEncoder()
    y_processed = label_encoder.fit_transform(y)
    
    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y_processed,
        test_size=0.2,  # 20% for testing
        stratify=y_processed,
        random_state=RANDOM_SEED
    )
    
    print(f"\n✅ Dataset {dataset_id} processed and split:")
    print(f"Training (80%): {X_train.shape}")
    print(f"Test (20%): {X_test.shape}")
    print(f"Class distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")
    print("NOTE: The test set will ONLY be used for final evaluation")
    
    return X_train, y_train, X_test, y_test, len(label_encoder.classes_)

def optimize_saint_hyperparameters(X_train_val: np.ndarray, y_train_val: np.ndarray, num_classes: int) -> Dict[str, Any]:
    """Optimizes SAINT hyperparameters using Optuna"""
    print("\n🔍 Optimizing SAINT hyperparameters...")
    
    def objective(trial):
        # First choose the number of heads
        heads = trial.suggest_int('heads', 2, 8)
        
        # Then choose the dimension that is a multiple of the number of heads
        dim = trial.suggest_int('dim', 32, 128)
        dim = (dim // heads) * heads  # Ensures dim is divisible by heads
        
        depth = trial.suggest_int('depth', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        print(f"\nTrial with parameters:")
        print(f"dim: {dim}, depth: {depth}, heads: {heads}, dropout: {dropout:.3f}, lr: {lr:.6f}")
        
        # Configure K-Fold
        kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val, y_train_val), 1):
            print(f"\nFold {fold}/{NUM_FOLDS}")
            
            X_train_fold = torch.tensor(X_train_val[train_idx], dtype=torch.float32)
            y_train_fold = torch.tensor(y_train_val[train_idx], dtype=torch.long)
            X_val_fold = torch.tensor(X_train_val[val_idx], dtype=torch.float32)
            y_val_fold = torch.tensor(y_train_val[val_idx], dtype=torch.long)
            
            # DataLoaders
            train_loader = DataLoader(
                TensorDataset(X_train_fold, y_train_fold),
                batch_size=32,
                shuffle=True
            )
            
            # Model
            model = LitSAINT(
                input_dim=X_train_val.shape[1],
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                dropout=dropout,
                lr=lr
            )
            
            # Trainer
            trainer = pl.Trainer(
                max_epochs=50,
                accelerator="auto",
                callbacks=[
                    EarlyStopping(monitor="train_f1", patience=5, mode="max")
                ],
                enable_progress_bar=True
            )
            
            # Training
            print(f"Training fold {fold}...")
            trainer.fit(model, train_loader)
            
            # Evaluation
            print(f"Evaluating fold {fold}...")
            model.eval()
            with torch.no_grad():
                y_hat = model(X_train_fold)
                y_pred = y_hat.argmax(dim=1).cpu().numpy()
                y_true = y_train_fold.cpu().numpy()
                fold_score = f1_score(y_true, y_pred, average='macro')
            
            fold_scores.append(fold_score)
            print(f"Fold {fold} F1-Score: {fold_score:.4f}")
            
            # If perfect accuracy is reached, return immediately
            if fold_score >= 0.9999:
                print("🎯 Perfect accuracy reached! Stopping optimization...")
                return 1.0, {'dim': dim, 'depth': depth, 'heads': heads, 'dropout': dropout, 'lr': lr}
        
        mean_score = np.mean(fold_scores)
        print(f"\nMean F1-Scores: {mean_score:.4f}")
        
        # Return parameters with the adjusted dimension
        return mean_score, {'dim': dim, 'depth': depth, 'heads': heads, 'dropout': dropout, 'lr': lr}
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    
    def objective_wrapper(trial):
        score, params = objective(trial)
        # Store the adjusted parameters in the trial
        trial.set_user_attr('adjusted_params', params)
        return score
    
    study.optimize(objective_wrapper, n_trials=N_TRIALS)
    
    # Get the adjusted parameters from the best trial
    best_params = study.best_trial.user_attrs['adjusted_params']
    
    print(f"\n✅ Best SAINT hyperparameters:")
    print(f"F1-Score: {study.best_value:.4f}")
    print("Parameters:", best_params)
    
    return best_params

def optimize_lightgbm_hyperparameters(X_train_val: np.ndarray, y_train_val: np.ndarray, num_classes: int) -> Dict[str, Any]:
    """Optimizes LightGBM hyperparameters using Optuna"""
    print("\n🔍 Optimizing LightGBM hyperparameters...")
    
    def objective(trial):
        # Define search space
        params = {
            'objective': 'multiclass' if num_classes > 2 else 'binary',
            'num_class': num_classes if num_classes > 2 else None,
            'metric': 'multi_logloss' if num_classes > 2 else 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbose': -1,
            'seed': RANDOM_SEED,
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True)
        }
        
        # Configure K-Fold
        kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val, y_train_val), 1):
            print(f"\nFold {fold}/{NUM_FOLDS}")
            
            X_train_fold = X_train_val[train_idx]
            y_train_fold = y_train_val[train_idx]
            X_val_fold = X_train_val[val_idx]
            y_val_fold = y_train_val[val_idx]
            
            # LightGBM Dataset
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold)
            
            # Training
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Prediction
            y_pred_proba = model.predict(X_val_fold)
            
            # Adjust prediction based on problem type
            if num_classes > 2:
                y_pred = y_pred_proba.argmax(axis=1)
            else:
                y_pred = (y_pred_proba > 0.5).astype(int)
            
            fold_score = f1_score(y_val_fold, y_pred, average='macro')
            fold_scores.append(fold_score)
            print(f"Fold {fold} F1-Score: {fold_score:.4f}")
            
            # If perfect accuracy is reached, return immediately
            if fold_score >= 0.9999:
                print("🎯 Perfect accuracy reached! Stopping optimization...")
                return 1.0
        
        mean_score = np.mean(fold_scores)
        print(f"\nMean F1-Scores: {mean_score:.4f}")
        return mean_score
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print(f"\n✅ Best LightGBM hyperparameters:")
    print(f"F1-Score: {study.best_value:.4f}")
    print("Parameters:", study.best_params)
    
    return study.best_params

def optimize_xgboost_hyperparameters(X_train_val: np.ndarray, y_train_val: np.ndarray, num_classes: int) -> Dict[str, Any]:
    """Optimizes XGBoost hyperparameters using Optuna"""
    print("\n🔍 Optimizing XGBoost hyperparameters...")
    
    def objective(trial):
        # Define search space
        params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'seed': RANDOM_SEED,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster and uses less memory
            'n_estimators': 1000    # Fixed number of trees
        }
        
        # Configure K-Fold
        kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val, y_train_val), 1):
            print(f"\nFold {fold}/{NUM_FOLDS}")
            
            X_train_fold = X_train_val[train_idx]
            y_train_fold = y_train_val[train_idx]
            X_val_fold = X_train_val[val_idx]
            y_val_fold = y_train_val[val_idx]
            
            # Training
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            # Evaluation
            y_pred = model.predict(X_val_fold)
            fold_score = f1_score(y_val_fold, y_pred, average='macro')
            fold_scores.append(fold_score)
            print(f"Fold {fold} F1-Score: {fold_score:.4f}")
            
            # If perfect accuracy is reached, return immediately
            if fold_score >= 0.9999:
                print("🎯 Perfect accuracy reached! Stopping optimization...")
                return 1.0
        
        mean_score = np.mean(fold_scores)
        print(f"\nMean F1-Scores: {mean_score:.4f}")
        return mean_score
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print(f"\n✅ Best XGBoost hyperparameters:")
    print(f"F1-Score: {study.best_value:.4f}")
    print("Parameters:", study.best_params)
    
    return study.best_params

def optimize_catboost_hyperparameters(X_train_val: np.ndarray, y_train_val: np.ndarray, num_classes: int) -> Dict[str, Any]:
    """Optimizes CatBoost hyperparameters using Optuna"""
    print("\n🔍 Optimizing CatBoost hyperparameters...")
    
    def objective(trial):
        # Define search space with more restrictive parameters
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
        grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Lossguide'])
        
        # Define loss function based on the number of classes
        if num_classes > 2:
            loss_function = 'MultiClass'
            boost_from_average = False  # Do not use with MultiClass
        else:
            loss_function = 'Logloss'
            boost_from_average = True   # Use with Logloss
        
        # Base parameters with much stronger regularization
        params = {
            'iterations': 500,  # Reduced
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),  # Very reduced
            'depth': trial.suggest_int('depth', 2, 4),  # Very reduced
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),  # Greatly increased
            'bootstrap_type': bootstrap_type,
            'random_strength': trial.suggest_float('random_strength', 1.0, 20.0, log=True),  # Greatly increased
            'loss_function': loss_function,
            'random_seed': RANDOM_SEED,
            'verbose': False,
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),  # Greatly increased
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 5),  # Reduced
            'grow_policy': grow_policy,
            'feature_border_type': 'UniformAndQuantiles',
            'boosting_type': 'Plain',
            'max_bin': trial.suggest_int('max_bin', 10, 50),  # New
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1) if bootstrap_type == 'Bayesian' else None,
            'subsample': trial.suggest_float('subsample', 0.6, 0.8) if bootstrap_type == 'Bernoulli' else None,
            'rsm': trial.suggest_float('rsm', 0.1, 0.5),  # New
            'score_function': 'Cosine',  # More conservative
            'leaf_estimation_method': 'Newton',  # More conservative
            'boost_from_average': boost_from_average,  # Adjusted based on loss function
            'use_best_model': True,
            'od_type': 'Iter',
            'od_wait': 50
        }
        
        # Add max_leaves only if grow_policy is Lossguide
        if grow_policy == 'Lossguide':
            params['max_leaves'] = trial.suggest_int('max_leaves', 10, 50)
        
        # Remove None parameters
        params = {k: v for k, v in params.items() if v is not None}
        
        # Configure K-Fold with the same number of folds as other models
        kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []
        fold_gaps = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val, y_train_val), 1):
            print(f"\nFold {fold}/{NUM_FOLDS}")
            
            X_train_fold = X_train_val[train_idx]
            y_train_fold = y_train_val[train_idx]
            X_val_fold = X_train_val[val_idx]
            y_val_fold = y_train_val[val_idx]
            
            # Training
            model = CatBoostClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Evaluation
            y_pred = model.predict(X_val_fold).flatten()
            fold_score = f1_score(y_val_fold, y_pred, average='macro')
            fold_scores.append(fold_score)
            
            # Check for overfitting
            train_pred = model.predict(X_train_fold).flatten()
            train_score = f1_score(y_train_fold, train_pred, average='macro')
            fold_gap = train_score - fold_score
            fold_gaps.append(fold_gap)
            
            print(f"Fold {fold} F1-Score: {fold_score:.4f}")
            print(f"Fold {fold} Train F1-Score: {train_score:.4f}")
            print(f"Fold {fold} Overfitting gap: {fold_gap:.4f}")
            
            # If perfect accuracy is reached, return immediately
            if fold_score >= 0.9999:
                print("🎯 Perfect accuracy reached! Stopping optimization...")
                return 1.0
        
        mean_score = np.mean(fold_scores)
        mean_gap = np.mean(fold_gaps)
        print(f"\nMean F1-Scores: {mean_score:.4f}")
        print(f"Mean overfitting gap: {mean_gap:.4f}")
        
        # More aggressive penalty
        if mean_score > 0.95:
            mean_score *= 0.7  # Penalize by 30%
            print("⚠️ Penalizing score due to possible overfitting")
        elif mean_gap > 0.1:  # If there is a significant gap
            mean_score *= 0.8  # Penalize by 20%
            print("⚠️ Penalizing score due to overfitting gap")
        
        return mean_score
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print(f"\n✅ Best CatBoost hyperparameters:")
    print(f"F1-Score: {study.best_value:.4f}")
    print("Parameters:", study.best_params)
    
    return study.best_params

def calculate_metrics(y_true, y_pred, y_pred_proba, num_classes):
    """Calculates all necessary metrics for evaluation"""
    metrics = {}
    
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
        metrics['auc_ovo'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    return metrics

def evaluate_model(model, X_train, y_train, X_test, y_test, num_classes, model_name):
    """Evaluates the model on both the training and test sets"""
    results = {}
    
    # Evaluation on the training set
    print(f"\nEvaluating {model_name} on the training set...")
    if model_name == 'SAINT':
        model.eval()
        with torch.no_grad():
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            train_loader = DataLoader(TensorDataset(X_train_t), batch_size=32)
            y_pred_proba_train = []
            y_pred_train = []
            for batch in train_loader:
                x, = batch
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                y_pred_proba_train.append(probs.cpu().numpy())
                y_pred_train.append(probs.argmax(dim=1).cpu().numpy())
            y_pred_proba_train = np.concatenate(y_pred_proba_train)
            y_pred_train = np.concatenate(y_pred_train)
    elif model_name == 'LightGBM':
        y_pred_proba_train = model.predict(X_train)
        if num_classes > 2:
            y_pred_train = y_pred_proba_train.argmax(axis=1)
        else:
            y_pred_train = (y_pred_proba_train > 0.5).astype(int)
            y_pred_proba_train = np.column_stack((1 - y_pred_proba_train, y_pred_proba_train))
    else:
        y_pred_proba_train = model.predict_proba(X_train)
        y_pred_train = model.predict(X_train)
    
    train_metrics = calculate_metrics(y_train, y_pred_train, y_pred_proba_train, num_classes)
    results['train'] = train_metrics
    
    # Evaluation on the test set
    print(f"\nEvaluating {model_name} on the test set...")
    if model_name == 'SAINT':
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            test_loader = DataLoader(TensorDataset(X_test_t), batch_size=32)
            y_pred_proba_test = []
            y_pred_test = []
            for batch in test_loader:
                x, = batch
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                y_pred_proba_test.append(probs.cpu().numpy())
                y_pred_test.append(probs.argmax(dim=1).cpu().numpy())
            y_pred_proba_test = np.concatenate(y_pred_proba_test)
            y_pred_test = np.concatenate(y_pred_test)
    elif model_name == 'LightGBM':
        y_pred_proba_test = model.predict(X_test)
        if num_classes > 2:
            y_pred_test = y_pred_proba_test.argmax(axis=1)
        else:
            y_pred_test = (y_pred_proba_test > 0.5).astype(int)
            y_pred_proba_test = np.column_stack((1 - y_pred_proba_test, y_pred_proba_test))
    else:
        y_pred_proba_test = model.predict_proba(X_test)
        y_pred_test = model.predict(X_test)
    
    test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test, num_classes)
    results['test'] = test_metrics
    
    return results

def train_saint_with_params(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                          num_classes: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trains and evaluates the SAINT model with the best hyperparameters"""
    print("\n🏗️  Training SAINT model with best hyperparameters...")
    start_time = time()
    
    try:
        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        
        # DataLoader for training
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=32,
            shuffle=True
        )
        
        # Model with best hyperparameters
        print("\nCreating model with parameters:", params)
        model = LitSAINT(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            **params
        )
        
        # Trainer with callbacks to monitor all metrics
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator="auto",
            callbacks=[
                EarlyStopping(
                    monitor="train_f1",
                    patience=5,
                    mode="max",
                    verbose=True
                ),
                ModelCheckpoint(
                    monitor="train_f1",
                    mode="max",
                    save_top_k=1,
                    verbose=True
                )
            ],
            enable_progress_bar=True,
            log_every_n_steps=1
        )
        
        # Training
        print("\nStarting training...")
        trainer.fit(model, train_loader)
        
        # Evaluation on train and test
        results = evaluate_model(model, X_train, y_train, X_test, y_test, num_classes, 'SAINT')
        results['train']['time'] = time() - start_time
        results['test']['time'] = time() - start_time
        
        return results
    
    except Exception as e:
        print(f"❌ Error in SAINT: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_lightgbm_with_params(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                              num_classes: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trains and evaluates the LightGBM model with the best hyperparameters"""
    print("\n💡 Training LightGBM with best hyperparameters...")
    start_time = time()
    
    try:
        # Split the training set into training and validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train,
            test_size=0.1,  # 10% for validation
            stratify=y_train,
            random_state=RANDOM_SEED
        )
        
        # LightGBM Datasets
        train_data = lgb.Dataset(X_train_final, label=y_train_final)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        # Adjust parameters for the problem type
        if num_classes > 2:
            params['objective'] = 'multiclass'
            params['num_class'] = num_classes
            params['metric'] = 'multi_logloss'
        else:
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
        
        # Training
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Evaluation on train and test
        results = evaluate_model(model, X_train, y_train, X_test, y_test, num_classes, 'LightGBM')
        results['train']['time'] = time() - start_time
        results['test']['time'] = time() - start_time
        
        return results
    
    except Exception as e:
        print(f"❌ Error in LightGBM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_xgboost_with_params(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                             num_classes: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trains and evaluates the XGBoost model with the best hyperparameters"""
    print("\n⚡ Training XGBoost with best hyperparameters...")
    start_time = time()
    
    try:
        # Add a fixed number of estimators to the parameters
        params['n_estimators'] = 1000
        
        # Training
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluation on train and test
        results = evaluate_model(model, X_train, y_train, X_test, y_test, num_classes, 'XGBoost')
        results['train']['time'] = time() - start_time
        results['test']['time'] = time() - start_time
        
        return results
    
    except Exception as e:
        print(f"❌ Error in XGBoost: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_catboost_with_params(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                              num_classes: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trains and evaluates the CatBoost model with the best hyperparameters"""
    print("\n🐱 Training CatBoost with best hyperparameters...")
    start_time = time()
    
    try:
        # Training
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=True)
        
        # Evaluation on train and test
        results = evaluate_model(model, X_train, y_train, X_test, y_test, num_classes, 'CatBoost')
        results['train']['time'] = time() - start_time
        results['test']['time'] = time() - start_time
        
        return results
    
    except Exception as e:
        print(f"❌ Error in CatBoost: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_comparison(dataset_id: int, results: dict):
    """Comparative plot of the models"""
    # Filter models with valid results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Create directory to save plots
    os.makedirs("comparison_plots", exist_ok=True)
    
    metrics = ['auc_ovo', 'accuracy', 'cross_entropy']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        model_names = []
        values = []
        
        for model, res in valid_results.items():
            model_names.append(model)
            values.append(res['train'][metric])
        
        bars = plt.bar(model_names, values)
        plt.title(f'{metric} Comparison - Dataset {dataset_id}', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.ylim(0, 1.1 if metric != 'cross_entropy' else None)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'comparison_plots/comparison_{dataset_id}_{metric}_train.png', dpi=300)
        plt.close()

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        model_names = []
        values = []
        
        for model, res in valid_results.items():
            model_names.append(model)
            values.append(res['test'][metric])
        
        bars = plt.bar(model_names, values)
        plt.title(f'{metric} Comparison - Dataset {dataset_id}', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.ylim(0, 1.1 if metric != 'cross_entropy' else None)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'comparison_plots/comparison_{dataset_id}_{metric}_test.png', dpi=300)
        plt.close()

    # Time plot
    plt.figure(figsize=(12, 6))
    model_names = []
    times = []
    
    for model, res in valid_results.items():
        model_names.append(model)
        times.append(res['train']['time'] + res['test']['time'])
    
    bars = plt.bar(model_names, times, color='orange')
    plt.title(f'Execution Time - Dataset {dataset_id}', fontsize=14)
    plt.ylabel('Time (s)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'comparison_plots/comparison_{dataset_id}_time.png', dpi=300)
    plt.close()

def generate_summary_table(results: dict):
    """Generates a summary table of the results"""
    summary_data = []
    
    for dataset_id, models in results.items():
        for model_name, metrics in models.items():
            if metrics is not None:
                # Training results
                summary_data.append({
                    'Dataset': dataset_id,
                    'Model': model_name,
                    'Set': 'Train',
                    'AUC OVO': f"{metrics['train']['auc_ovo']:.4f}",
                    'Accuracy': f"{metrics['train']['accuracy']:.4f}",
                    'Cross-Entropy': f"{metrics['train']['cross_entropy']:.4f}",
                    'Time (s)': f"{metrics['train']['time']:.2f}"
                })
                # Test results
                summary_data.append({
                    'Dataset': dataset_id,
                    'Model': model_name,
                    'Set': 'Test',
                    'AUC OVO': f"{metrics['test']['auc_ovo']:.4f}",
                    'Accuracy': f"{metrics['test']['accuracy']:.4f}",
                    'Cross-Entropy': f"{metrics['test']['cross_entropy']:.4f}",
                    'Time (s)': f"{metrics['test']['time']:.2f}"
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    df.to_csv('results_summary.csv', index=False)
    
    # Return formatted table
    return tabulate(df, headers='keys', tablefmt='grid', showindex=False)

def compare_models_with_hyperopt(dataset_id: int):
    """Compares the 4 models on a specific dataset with hyperparameter optimization"""
    print(f"\n{'='*50}")
    print(f"Comparing models on dataset {dataset_id}")
    
    try:
        # Load data with 80/20 split
        X_train, y_train, X_test, y_test, num_classes = load_and_preprocess_data(dataset_id)
        
        # Optimize hyperparameters for each model using ONLY the training set
        models = {
            'SAINT': (optimize_saint_hyperparameters, train_saint_with_params),
            'LightGBM': (optimize_lightgbm_hyperparameters, train_lightgbm_with_params),
            'XGBoost': (optimize_xgboost_hyperparameters, train_xgboost_with_params),
            'CatBoost': (optimize_catboost_hyperparameters, train_catboost_with_params)
        }
        
        dataset_results = {}
        dataset_best_params = {}
        
        for name, (optimizer, trainer) in models.items():
            try:
                print(f"\n🔍 Optimizing {name} using ONLY the training set...")
                best_params = optimizer(X_train, y_train, num_classes)
                dataset_best_params[name] = best_params
                
                print(f"\n🎯 Training {name} with best hyperparameters on the ENTIRE training set...")
                results = trainer(X_train, y_train, X_test, y_test, num_classes, best_params)
                
                if results is not None:
                    dataset_results[name] = results
                    print(f"✅ {name} - Results:")
                    print("Train (80%):")
                    print(f"  AUC OVO: {results['train']['auc_ovo']:.4f}")
                    print(f"  Accuracy: {results['train']['accuracy']:.4f}")
                    print(f"  Cross-Entropy: {results['train']['cross_entropy']:.4f}")
                    print(f"  Time: {results['train']['time']:.2f}s")
                    print("Test (20%):")
                    print(f"  AUC OVO: {results['test']['auc_ovo']:.4f}")
                    print(f"  Accuracy: {results['test']['accuracy']:.4f}")
                    print(f"  Cross-Entropy: {results['test']['cross_entropy']:.4f}")
                    print(f"  Time: {results['test']['time']:.2f}s")
                else:
                    dataset_results[name] = None
                    print(f"❌ {name} returned None")
                    
            except Exception as e:
                print(f"❌ Error in {name}: {str(e)}")
                dataset_results[name] = None
                dataset_best_params[name] = None
        
        # Store results
        results_comparison[dataset_id] = dataset_results
        best_params[dataset_id] = dataset_best_params
        
        # Plot comparison
        plot_comparison(dataset_id, dataset_results)
        
        # Save results to CSV in a structured way
        save_results_to_csv(results_comparison)
        
        return dataset_results, dataset_best_params
    
    except Exception as e:
        print(f"❌ Error processing dataset {dataset_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def save_results_to_csv(results: dict):
    """Saves results to a well-structured CSV"""
    # Create lists to store data
    data = []
    
    # For each dataset
    for dataset_id, models in results.items():
        # For each model
        for model_name, metrics in models.items():
            if metrics is not None:
                # Add training results
                data.append({
                    'Dataset': dataset_id,
                    'Model': model_name,
                    'Set': 'Train',
                    'AUC_OVO': float(metrics['train']['auc_ovo']),
                    'Accuracy': float(metrics['train']['accuracy']),
                    'Cross_Entropy': float(metrics['train']['cross_entropy']),
                    'Time': float(metrics['train']['time'])
                })
                # Add test results
                data.append({
                    'Dataset': dataset_id,
                    'Model': model_name,
                    'Set': 'Test',
                    'AUC_OVO': float(metrics['test']['auc_ovo']),
                    'Accuracy': float(metrics['test']['accuracy']),
                    'Cross_Entropy': float(metrics['test']['cross_entropy']),
                    'Time': float(metrics['test']['time'])
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv('results_comparison.csv', index=False)
    
    # Also save a pivoted version for analysis
    pivot_df = df.pivot_table(
        index=['Dataset', 'Model'],
        columns='Set',
        values=['AUC_OVO', 'Accuracy', 'Cross_Entropy', 'Time']
    )
    pivot_df.to_csv('results_comparison_pivot.csv')

def compute_cd(mean_ranks, n_datasets, alpha=0.05):
    """Calculates the Critical Difference (CD) for the Nemenyi test"""
    k = len(mean_ranks)  # number of algorithms
    q_alpha = norm.ppf(1 - alpha/2)  # critical value from the normal distribution
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6 * n_datasets))
    return cd

def perform_demsar_test(results: dict, metric: str):
    """Performs Demšar's statistical test for a specific metric"""
    # Prepare data for the test
    datasets = list(results.keys())
    models = list(next(iter(results.values())).keys())
    
    # Create results matrix
    results_matrix = np.zeros((len(datasets), len(models)))
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            if results[dataset][model] is not None:
                # Access the metric from the test set
                results_matrix[i, j] = results[dataset][model]['test'][metric]
            else:
                results_matrix[i, j] = np.nan
    
    # Calculate rankings for each dataset
    ranks = np.zeros_like(results_matrix)
    for i in range(len(datasets)):
        valid_indices = ~np.isnan(results_matrix[i])
        if np.sum(valid_indices) > 0:
            ranks[i, valid_indices] = rankdata(results_matrix[i, valid_indices])
    
    # Calculate mean rankings
    mean_ranks = np.nanmean(ranks, axis=0)
    
    # Friedman test
    valid_data = results_matrix[~np.isnan(results_matrix).any(axis=1)]
    if len(valid_data) > 0:
        friedman_stat, p_value = friedmanchisquare(*[valid_data[:, i] for i in range(valid_data.shape[1])])
    else:
        friedman_stat, p_value = np.nan, np.nan
    
    return {
        'mean_ranks': mean_ranks,
        'friedman_stat': friedman_stat,
        'p_value': p_value,
        'ranks': ranks,
        'results_matrix': results_matrix
    }

def plot_cd_diagram(mean_ranks, models, metric_name, alpha=0.05):
    """Plots the Critical Difference (CD) diagram"""
    # Calculate CD
    n_datasets = len(mean_ranks)
    cd = compute_cd(mean_ranks, n_datasets, alpha)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot rankings
    y_pos = np.arange(len(models))
    plt.barh(y_pos, mean_ranks, align='center')
    plt.yticks(y_pos, models)
    plt.xlabel('Mean Ranking')
    plt.title(f'Rankings for {metric_name}\nCD = {cd:.3f}')
    
    # Add CD lines
    plt.axvline(x=mean_ranks.min() + cd, color='r', linestyle='--', label=f'CD = {cd:.3f}')
    
    # Add values to rankings
    for i, v in enumerate(mean_ranks):
        plt.text(v, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'comparison_plots/cd_diagram_{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistical_report(results: dict):
    """Generates a complete statistical report using Demšar's protocol"""
    metrics = ['auc_ovo', 'accuracy', 'cross_entropy']
    models = list(next(iter(results.values())).keys())
    
    print("\n📊 Statistical Report (Demšar's Protocol)")
    print("=" * 50)
    
    for metric in metrics:
        print(f"\nMetric: {metric.upper()}")
        print("-" * 30)
        
        # Perform Demšar's test
        demsar_results = perform_demsar_test(results, metric)
        
        # Print results
        print("\nMean Rankings:")
        for model, rank in zip(models, demsar_results['mean_ranks']):
            print(f"{model}: {rank:.3f}")
        
        print(f"\nFriedman Test:")
        print(f"Statistic: {demsar_results['friedman_stat']:.3f}")
        print(f"p-value: {demsar_results['p_value']:.3f}")
        
        # Plot CD diagram
        plot_cd_diagram(demsar_results['mean_ranks'], models, metric)
        
        # Post-hoc analysis if necessary
        if demsar_results['p_value'] < 0.05:
            print("\nSignificant differences found!")
            print("Performing post-hoc analysis...")
            
            # Calculate CD
            n_datasets = len(demsar_results['mean_ranks'])
            cd = compute_cd(demsar_results['mean_ranks'], n_datasets)
            print(f"Critical Difference (CD): {cd:.3f}")
            
            # Identify groups of not significantly different models
            ranks = demsar_results['mean_ranks']
            groups = []
            current_group = []
            
            for i in range(len(ranks)):
                if not current_group:
                    current_group.append(i)
                else:
                    if ranks[i] - ranks[current_group[0]] <= cd:
                        current_group.append(i)
                    else:
                        groups.append(current_group)
                        current_group = [i]
            
            if current_group:
                groups.append(current_group)
            
            print("\nGroups of not significantly different models:")
            for i, group in enumerate(groups):
                group_models = [models[j] for j in group]
                print(f"Group {i+1}: {', '.join(group_models)}")
        
        print("\n" + "-" * 30)
    
    print("\n" + "=" * 50)
    print("CD diagrams have been saved in the 'comparison_plots' folder")

if __name__ == "__main__":
    # List of datasets to compare
    dataset_ids = [23381, 1063, 6332, 40994, 1510, 1480, 11, 29, 15, 188, 1464, 37, 469, 458, 54, 50, 307, 31, 1494, 1468, 40966, 1068, 1462, 1049, 23, 1050, 1501, 40975, 40982]
    
    # Compare models on each dataset
    for dataset_id in dataset_ids:
        compare_models_with_hyperopt(dataset_id)
    
    # Final summary
    print("\n🎉 Comparison complete! Summary:")
    
    # Generate summary table
    summary_table = generate_summary_table(results_comparison)
    print(summary_table)
    
    # Generate statistical report using Demšar's protocol
    generate_statistical_report(results_comparison)
    
    print("\n📊 Comparison plots saved in the 'comparison_plots' folder")
    print("📝 Detailed results saved in 'results_comparison.csv'")
