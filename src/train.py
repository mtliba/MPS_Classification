# src/train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import MedicalImageDataset, get_combined_transformations
from models import ClassificationModel
import wandb
import argparse
import os
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser(description='Train classification models for medical images using K-Fold Cross-Validation.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for K-Fold Cross-Validation')
parser.add_argument('--save_dir', type=str, default='results', help='Directory to save models and logs')
args = parser.parse_args()

# # Initialize Weights & Biases
# wandb.init(project='medical_image_classification', config={
#     'batch_size': args.batch_size,
#     'num_classes': args.num_classes,
#     'epochs': args.epochs,
#     'learning_rate': args.learning_rate,
#     'patience': args.patience,
#     'k_folds': args.k_folds
# })

# Hyperparameters
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
PATIENCE = args.patience
K_FOLDS = args.k_folds

# Prepare dataset
image_dir = 'data/raw/medical_images'
transform = get_combined_transformations(image_size=(224, 224))
dataset = MedicalImageDataset(root_dir=image_dir, transform=transform)

# K-Fold Cross-Validation setup
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Initialize models
models = {
    'vit_small_softmax': ClassificationModel('vit_small', NUM_CLASSES, use_sigmoid=False),
    'resnet_small_softmax': ClassificationModel('resnet_small', NUM_CLASSES, use_sigmoid=False),
    'efficientnet_small_softmax': ClassificationModel('efficientnet_small', NUM_CLASSES, use_sigmoid=False),
    'vit_small_sigmoid': ClassificationModel('vit_small', NUM_CLASSES, use_sigmoid=True),
    'resnet_small_sigmoid': ClassificationModel('resnet_small', NUM_CLASSES, use_sigmoid=True),
    'efficientnet_small_sigmoid': ClassificationModel('efficientnet_small', NUM_CLASSES, use_sigmoid=True),
}

# Store metrics for ANOVA test
metrics_for_anova = {model_name: {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for model_name in models.keys()}

# Training function for each fold
def train_fold(model, criterion, optimizer, train_loader, val_loader, fold_idx, scheduler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            if model.use_sigmoid:
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels.long())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(model, criterion, val_loader)

        # Adjust learning rate
        if scheduler:
            scheduler.step(val_loss)



        print(f'Fold [{fold_idx+1}/{K_FOLDS}], Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            save_model(model, model_name, fold_idx)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs in fold {fold_idx+1}.')
            break

    # Log metrics to Weights & Biases
    wandb.log({
        f'fold_{fold_idx}_train_loss': train_loss / len(train_loader), 
        f'fold_{fold_idx}_val_loss': val_loss,
        f'fold_{fold_idx}_val_accuracy': val_accuracy,
        f'fold_{fold_idx}_val_precision': val_precision,
        f'fold_{fold_idx}_val_recall': val_recall,
        f'fold_{fold_idx}_val_f1': val_f1,
        'epoch': epoch
    })
    # Store the best validation metrics for this fold
    metrics_for_anova[model_name]['loss'].append(best_val_loss)
    metrics_for_anova[model_name]['accuracy'].append(val_accuracy)
    metrics_for_anova[model_name]['precision'].append(val_precision)
    metrics_for_anova[model_name]['recall'].append(val_recall)
    metrics_for_anova[model_name]['f1'].append(val_f1)

# Validation function
def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            if model.use_sigmoid:
                loss = criterion(outputs, labels.float())
                preds = (outputs > 0.5).int()  # Convert sigmoid output to binary prediction
            else:
                loss = criterion(outputs, labels.long())
                preds = torch.argmax(outputs, dim=1)

            val_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)



    return val_loss, val_accuracy, val_precision, val_recall, val_f1

# Save model function
def save_model(model, model_name, fold_idx):
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f'{model_name}_fold_{fold_idx}_best.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Saved best model for fold {fold_idx+1} to {model_path}')
    wandb.save(model_path)

# K-Fold Cross-Validation
for model_name, model in models.items():
    # Initialize a new W&B run for each model
    wandb.init(project='medical_image_classification', name=model_name, group='model_comparison', config={
        'batch_size': BATCH_SIZE,
        'num_classes': NUM_CLASSES,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'patience': PATIENCE,
        'k_folds': K_FOLDS,
        'model_name': model_name
    })

    print(f'Training {model_name} with {K_FOLDS}-Fold Cross-Validation...')
    fold_idx = 0
    for train_idx, val_idx in kf.split(dataset):
        # Subset the dataset for the current fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create DataLoader for each fold
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        criterion = nn.BCELoss() if model.use_sigmoid else nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

        wandb.watch(model, log='all')
        train_fold(model, criterion, optimizer, train_loader, val_loader, fold_idx, model_name, scheduler)
        fold_idx += 1

    # Finish the W&B run for the current model
    wandb.finish()

# Save collected metrics for ANOVA test
np.save(os.path.join(args.save_dir, 'metrics_for_anova.npy'), metrics_for_anova)