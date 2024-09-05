import wandb

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # Use Bayesian optimization to find the best hyperparameters
    'metric': {
        'name': 'val_loss',  # Metric to optimize
        'goal': 'minimize'   # Goal: minimize or maximize the metric
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64]
        },
        'learning_rate': {
            'min': 1e-5,
            'max': 1e-3
        },
        'epochs': {
            'values': [10, 20, 30]
        },
        'model_name': {
            'values': ['vit_small_softmax', 'resnet_small_softmax', 'efficientnet_small_softmax']
        },
        'patience': {
            'values': [3, 5, 7]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='medical_image_classification')
print(f'Sweep ID: {sweep_id}')
