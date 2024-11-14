from config import hdf5_path, scaler_path_aug, pca_path_aug, all_pprocessed_data_directory_aug
from preprocessing import DataPreprocessor
import warnings
import pandas as pd
import numpy as np
from rich import print
from rich.pretty import pprint
from sklearn.model_selection import train_test_split
from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig,
)
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.tabular_model_tuner import TabularModelTuner

from torchinfo import summary



def np_to_pd(x,y):
    xy = np.concatenate([x, y[:, np.newaxis]], axis=1)
    num_features = xy.shape[1] - 1  
    df = pd.DataFrame(xy, columns=[f"feature_{i}" for i in range(num_features)] + ["target"])
    
    return df

paths = {
        'data_path': hdf5_path,
        'scaler_path':scaler_path_aug,
        'pca_path':pca_path_aug,
        'all_pprocessed_data_path':all_pprocessed_data_directory_aug,
    }

data_preprocessor = DataPreprocessor(paths,crop_augment_fold = 3)
data_splits = data_preprocessor.load_pprocessed_data()

x_train, y_train, x_val, y_val, x_test, y_test = data_splits['x_train'], data_splits['y_train'], data_splits['x_val'], data_splits['y_val'], data_splits['x_test'], data_splits['y_test']



print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

train_df = np_to_pd(x_train,y_train)
val_df = np_to_pd(x_val,y_val)

num_col_names = train_df.columns[:-1].tolist()  
cat_col_names = [] 
data_config = DataConfig(
    target=["target"],  
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)

trainer_config = TrainerConfig(
    batch_size=64,
    max_epochs=300,
    #min_epochs=120,
    #early_stopping="valid_loss",  # Monitor valid_loss for early stopping
    #early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
    #early_stopping_patience=10,  # No. of epochs of degradation training will wait before terminating
    checkpoints=None,
    load_best=False,  # After training, load the best checkpoint
    progress_bar="none",  # Turning off Progress bar
    trainer_kwargs=dict(enable_model_summary=False),  # Turning off model summary
)
optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="", initialization="random" , use_batch_norm=False
).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-512",  # Number of nodes in each layer
    activation="ReLU",  # Activation between each layers
    learning_rate=1e-4,
    head="LinearHead",  # Linear Head
    head_config=head_config,  # Linear Head Config
    dropout=0.4
)

search_space = {
    "model_config__layers": ["1024-256"],
    "model_config__dropout": [0.4],
    "trainer_config__early_stopping": [None],
}

tuner = TabularModelTuner(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = tuner.tune(
        train=train_df,
        validation=val_df,
        search_space=search_space,
        strategy="grid_search",
        # cv=5, # Uncomment this to do a 5 fold cross validation
        metric="accuracy",
        mode="max",
        progress_bar=True,
        verbose=True, # Make True if you want to log metrics and params each iteration
        return_best_model=True,
    )

print(result.best_model)
print(result)