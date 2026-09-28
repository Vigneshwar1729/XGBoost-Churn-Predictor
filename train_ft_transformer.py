import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FTTransformerConfig

def train_ft_transformer(train_path: str, val_path: str):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    cat_col_names = ["Gender", "HasCrCard", "IsActiveMember"]
    num_col_names = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts"]
    target_col = "Exited"

    print("Configuring Data...")
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )

    # Set gpus=1 if running in Google Colab or on a machine with an NVIDIA GPU
    # Set gpus=0 if running locally on CPU
    trainer_config = TrainerConfig(
        auto_lr_find=False, 
        batch_size=128,
        max_epochs=20,
        gpus=0, 
    )

    optimizer_config = OptimizerConfig()

    print("Configuring FT-Transformer Architecture...")
    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        num_attn_blocks=3,
        num_heads=4,
        out_ff_layers="128-64-32",
        out_ff_activation="ReLU",
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    print("Starting Model Training...")
    tabular_model.fit(train=train_df, validation=val_df)
    
    model_dir = "ft_transformer_model"
    tabular_model.save_model(model_dir)
    print(f"FT-Transformer Model weights and architecture saved to ./{model_dir}/")

if __name__ == "__main__":
    train_ft_transformer(train_path="./splits/train.csv", val_path="./splits/val.csv")
