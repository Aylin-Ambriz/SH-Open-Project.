from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = 'gpt2'
    output_dir: str = "./models/sound_horizon_model"
    num_train_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    use_fp16: bool = True
    use_gradient_checkpointing: bool = True