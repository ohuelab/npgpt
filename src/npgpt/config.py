from pydantic import BaseModel


class SmilesGptTrainingConfig(BaseModel):
    run_name: str | None = None
    project_name: str | None = "npgpt"

    accelerator: str = "gpu"
    devices: int | list[int] = 1
    strategy: str = "ddp"

    batch_size: int = 256
    train_ratio: float = 0.8
    num_workers: int = 32
    max_epochs: int = 30
    min_epochs: int = 15
    max_length: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    adam_eps: float = 1e-8
    adam_betas: tuple[float, float] = (0.9, 0.999)
    scheduler_T_max: int = 150_000
    final_learning_rate: float = 5e-8
    vocab_size: int = 1_000
    min_frequency: int = 2
    top_p: float = 0.96
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 12 * 48


class SmilesGptGenerationConfig(BaseModel):
    num_samples: int = 5
    max_length: int = 512
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.96
