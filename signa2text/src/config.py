"""doc
"""

from pydantic import BaseModel
import hydra


class Arg_type(BaseModel):
    save_every: int


@hydra.main(config_name="train", config_path="config", version_base="1.2")
def main(cfg):
    A = cfg.wandb_params.model_run_id
    print(type(A))


main()
