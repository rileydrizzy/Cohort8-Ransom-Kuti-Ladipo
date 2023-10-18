"""doc
"""

import os
import wandb
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="yb-audioloom"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"



