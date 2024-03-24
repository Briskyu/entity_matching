# Enhanced Entity Matching: Fine-tuning and Prompt Learning
This is the source code for paper "Leveraging Pre-trained Language Models for Enhanced Entity Matching: A Comprehensive Study of Fine-tuning and Prompt Learning Paradigms"

# Environment Setup
- Python 3.7
- PaddlePaddle 2.4.0
- PaddleNLP 2.5.0

Download all files to the user directory. Like /home/user_name/

## Directory Structure

- `~/training_args.py`: Script for setting training parameters such as learning_rate, num_train_epochs, per_device_train_batch_size.
- `~/trainer.py`: Modify the display effect during training. Calculate F1 score at the end of each epoch.
- `~/work/data/`: Training datasets. Each dataset is divided into training data for prompt and fine-tuning. (Due to GitHub's file size upload limit, the Company dataset needs to be downloaded from the original data repository: [https://github.com/anhaidgroup/deepmatcher/](https://github.com/anhaidgroup/deepmatcher/))

### Prompt Training
- `~/work/train_prompt.py`: Script for prompt training. Modify `data_dir` to set the path for training data. The `template` object of the `ManualTemplate` class represents hard prompt, while the `template` object of the `SoftTemplate` class represents soft prompt.
- `~/work/utils.py`: Utilities for prompt training, including reading training data based on file names.

### Fine-Tuning
- `~/work/fine_tuning/train.py`: Script for fine-tuning
- `~/work/fine_tuning/utils.py`: Utilities for fine-tuning
