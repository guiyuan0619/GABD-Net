import os
import torch
import ml_collections


save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)


n_filts = 32
cosineLR = True
n_channels = 3
n_labels = 1
epochs = 3000
img_size = 224


print_frequency = 10
save_frequency = 100
vis_frequency = 100
early_stopping_patience = 300


pretrain = False


task_name = 'Kvasir-SEG'


learning_rate = 1e-3
batch_size = 16
gradient_accumulation_steps = 1


model_name = 'GABDNet'


use_adamw = True
use_cosine_lr = True


session_name = 'session3'
test_session = "session3"


train_dataset = os.path.join('./datasets', task_name, 'Train_Folder')
val_dataset = os.path.join('./datasets', task_name, 'Val_Folder')
test_dataset = os.path.join('./datasets', task_name, 'Test_Folder')


save_path = os.path.join(task_name, model_name, session_name)
model_path = os.path.join(save_path, 'models')
tensorboard_folder = os.path.join(save_path, 'tensorboard_logs')
logger_path = os.path.join(save_path, f"{session_name}.log")
visualize_path = os.path.join(save_path, 'visualize_val')


def ensure_directories():
    directories = [save_path, model_path, tensorboard_folder,
                   os.path.dirname(logger_path), visualize_path]

    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            except OSError as e:
                print(f"Failed to create directory {directory}: {e}")
                raise


def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64
    config.n_classes = 1
    return config


def validate_config():
    assert batch_size > 0, "batch_size must be greater than 0"
    assert learning_rate > 0, "learning_rate must be greater than 0"
    assert epochs > 0, "epochs must be greater than 0"
    assert img_size > 0, "img_size must be greater than 0"
    assert n_filts > 0 and n_filts <= 32, "n_filts must be in (0, 32] to avoid parameter explosion"
    assert n_channels > 0, "n_channels must be greater than 0"
    assert n_labels >= 1, "n_labels must be greater than or equal to 1"
    assert gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be greater than or equal to 1"


    if not os.path.exists(train_dataset):
        print(f"Warning: training dataset path does not exist: {train_dataset}")
    if not os.path.exists(val_dataset):
        print(f"Warning: validation dataset path does not exist: {val_dataset}")
    if not os.path.exists(test_dataset):
        print(f"Warning: test dataset path does not exist: {test_dataset}")


if __name__ != "__main__":
    validate_config()
