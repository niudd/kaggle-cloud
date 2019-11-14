import torch
import os
import shutil
import logging


# save_dict = \
# {
#     'epoch': epoch,
#     'state_dict': model.state_dict(),
#     'optim_dict' : optimizer.state_dict(),
#     'metrics': {'train_loss': train_loss, 'val_loss': val_loss, 'train_iou': train_iou, 'val_iou': val_iou}
# }


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    if 'state_dict' not in state.keys() or 'optim_dict' not in state.keys() or \
    'epoch' not in state.keys() or 'metrics' not in state.keys():
        raise ValueError('save_checkpoint: must at least contains state_dict, optim_dict, metrics, epoch')
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        #print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        pass
        #print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def save_checkpoint_snapshot(state, checkpoint, cycle_id):
    """for snapshot ensembling
    """
    if 'state_dict' not in state.keys() or 'optim_dict' not in state.keys() or \
    'cycle' not in state.keys() or 'metrics' not in state.keys():
        raise ValueError('save_checkpoint: must at least contains state_dict, optim_dict, metrics, epoch')
    filepath = os.path.join(checkpoint, 'cycle%d.pth.tar'%cycle_id)
    if not os.path.exists(checkpoint):
        #print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        pass
        #print("Checkpoint Directory exists! ")
    torch.save(state, filepath)

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return model, optimizer#checkpoint


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)




