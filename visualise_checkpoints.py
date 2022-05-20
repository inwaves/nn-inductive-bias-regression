import pytorch_lightning as pl

import models.shallow_relu

# This needs a helper function that initialises different types of models
# otherwise there's no way to load checkpoints...
if __name__ == '__main__':
    model = models.shallow_relu.AsiShallowNetwork()
    model.load_from_checkpoint("ckpts/swept-lion-1641_square-baseline_10dp_ASIShallowRelu_sgd_640_relu_no_earlystopping"
                               "_100000epochs_none_schedule_cpu/epoch=6999-train_loss=0.154-val_error=0.000.ckpt")
    model.eval()
    print(model.state_dict())