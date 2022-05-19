from pytorch_lightning import Callback


class ValidationCallback(Callback):
    def on_train_epoch_end(self) -> None:
