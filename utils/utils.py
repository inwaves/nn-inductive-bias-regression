import torch
import wandb

from datasets.dataset import *
from utils.custom_dataloader import CustomDataLoader
from utils.parsers import parse_args, parse_bool
from utils.adjust_data import DataAdjuster
from utils.selectors import select_dataset, select_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def setup():
    args = parse_args()
    wandb.init(project="gen2",
               entity="inwaves",
               config={"model_type":           args.model_type,
                       "nonlinearity_type":    args.nonlinearity,
                       "hidden_units":         args.hidden_units,
                       "lr":                   args.learning_rate,
                       "dataset":              args.dataset,
                       "generalisation_task":  args.generalisation_task,
                       "adjust_data_linearly": args.adjust_data_linearly,
                       "normalise":            args.normalise,
                       "num_datapoints":       args.num_datapoints,
                       "optimiser":            args.optimiser,
                       "internal_tag":         args.tag,
                       "lr_schedule":          args.lr_schedule,
                       "early_stopping":       args.early_stopping,
                       "loss":                 args.loss,})

    # Set up the data.
    (x_train, y_train, x_test, y_test), fn = select_dataset(args)
    da_train = DataAdjuster(x_train, y_train)
    da_test = DataAdjuster(x_test, y_test, da_train.x_min, da_train.x_max)

    if parse_bool(args.normalise):
        da_train.normalise()
        da_test.normalise()
        print(f"Normalising data because flag is: {args.normalise}")

    # Adjust the data linearly.
    if parse_bool(args.adjust_data_linearly):
        da_train.adjust()
        da_test.adjust()
        print(f"Adjusting data linearly because flag is: {args.adjust_data_linearly}")

    training_data = np.array(list(zip(da_train.x, da_train.y)))
    test_data = np.array(list(zip(da_test.x, da_test.y)))

    custom_dataloader = CustomDataLoader(training_data, test_data, device)
    train_dataloader = custom_dataloader.train_dataloader()
    test_dataloader = custom_dataloader.test_dataloader() if len(da_test.x) > 0 else None

    # Set up the model.
    model = select_model(da_train, da_test, fn, args.adjust_data_linearly, args.normalise, args.grid_resolution,
                         args.model_type, args.hidden_units, args.learning_rate, args.optimiser, args.lr_schedule,
                         args.init, args.a_w, args.a_b, args.loss)

    return train_dataloader, test_dataloader, da_train, da_test, args, model, fn
