import torch
import os
import sys
import pathlib
import yaml
import thop
import time
import datetime
sys.path.append(os.path.abspath(''))
import src.utils.losses
import src.utils.solver
import src.utils.dataset
import DeFTAN2


torch.manual_seed(1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load configuration file.
CONFIG_PATH = pathlib.Path('src', 'config', 'training.yaml')
with CONFIG_PATH.open('r') as f:
    CONFIG = yaml.safe_load(f)

print(f"\nLaunching training for '{CONFIG['model']['model_name']}'.")
print(f"\tsleeping {CONFIG['training']['sleep_time_minutes']}min...")
time.sleep(60*CONFIG['training']['sleep_time_minutes'])

SOLVER_PATH = pathlib.Path('out', 'pkls', f"{datetime.datetime.today().strftime('%m%d')}_{CONFIG['model']['model_name']}.pkl")
os.makedirs(os.path.dirname(SOLVER_PATH), exist_ok=True)
SRC_LOG_PATH = pathlib.Path('.', 'training.log')
DST_LOG_PATH = pathlib.Path('out', 'logs', f"{datetime.datetime.today().strftime('%m%d')}_{CONFIG['model']['model_name']}.log")
os.makedirs(os.path.dirname(DST_LOG_PATH), exist_ok=True)

def train(config: dict, database_root: str | pathlib.Path):
    """Define model, loaders, optimizer, criterion, scheduler, solver and launch training.

    Args:
        config (dict): Dictionary containing the training parameters.
        database_root (str, pathlib.Path): Path to the root of the training database.
    Returns:
        solver (Solver): Solver instance containing the training information, e.g. model and loss history.
    """

    f = DeFTAN2.DeFTAN2
    #f = src.model.hybridvsf.HybridVariableSpanFilter
    # Print model specifications.
    model = f( # src.model.hybridvsf.HybridVariableSpanFilter(
        **config['model']['model_args'],
    )
    print('----------------------------------------------------------------')
    num_macs, num_params, *_ = thop.profile(model, inputs=torch.randn(1, 1, 4, 64000), verbose=False)
    print(f"Num. Parameters: {int(num_params):,} | Num. MACs: {int(num_macs):,}.")    
    # Define model.
    model = f( # src.model.hybridvsf.HybridVariableSpanFilter(
        **config['model']['model_args'],
    )    
    model.to(DEVICE)
    # Define criterion.
    criterion = src.utils.losses.MultiLoss(
        list_criterion = [
            src.utils.losses.LossL1(reduction='sum'), 
            src.utils.losses.LossSTFT(reduction='sum', win_size=512, hop_size=256, beta=0.0, norm=1, comp=1.0, win_func='hamming', device=DEVICE)
        ],
        list_criterion_scale = [
            1.0,
            0.25,
        ],
    )
    # Define optimizer.
    optimizer = getattr(torch.optim, config['optimizer']['optimizer_name'])(
        params = model.parameters(),
        **config['optimizer']['optimizer_args'],
    )
    # Define scheduler.
    if config['scheduler']['scheduler_name'] == 'LambdaLR':
        # Use a custom lambda function for the learning rate schedule.
        scheduler = getattr(torch.optim.lr_scheduler, config['scheduler']['scheduler_name'])(
            optimizer = optimizer,
            #lr_lambda = lambda epoch: 0.5**next((i for i, x in enumerate([50, 70, 75, 80, 85, 90, 95, 98, 100]) if epoch <= x)),
            lr_lambda = lambda epoch: 0.5**next((i for i, x in enumerate([50, 60, 70, 80, 85, 90, 95, 98, 100]) if epoch <= x)),            

        )
    else:
        scheduler = getattr(torch.optim.lr_scheduler, config['scheduler']['scheduler_name'])(
            optimizer = optimizer,
            **config['scheduler']['scheduler_args'],  
        )
    # Define dataloaders.
    dataloaders = src.utils.dataset.dataloaders(
        root = database_root, 
        batch_size = config['training']['batch_size'],
        num_samples_trn = config['dataloader']['num_samples_trn'], 
        num_samples_val = config['dataloader']['num_samples_val'], 
        num_samples_tst = 0,            
        fixed_ref_mic = config['dataloader']['fixed_ref_mic'],
        hard_mode = config['dataloader']['hard_mode'],
    )
   # Define solver.
    solver = src.utils.solver.Solver(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        dataloaders, 
        config = config,
        path = SOLVER_PATH,
        device = DEVICE,
    )    
    # Launch training.
    solver = solver.train()

    return solver


def prepare_log_file(log_path_src: str | pathlib.Path, log_path_dst: str | pathlib.Path):
    # Handle exceptions.
    def excepthook(exc_type, exc_value, exc_traceback):
        if isinstance(exc_value, KeyboardInterrupt):
            print("\n*** TRAINING INTERRUPTED BY USER ***")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            os.replace(src=log_path_src, dst=log_path_dst)
        else:
            print("\n*** TRAINING INTERRUPTED BY EXCEPTION***")
            print(f"{exc_type.__name__}: {exc_value}")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            os.replace(src=log_path_src, dst=log_path_dst)
    # Redirect stdout to a log file.
    sys.stdout = open(log_path_src, 'wt') 
    sys.excepthook = excepthook


if __name__ == '__main__':

    # set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Redirect stdout to a default log file. 
    prepare_log_file(SRC_LOG_PATH, DST_LOG_PATH)
    print("*** START TRAINING ***\n")

    root = "/home/ovistetom/Documents/Databases_Local/MIXTURES/standard"
    # root = os.path.join('database', 'MIXTURES')
    # Train the model.
    solver = train(config=CONFIG, database_root=root)
    
    # Close and save log file.
    print("\n*** FINISHED TRAINING ***")
    sys.stdout.close()
    os.replace(src=SRC_LOG_PATH, dst=DST_LOG_PATH)