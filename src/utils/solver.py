import torch
import pathlib
import tqdm
import pickle
import src.utils.unpickler


class Solver:
    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.modules.loss._Loss,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            dataloaders: dict,
            config: dict,
            path: str | pathlib.Path = '',
            device: str | torch.device = 'cpu',
    ):
        """A class to train and evaluate a PyTorch model.
        
        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            criterion (torch.nn.modules.loss._Loss): The loss function to be optimized.
            optimizer (torch.optim.Optimizer): The optimizer to be used for training.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler to be used for training.
            dataloaders (dict): Dictionary containing the DataLoaders for the training, validation, and test sets.
            config (dict): Dictionary containing the training configuration.
            path (str, pathlib.Path): The path to save the solver to. Default: `''`.
            device (str, torch.device): The device to use for training. Default: `'cpu'`.        
        """
        # Save attributes.
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = dataloaders
        self.config = config
        self.path = path
        self.num_epochs = config['training']['num_epochs']
        self.device = device
        # Initialize loss history.
        self._reset()

    def train(self):

        for epoch in range(self.running_epoch, self.num_epochs):
            # Empty the GPU cache.
            torch.cuda.empty_cache()
            # Train.
            self.model.train()
            trn_loss = self._run_one_trn_epoch()
            print(f"Train Summary | Epoch {epoch:02d} | Loss = {trn_loss:.10f}")
            # Validate.
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_one_val_epoch()
            print(f"Valid Summary | Epoch {epoch:02d} | Loss = {val_loss:.10f}")
            # Update scheduler.
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            last_lr = self.scheduler.get_last_lr()[0]
            print(f"\tLearning rate = {last_lr}")
            # Save model.
            self.trn_loss_history[epoch] = trn_loss
            self.val_loss_history[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_to_path(self.path);
                print(f"Solver for best model saved at '{self.path}'.")
            # Update epoch.
            self.running_epoch += 1
            print('----------------------------------------------------------------')

        return self

    def _run_one_trn_epoch(self):
        running_loss = 0.0
        for batch in tqdm.tqdm(self.loaders['trn_loader'], "Training epoch"):
            # Get the inputs and targets.waveform_mixtr, waveform_truth, waveform_clean, waveform_noise
            batch_mixtr, batch_truth, _, batch_noise, batch_ref_mic, _ = batch
            batch_mixtr = batch_mixtr.to(self.device)
            batch_truth = batch_truth.to(self.device)
            batch_noise = batch_noise[:,0].to(self.device)
            batch_ref_mic = batch_ref_mic.to(self.device)
            # Forward pass.
            batch_estim = self.model(batch_mixtr).squeeze(1)
            # Compute loss.
            loss = self.criterion(
                waveform_clean_estim = batch_estim,
                waveform_clean_truth = batch_truth,
                waveform_noise_estim = batch_mixtr[:,0] - batch_estim,
                waveform_noise_truth = batch_noise,
            )
            try:
                # Backward pass and optimization.
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['training']['grad_max_norm'], norm_type=1)
                # Optimization step.
                self.optimizer.step()
            except RuntimeError as e:
                print("Error. Skipping this batch.")
                torch.cuda.empty_cache()
            running_loss += loss.item()
        return running_loss
    
    def _run_one_val_epoch(self):
        running_loss = 0.0
        for batch in tqdm.tqdm(self.loaders['val_loader'], "Validating epoch"):
            batch_mixtr, batch_truth, _, batch_noise, batch_ref_mic, _ = batch
            batch_mixtr = batch_mixtr.to(self.device)
            batch_truth = batch_truth.to(self.device)
            batch_noise = batch_noise[:,0].to(self.device)
            batch_ref_mic = batch_ref_mic.to(self.device)
            # Forward pass.
            batch_estim = self.model(batch_mixtr).squeeze(1)
            # Compute loss.
            loss = self.criterion(
                waveform_clean_estim = batch_estim,
                waveform_clean_truth = batch_truth,
                waveform_noise_estim = batch_mixtr[:,0] - batch_estim,
                waveform_noise_truth = batch_noise,
            )
            # Update loss.
            running_loss += loss.item()
        return running_loss

    def _reset(self):
        print('----------------------------------------------------------------')
        if self.config['training']['continue_from']:
            print(f"Loading checkpoint solver: '{self.config['training']['continue_from']}'.")     
            self.load_from_path(self.config['training']['continue_from'])
        else:
            print(f"Initializing solver from scratch.")
            self.running_epoch = 0
            self.trn_loss_history = torch.zeros(self.num_epochs, device=self.device)
            self.val_loss_history = torch.zeros(self.num_epochs, device=self.device)            
            self.prev_val_loss = float('inf')
            self.best_val_loss = float('inf')   
        print('----------------------------------------------------------------')

    def _serialize(self):
        """Serialize the solver into a dictionary containing relevant attributes."""
        package = {
            'model_name': self.config['model']['model_name'],
            'model_args': self.config['model']['model_args'],            
            'model_state_dict': self.model.state_dict(),
            'optimizer_name': self.config['optimizer']['optimizer_name'],
            'optimizer_args': self.config['optimizer']['optimizer_args'],
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_name': self.config['scheduler']['scheduler_name'],
            'scheduler_args': self.config['scheduler']['scheduler_args'],
            'scheduler_state_dict': self.scheduler.state_dict(),
            'running_epoch': self.running_epoch,
            'trn_loss_history': self.trn_loss_history.tolist(),
            'val_loss_history': self.val_loss_history.tolist(),
        }
        return package
    
    def _deserialize(self, package):
        """Deserialize a solver from a dictionary."""
        self.model.load_state_dict(package['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(package['optimizer_state_dict'])
        self.scheduler.load_state_dict(package['scheduler_state_dict'])
        self.running_epoch = package['running_epoch'] + 1
        self.trn_loss_history = torch.Tensor(package['trn_loss_history']).to(self.device)
        self.val_loss_history = torch.Tensor(package['val_loss_history']).to(self.device)
        self.prev_val_loss = self.val_loss_history[self.running_epoch-1]
        self.best_val_loss = self.val_loss_history[:self.running_epoch].min()

    def save_to_path(self, solver_path):
        """Save the solver to a given file path."""
        solver_package = self._serialize()
        with open(solver_path, 'wb') as solver_file: 
            pickle.dump(solver_package, solver_file)

    def load_from_path(self, solver_path):
        """Load a solver from a given file path."""
        if str(self.device == 'cpu'):
            with open(solver_path, 'rb') as solver_file:
                solver_package = src.utils.unpickler.UnpicklerCPU(solver_file).load()            
        else:
            with open(solver_path, 'rb') as solver_file: 
                solver_package = pickle.load(solver_file)
        self._deserialize(solver_package)