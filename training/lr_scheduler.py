class HybridLRScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, plateau_patience=5, plateau_factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.min_lr = min_lr
        self.lr_history = []  # ✅ Add this line

        self.current_epoch = 0
        self.best_val_acc = 0
        self.epochs_since_improvement = 0

    def step(self, val_acc):
        lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(lr)  # ✅ Log LR

        if self.current_epoch >= self.warmup_epochs:
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
                if self.epochs_since_improvement >= self.plateau_patience:
                    new_lr = max(lr * self.plateau_factor, self.min_lr)
                    for group in self.optimizer.param_groups:
                        group['lr'] = new_lr
                    print(f"🔻 Plateau: Reducing LR to {new_lr:.6f}")
                    self.epochs_since_improvement = 0

        self.current_epoch += 1

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]