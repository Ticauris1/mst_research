import math

class HybridLRScheduler:
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        mode="plateau",  # ✅ 'cosine' or 'plateau'
        plateau_patience=5,
        plateau_factor=0.5,
        min_lr=1e-6
    ):
        assert mode in ["cosine", "plateau"], "mode must be 'cosine' or 'plateau'"
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.mode = mode
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.min_lr = min_lr
        self.lr_history = []

        self.current_epoch = 0
        self.best_val_acc = 0
        self.epochs_since_improvement = 0

        # ✅ Store original learning rates
        self.initial_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, val_acc=None):
        lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(lr)

        if self.current_epoch < self.warmup_epochs:
            # 🔼 Linear Warmup
            scale = (self.current_epoch + 1) / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                base_lr = self.initial_lr[i] if i < len(self.initial_lr) else group['lr']
                group['lr'] = base_lr * scale

        elif self.mode == "cosine":
            # 🌀 Cosine Annealing
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            for i, group in enumerate(self.optimizer.param_groups):
                base_lr = self.initial_lr[i] if i < len(self.initial_lr) else group['lr']
                new_lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                group['lr'] = new_lr

        elif self.mode == "plateau":
            # 📉 Reduce LR on Plateau
            if val_acc is not None:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.epochs_since_improvement = 0
                else:
                    self.epochs_since_improvement += 1
                    if self.epochs_since_improvement >= self.plateau_patience:
                        for i, group in enumerate(self.optimizer.param_groups):
                            new_lr = max(group['lr'] * self.plateau_factor, self.min_lr)
                            group['lr'] = new_lr
                            print(f"🔻 Plateau: Reducing LR group {i} to {new_lr:.6f}")
                        self.epochs_since_improvement = 0

        self.current_epoch += 1

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
