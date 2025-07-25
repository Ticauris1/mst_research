class EarlyStopper:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.verbose = verbose

    def check(self, val_score, model):
        if self.best_score is None or val_score > self.best_score:
            self.best_score = val_score
            self.best_weights = model.state_dict()
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ Early stop counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience

    def get_best_weights(self):
        return self.best_weights
