class EarlyStopping:
    """Early stops the training if metric doesn't improve after a given patience.
    
    Attributes:
        mode (str) : Mode of metric optimization. Defalut: min
        patience (int) : How long to wait after last time metric improved.
        delta (int) : Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, mode='min', patience=10, delta=0):
        """Initialize EarlyStopping class."""

        assert mode in ['min', 'max'], "mode must be 'min' or 'max'."
        self.mode = mode
        self.patience = patience
        self.delta = delta 
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, metric):
        """Earlystop call"""
        improved = (self.mode == 'min' and metric < self.best_score - self.delta ) or \
                   (self.mode == 'max' and metric > self.best_score + self.delta)

        if improved:
            self.best_score = metric
            self.counter = 0
            update = True
            return update, self.best_score, self.counter
        else:
            self.counter += 1
            update = False
            if self.counter >= self.patience:
                self.early_stop = True
            return update, self.best_score, self.counter