
class MoeaResult():
    def __init__(self, opt, times, metrics, train_metrics = None, train_metrics_last = None):
        self.opt = opt
        self.times = times
        self.metrics = metrics
        self.train_metrics = train_metrics
        self.train_metrics_last = train_metrics_last
