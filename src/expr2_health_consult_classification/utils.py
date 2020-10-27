class Metric(object):
    def __init__(self):
        self.loss_sum = 0
        self.num = 0

    def update(self, loss):
        self.loss_sum += loss
        self.num += 1

    def value(self):
        loss_average = self.loss_sum / self.num
        self.reset()

        return loss_average

    def reset(self):
        self.loss_sum = 0
        self.num = 0


health_consult_class_map = {
    '中医': 0,
    '手术': 1,
    '点滴': 2,
    '休息': 3,
    '多喝水吃水果蔬菜': 4
}
