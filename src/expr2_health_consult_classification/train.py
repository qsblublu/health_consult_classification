import torch
import torch.nn.functional as F
import yaml
import time
from torch.utils.data import DataLoader
from easydict import EasyDict

from dataset import TextDataset
from model import TextCNN
from utils import Metric


def train(net: torch.nn.Module, dataset: torch.utils.data.dataset, cfg: EasyDict):
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    loss_metric = Metric()
    # optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr)
    total_iter = len(dataset)

    print('--------------------- start train ------------------------')
    for epoch in range(cfg.epoch):
        optimizer = torch.optim.SGD(net.parameters(), lr=(cfg.lr / pow(2, epoch)))
        for idx, sample in enumerate(dataloader):
            sentence, label = sample['sentence'], sample['label']
            logit = net(sentence)

            optimizer.zero_grad()
            loss = F.cross_entropy(logit, label)
            loss_metric.update(loss)
            loss.backward()
            optimizer.step()

            if (idx + 1) % cfg.print_freq == 0:
                print(f'epoch {epoch} iter {idx + 1}/{total_iter} loss {loss_metric.value()}')

    torch.save(net.state_dict(), '../../model/health_consult_classification.pth')
    print('save model to ../../model/health_consult_classification.pth')
    print('------------------------------ end train --------------------------------')


def evaluate(net: torch.nn.Module, dataset: torch.utils.data.dataset, pretrain_file):
    net.load_state_dict(torch.load(pretrain_file))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    total_num = len(dataset)
    accurate_num = 0

    for idx, sample in enumerate(dataloader):
        sentence, label = sample['sentence'], sample['label']

        with torch.no_grad():
            predict = F.softmax(net(sentence), dim=1)
            _, top1 = predict.topk(1, 1)

            if top1 == label:
                accurate_num += 1

    print(f'evaluate model accurate rate is {accurate_num / total_num}')


if __name__ == '__main__':
    with open('config.yaml', mode='r') as f:
        config = EasyDict(yaml.load(f))

    train_dataset = TextDataset(config.dataset.train_data_file, config.dataset.dict_file)
    test_dataset = TextDataset(config.dataset.test_data_file, config.dataset.dict_file)
    model = TextCNN(config.model)

    train(model, train_dataset, config.train)
    evaluate(model, test_dataset, '../../model/health_consult_classification.pth')
