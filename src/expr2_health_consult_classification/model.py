import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, config: dict):
        super(TextCNN, self).__init__()

        self.config = config
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.out_channel = 1
        self.kernel_sizes = config.kernel_sizes
        self.class_num = config.class_num
        self.dropout_p = config.dropout_p

        self.embed = nn.Embedding(self.embed_num, self.embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.out_channel, (kernel_size, self.embed_dim)) for kernel_size in self.kernel_sizes]
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.out_channel, self.class_num)

        # self.embed.weight.requires_grad = False

    def forward(self, x):
        # x (N, W)
        x = self.embed(x)  # x (N, W, D)

        x = x.unsqueeze(1)  # x (N, 1, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...] * len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co)] * len(Ks)

        x = torch.cat(x, 1)  # x (N, len)
        x = self.dropout(x)

        logit = self.fc(x)

        return logit
