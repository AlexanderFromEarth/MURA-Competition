import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self,
                 num_input,
                 growth_rate,
                 bn_size=4,
                 drop_rate=0,
                 drop_mode=0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_input)
        self.conv1 = nn.Conv2d(num_input,
                               bn_size * growth_rate,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.drop_rate = drop_rate
        self.drop_mode = drop_mode

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.drop_mode == 0 or self.drop_rate == 0:
            if self.drop_mode == 1:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
            elif self.drop_mode == 2:
                out = F.dropout2d(out,
                                  p=self.drop_rate,
                                  training=self.training)
        return torch.cat([out, x], 1)


class Transition(nn.Module):
    def __init__(self, num_input, num_ouput):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_input)
        self.conv = nn.Conv2d(num_input, num_ouput, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x), inplace=True))
        out = F.avg_pool2d(out, 2)
        return out


class DenseBlock(nn.Sequential):
    def __init__(self, block, num_input, growth_rate, nblock, **kwargs):
        super(DenseBlock, self).__init__()
        for i in range(nblock):
            self.add_module(
                block.__name__.lower() + str(i),
                block(num_input + i * growth_rate, growth_rate, **kwargs))


class DenseNet(nn.Module):
    def __init__(self,
                 growth_rate=32,
                 block_types=(Bottleneck, ) * 4,
                 block_nums=(6, 12, 32, 32),
                 num_classes=2,
                 **kwargs):
        super(DenseNet, self).__init__()
        num_input = 2 * growth_rate
        self.conv = nn.Conv2d(3,
                              num_input,
                              kernel_size=7,
                              stride=2,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(num_input)
        if len(block_types) != len(block_nums):
            raise Exception('Lengths of block_types and block_nums must equal')
        self.num_blocks = len(block_nums)
        for block_idx, block_config in enumerate(zip(block_types, block_nums)):
            block_type, block_num = block_config
            block = DenseBlock(block_type, num_input, growth_rate, block_num)
            setattr(self, f'dense{block_idx}', block)
            num_input += block_num * growth_rate
            if block_idx != self.num_blocks - 1:
                trans = Transition(num_input, num_input // 2)
                setattr(self, f'trans{block_idx}', trans)
                num_input //= 2
        self.bn2 = nn.BatchNorm2d(num_input)
        self.fc = nn.Linear(num_input, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(F.relu(self.bn1(out), inplace=True),
                           kernel_size=3,
                           stride=2,
                           padding=1)
        for i in range(self.num_blocks):
            out = getattr(self, f'dense{i}')(out)
            if i != self.num_blocks - 1:
                out = getattr(self, f'trans{i}')(out)
        out = F.avg_pool2d(F.relu(self.bn2(out), inplace=True),
                           kernel_size=7,
                           stride=1).view(out.size(0), -1)
        out = F.log_softmax(self.fc(out))
        return out


def densenet121():
    return DenseNet(growth_rate=12, block_nums=(6, 12, 24, 16))
