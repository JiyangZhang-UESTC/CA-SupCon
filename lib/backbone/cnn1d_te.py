from .cnn1d_cwru import BasicBlock

import torch
import torch.nn as nn 
import torch.nn.functional as F


class ResNetTE(nn.Module):
    def __init__(self, block, seq_len, num_blocks, planes, kernel_size, 
                    pool_size, linear_plane):
        super(ResNetTE, self).__init__()

        self.seq_len = seq_len
        self.planes = planes
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(52, self.planes[0], kernel_size=self.kernel_size, 
                                padding=self.kernel_size//2)
        self.bn1 = nn.BatchNorm1d(self.planes[0])
        self.layer = self._make_layer(block, num_blocks, pool_size)

        self.linear = nn.Linear(self.planes[-1]*(self.seq_len
                                //(pool_size**(num_blocks-1))), linear_plane)


    def _make_layer(self, block, num_blocks, pool_size):
        layers = []

        for i in range(num_blocks-1):
            layers.append(block(self.planes[i], self.planes[i+1], 
                            kernel_size=self.kernel_size))
            layers.append(nn.MaxPool1d(pool_size))
        
        layers.append(block(self.planes[-2], self.planes[-1], 
                        kernel_size=self.kernel_size))

        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = out[:,:,:self.seq_len]
        out = self.layer(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear(out))

        return out


def create_cnn1d_te(logger, seq_len, num_blocks, planes, kernel_size, pool_size, linear_plane):
    logger.info("===> create model from create_cnn1d_te func")
    logger.info("sequence length: {}".format(seq_len))
    logger.info("number of blocks: {}".format(num_blocks))
    logger.info("planes: {}".format(planes))
    logger.info("kernel size: {}".format(kernel_size))
    logger.info("pool size: {}".format(pool_size))
    logger.info("linear plane: {}".format(linear_plane))
    
    net = ResNetTE(BasicBlock, seq_len, num_blocks, planes, kernel_size, pool_size, linear_plane=linear_plane)
    logger.info("model: {}\n".format(net))
    
    return net


if __name__ == "__main__":
    from torchsummary import summary
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    net = ResNetTE(BasicBlock, 200, 2, [16, 16, 16], kernel_size=10, pool_size=2, linear_plane=100)
    device = torch.device("cuda:0")
    net = net.to(device)

    summary(net, (52, 200))
