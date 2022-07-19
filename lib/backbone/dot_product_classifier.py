import torch.nn as nn

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, linear_plane, num_classes):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(linear_plane, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x


def create_dot_product_classifier(logger, linear_plane, num_classes):
    logger.info("===> create dot product classifier")
    
    net = DotProduct_Classifier(linear_plane, num_classes)
    logger.info("model: {}\n".format(net))
    
    return net
