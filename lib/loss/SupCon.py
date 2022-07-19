import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.similarity_function = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels=None, mask=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device) 
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device) # shape [batch_size, batch_size]
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape [n_views * batch_size, feature_dim]

        if self.contrast_mode == 'one':  
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

  
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # shape [2 * batch_size, 2 * batch_size]
        # mask-out self-contrast cases  
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask 

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


def create_supcon_loss(logger, device, temperature, contrast_mode, base_temperature):
    logger.info("===> create supcon loss")

    return SupConLoss(device, temperature, contrast_mode, base_temperature).to(device)



if __name__ == "__main__":
    device = torch.device("cuda:0")
    loss = SupConLoss(device, contrast_mode='all')
    features_1 = torch.tensor([[1.1, 2.1, 3.1], [7.1, 8.1, 9.1]]).float().to(device)
    features_2 = torch.tensor([[4.1, 5.1, 6.1], [10.1, 11.1, 12.1]]).float().to(device)
    
    features_1 = F.normalize(features_1, dim=1)
    features_2 = F.normalize(features_2, dim=1)

    features = torch.cat([features_1.unsqueeze(1), features_2.unsqueeze(1)], dim=1)
    
    features = F.normalize(features, dim=1)
    labels = torch.tensor([0, 1]).long().to(device)
    
    print("features: {}".format(features))
    print("labels: {}".format(labels))
    print("features shape: {}".format(features.shape))
    print("labels shape: {}".format(labels.shape))

    print(loss(features, labels))
