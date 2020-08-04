import torch

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, ground_truth):
        l = ((pred - ground_truth)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)   #[4, 16, 64, 64] -> [4]
        return l    # size = batch_size