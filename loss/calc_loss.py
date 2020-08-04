import torch
from .loss import HeatmapLoss

class Calc_loss(torch.nn.Module):

    def __init__(self, nstack):
        super(Calc_loss, self).__init__()
        self.heatmapLoss = HeatmapLoss()
        self.nstack = nstack

    def forward(self, combined_heatmap_preds, heatmaps_gt):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_heatmap_preds[: ,i], heatmaps_gt))
        combined_loss = torch.stack(combined_loss, dim=1)
        mean_loss = torch.mean(combined_loss)
        return mean_loss

