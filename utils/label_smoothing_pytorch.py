
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


if __name__ == '__main__':
    torch.manual_seed(15)
    criterion = LabelSmoothingCrossEntropy()
    out = torch.randn(20, 10)
    lbs = torch.randint(10, (20,))
    print('out:', out, out.size())
    print('lbs:', lbs, lbs.size())

    import torch.nn.functional as F
    
    loss = criterion(out, lbs)
    print('loss:', loss)
