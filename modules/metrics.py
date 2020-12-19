import torch

__all__ = ['accuracy', 'accuracy_per_cate']

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if batch_size == 0:
            batch_size = 1

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_per_cate(output, target, topk=(1,)):
    cate_num = output.shape[1]
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {k: None for k in topk}
        for k in topk:
            cate_topk = []
            for cate in range(cate_num):
                cate_idx = torch.nonzero(target == cate, as_tuple=False).reshape([-1])
                cate_correct = correct[:, cate_idx]
                correct_k = cate_correct[:k].view(-1).float().sum(0, keepdim=True)
                y = correct_k / cate_idx.size(0) * 100
                if isinstance(y, torch.Tensor):
                    y = y.item()
                cate_topk.append(y)
            res[k] = cate_topk
        return res