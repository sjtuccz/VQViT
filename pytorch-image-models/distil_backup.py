import torch
import torch.nn as nn
import torch.nn.functional as F
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T,dim=1):
        super(DistillKL, self).__init__()
        self.T = T
        self.dim=dim
        print(f'using KL divergence loss T={T} for logits distillation')

    def forward(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits/self.T, dim=self.dim)
        p_t = F.softmax(t_logits/self.T, dim=self.dim)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / s_logits.shape[0]
        return loss
    

class DistillKLandCosSim(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T,dim=1):
        super(DistillKLandCosSim, self).__init__()
        self.T = T
        self.dim=dim
        print(f'using KL divergence loss T={T} and cos similarity for logits distillation')

    def forward(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits/self.T, dim=self.dim)
        p_t = F.softmax(t_logits/self.T, dim=self.dim)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / s_logits.shape[0]
        loss += (1 - F.cosine_similarity(s_logits, t_logits, dim=-1)).mean()
        return loss
    
class DistillCosSim(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self,top_k=None):
        super(DistillCosSim, self).__init__()
        print('using cos similarity loss for logits distillation')
        self.top_k=top_k

    def forward(self, s_logits, t_logits):
        loss=0
        if isinstance(s_logits, (list, tuple)) or isinstance(t_logits, (list, tuple)):
            for s, t in zip(s_logits, t_logits):
                assert s.size() == t.size(), 'sizes of teacher and student outputs must be the same'
                loss += (1 - F.cosine_similarity(s, t, dim=-1)).mean()
        else:
            assert s_logits.size() == t_logits.size(), 'sizes of teacher and student outputs must be the same'
            if self.top_k is not None:
                topk_idx = t_logits.topk(k=100, dim=-1).indices
                t_topk = t_logits.gather(-1, topk_idx)
                s_topk = s_logits.gather(-1, topk_idx)
                loss = (1 - F.cosine_similarity(s_topk, t_topk, dim=-1)).mean()
            else:
                loss = (1 - F.cosine_similarity(s_logits, t_logits, dim=-1)).mean()
        return loss
class DistillCE(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self,dim=1):
        super(DistillCE, self).__init__()
        self.dim=dim

    def forward(self, s_logits, t_logits):
        soft_labels = F.softmax(t_logits, dim=self.dim)
        loss = F.cross_entropy(s_logits, soft_labels) 
        return loss
    
class FeatureLossL2(nn.Module):
    def __init__(self, reduction='sum'):
        super(FeatureLossL2, self).__init__()
        self.reduction = reduction  # 选择损失的计算方式

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def forward(self, fstudent, fteacher):
        # fstudent = fstudent[1:-4]
        # fteacher = fteacher[1:-4]
        loss_all = 0.0
        index=0
        for fs, ft in zip(fstudent, fteacher):
            if isinstance(fs, (list, tuple)) or isinstance(ft, (list, tuple)):
                # print('using cosine similarity loss for feature 2,4 distillation')
                for s, t in zip(fs, ft):
                    assert s.size() == t.size(), 'sizes of teacher and student outputs must be the same'
                    loss_all += F.mse_loss(s, t, reduction='mean')
            else:
                loss = F.mse_loss(fs, ft, reduction='mean')
                # print(f'index: {index}, loss: {loss}')
                # index+=1
                loss_all += loss
        if self.reduction == 'mean':
            return loss_all / len(fstudent)  # 返回均值
        elif self.reduction == 'sum':
            return loss_all  # 返回总和
        else:
            raise ValueError("Invalid reduction type. Use 'mean' or 'sum'.")

class FeatureLossKL(nn.Module):
    def __init__(self, reduction='mean'):
        super(FeatureLossKL, self).__init__()
        self.reduction = reduction  # 选择损失的计算方式
        self.kl = DistillKL(T=1.0,dim=-1)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def forward(self, fstudent, fteacher):
        # fstudent = fstudent[-5:]
        # fteacher = fteacher[-5:]
        loss_all = 0.0
        for idx, (fs, ft) in enumerate(zip(fstudent, fteacher)):
            if idx==8:
                continue
            # 确保特征维度一致
            if isinstance(fs, (list, tuple)) or isinstance(ft, (list, tuple)):
                for s, t in zip(fs, ft):
                    assert s.size() == t.size(), 'sizes of teacher and student outputs must be the same'
                    loss = self.kl(s, t)
                    loss_all += loss
            else:
                loss = self.kl(fs, ft)
                loss_all += loss
        if self.reduction == 'mean':
            return loss_all / len(fstudent)  # 返回均值
        elif self.reduction == 'sum':
            return loss_all  # 返回总和
        else:
            raise ValueError("Invalid reduction type. Use 'mean' or 'sum'.")
class FeatureLossCosine(nn.Module):
    def __init__(self, reduction='sum'):
        super(FeatureLossCosine, self).__init__()
        self.reduction = reduction  # 'mean' 或 'sum'
        self.choose_feat = [0,2,4,6,8,10]  # 选择要计算余弦相似度的特征层索引
        # self.choose_feat = [3, 5, 7,9,11]  

    def forward(self, fstudent, fteacher):
        """
        计算学生特征与教师特征之间的余弦相似度损失
        Args:
            fstudent: 学生网络的多层特征列表 [Tensor1, Tensor2, ...]
            fteacher: 教师网络的多层特征列表 [Tensor1, Tensor2, ...]
        Returns:
            余弦相似度损失（标量）
        """
        loss_all = 0.0
        for idx, (fs, ft) in enumerate(zip(fstudent, fteacher)):
            # if idx not in self.choose_feat:
            #     continue
            # 确保特征维度一致
            if isinstance(fs, (list, tuple)) or isinstance(ft, (list, tuple)):
                # print('using cosine similarity loss for feature 2,4 distillation')
                for s, t in zip(fs, ft):
                    # print(s.shape)
                    # print(t.shape)
                    assert s.size() == t.size(), 'sizes of teacher and student outputs must be the same'
                    loss_all += (1 - F.cosine_similarity(s, t, dim=-1)).mean()
            else:
                assert fs.shape == ft.shape, f"Shape mismatch: {fs.shape} vs {ft.shape}"
                cos_sim = F.cosine_similarity(fs, ft, dim=-1) 
                loss = (1 - cos_sim).mean()  # 损失 = 1 - 平均相似度
                loss_all += loss
        # teacher_cls = fteacher[-1][:, 0, :]  # 取Class Token [B, D]
        # student_cls = fstudent[-1][:, 0, :]
        # class_token_loss = F.mse_loss(student_cls, teacher_cls) 
        if self.reduction == 'mean':
            # print(class_token_loss)
            return loss_all / len(fstudent) #+ 6.0 *class_token_loss
        elif self.reduction == 'sum':
            return loss_all #+ 2.0 * class_token_loss
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'.")
class FeatureLossCosineKL(nn.Module):
    def __init__(self, reduction='sum'):
        super(FeatureLossCosineKL, self).__init__()
        self.reduction = reduction  # 'mean' 或 'sum'
        self.kl = DistillKL(T=1.0,dim=-1)
    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            # 确保特征维度一致
            if isinstance(fs, (list, tuple)) or isinstance(ft, (list, tuple)):
                # print('using cosine similarity loss for feature 2,4 distillation')
                assert fs[1].size() == ft[1].size(), 'sizes of teacher and student outputs must be the same'
                loss_all += 5*(1 - F.cosine_similarity(fs[1], ft[1], dim=-1)).mean()#对于FFN蒸馏使用余弦相似度
                loss_all += self.kl(fs[0], ft[1]) #对于MHA蒸馏使用KL

        if self.reduction == 'mean':
            # print(class_token_loss)
            return loss_all / len(fstudent) #+ 6.0 *class_token_loss
        elif self.reduction == 'sum':
            return loss_all #+ 2.0 * class_token_loss
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'.")
        

# def _get_gt_mask(logits, target):
#     target = target.reshape(-1).long()
#     mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
#     return mask
def _get_gt_mask(logits, target):
    assert target.dim() == 2, "Target must be 2D (batch, seq_len)"
    mask = (target != 0)
    return mask


# def _get_other_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
#     return mask
def _get_other_mask(logits, target):
    mask = (target == 0)
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
class DKD(nn.Module):
    def __init__(self, T=1.0, alpha=1.0, beta=1.0):
        super(DKD, self).__init__()
        print(f'using DKD divergence loss T={T} for logits distillation')
        self.alpha = alpha
        self.beta = beta
        self.temperature = T
        # self.warmup = cfg.DKD.WARMUP
        # self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        # self.trainer = cfg.SOLVER.TRAINER
        # self.topk = cfg.SOLVER.DEEPKD.TOPK

    def forward(self, logits_student, logits_teacher, target):
        # print(f'target: {target}, shape:{target.shape}')
        if target.dim() == 1:
            target = F.one_hot(target.long(), num_classes=1000).float()
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / self.temperature , dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature , dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (self.temperature **2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature  - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature  - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
            * (self.temperature **2)
            / target.shape[0]
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss