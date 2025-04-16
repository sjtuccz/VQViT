import torch
import torch.nn as nn
import torch.nn.functional as F
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T,dim=1):
        super(DistillKL, self).__init__()
        self.T = T
        self.dim=dim

    def forward(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits/self.T, dim=self.dim)
        p_t = F.softmax(t_logits/self.T, dim=self.dim)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / s_logits.shape[0]
        return loss
class DistillCosSim(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self,top_k=None):
        super(DistillCosSim, self).__init__()
        print('using cos similarity loss for logits distillation')
        self.top_k=top_k

    def forward(self, s_logits, t_logits):
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
        self.kl = DistillKL(T=1,dim=-1)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def forward(self, fstudent, fteacher):
        # fstudent = fstudent[-5:]
        # fteacher = fteacher[-5:]
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
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
        for fs, ft in zip(fstudent, fteacher):
            # 确保特征维度一致
            assert fs.shape == ft.shape, f"Shape mismatch: {fs.shape} vs {ft.shape}"
            cos_sim = F.cosine_similarity(fs, ft, dim=-1)  # 逐样本余弦相似度
            loss = (1 - cos_sim).mean()  # 损失 = 1 - 平均相似度
            loss_all += loss
        teacher_cls = fteacher[-1][:, 0, :]  # 取Class Token [B, D]
        student_cls = fstudent[-1][:, 0, :]
        class_token_loss = F.mse_loss(student_cls, teacher_cls) 
        if self.reduction == 'mean':
            # print(class_token_loss)
            return loss_all / len(fstudent) #+ 6.0 *class_token_loss
        elif self.reduction == 'sum':
            return loss_all #+ 2.0 * class_token_loss
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'.")