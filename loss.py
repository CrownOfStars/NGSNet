import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class Skin_GTIoULoss(nn.Module):
    def __init__(self, edge_range=3, weight=None):
        super().__init__()

        self.Pool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)

        if weight is None:
            self.weight = torch.ones([2])
        else:
            self.weight = weight

    def edge_decision(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable

        return F.relu(self.Pool(seg_map)-seg_map)

    def forward(self, preds, targets):
        """
        preds: logits pred of sod map 
        targets: ground truth[0 1] of sod map
        """
        
        preds = torch.cat([1-preds,preds],dim=1)
        targets = torch.cat([1-targets,targets],dim=1)

        # edge IoU Loss

        preds = self.edge_decision(preds)#生成边界的内外层
        targets = self.edge_decision(targets)#生成边界的内外层


        intersectoin = (preds * targets).sum(dim=(2, 3))
        union = targets.sum(dim=(2, 3))
        edge_IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        edge_IoU_loss = 1 - edge_IoU_loss
        edge_IoU_loss = self.weight.to(edge_IoU_loss)[None] * edge_IoU_loss

        return edge_IoU_loss.mean()



class Skin_EdgeIoULoss(nn.Module):
    def __init__(self, edge_range=3, weight=None):
        super().__init__()

        self.Pool = nn.MaxPool2d(2 * edge_range + 1, stride=1, padding=edge_range)

        if weight is None:
            self.weight = torch.ones([2])
        else:
            self.weight = weight

    def edge_decision(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable

        return self.Pool(seg_map)


    def forward(self, preds, targets):
        """
        preds: logits pred of sod map 
        targets: ground truth[0 1] of sod map
        """
        
        # edge IoU Loss

        preds = self.edge_decision(preds)#生成边界的内外层
        targets = self.edge_decision(preds)#生成边界的内外层


        intersectoin = (preds * targets).sum(dim=(2, 3))
        union = targets.sum(dim=(2, 3))
        edge_IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        edge_IoU_loss = 1 - edge_IoU_loss
        edge_IoU_loss = self.weight.to(edge_IoU_loss)[None] * edge_IoU_loss

        return edge_IoU_loss.mean()    




def IoU(pred,target):
    return sum([torch.sum(t*p)/(torch.sum(t)+torch.sum(p)-torch.sum(t*p)) for p,t in zip(pred,target)])

def IoUs(pred,target):
    return torch.stack(([torch.sum(t*p)/(torch.sum(t)+torch.sum(p)-torch.sum(t*p)) for p,t in zip(pred,target)]))

def iou_loss(pred, target):
    # 计算交集
    intersection = (pred * target).float().sum((2, 3))  
    # 计算并集
    union = (pred + target).float().sum((2, 3))/2
    # 计算 IoU
    iou = (intersection + 1e-16) / (union + 1e-16)
    # 计算 IoU Loss
    iou_loss = 1 - iou.mean()
    return iou_loss

class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)#if soft
        return iou_loss(pred, target)


CELoss = torch.nn.BCELoss(reduction='mean')
CEWLoss = nn.BCEWithLogitsLoss()

IOULoss = IOU()

skin_GTIOULoss = Skin_GTIoULoss(5,weight=torch.tensor([0.25,5]))
skin_EdgeIoULoss = Skin_EdgeIoULoss(3,weight=torch.tensor([0.5,5]))



def one_hot_changer(tensor, vector_dim, dim=-1, bool_=False):
    """index tensorをone hot vectorに変換する関数

    Args:
        tensor (torch.tensor,dtype=torch.long): index tensor
        vector_dim (int): one hot vectorの次元。index tensorの最大値以上の値でなくてはならない
        dim (int, optional): one hot vectorをどこの次元に組み込むか. Defaults to -1.
        bool_ (bool, optional): Trueにするとbool型になる。Falseの場合はtorch.float型. Defaults to False.

    Raises:
        TypeError: index tensor is not torch.long
        ValueError: index tensor is greater than vector_dim

    Returns:
        torch.tensor: one hot vector
    """
    if bool_:
        data_type = bool
    else:
        data_type = torch.float

    if tensor.dtype != torch.long:
        raise TypeError("入力テンソルがtorch.long型ではありません")
    if tensor.max() >= vector_dim:
        raise ValueError(f"入力テンソルのindex番号がvector_dimより大きくなっています\ntensor.max():{tensor.max()}")

    # one hot vector用単位行列
    one_hot = torch.eye(vector_dim, dtype=data_type, device=tensor.device)
    vector = one_hot[tensor]

    # one hot vectorの次元変更
    dim_change_list = list(range(tensor.dim()))
    # もし-1ならそのまま出力
    if dim == -1:
        return vector
    # もしdimがマイナスならスライス表記と同じ性質にする
    if dim < 0:
        dim += 1  # omsertは-1が最後から一つ手前

    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector


class Edge_IoULoss(nn.Module):
    def __init__(self, n_class, edge_range=3, lamda=1.0, weight=None):
        super().__init__()
        self.n_class = n_class
        
        self.maxPool = nn.MaxPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)

        if weight is None:
            self.weight = torch.ones([self.n_class])
        else:
            self.weight = weight
        self.lamda = lamda

    def edge_decision(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        return object_edge_inside_flag

    def forward(self, preds, targets):
        """
        preds: logits pred of sod map 
        targets: ground truth[0 1] of sod map
        """
        
        preds = torch.cat([1-preds,preds],dim=1)
        targets = torch.cat([1-targets,targets],dim=1)


        intersectoin = (preds * targets).sum(dim=(2, 3))
        
        total = (preds + targets).sum(dim=(2, 3))
        union = total - intersectoin
        IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)

        IoU_loss = 1 - IoU_loss
        IoU_loss = self.weight.to(IoU_loss)[None] * IoU_loss

        # edge IoU Loss
        predicts_idx = preds.argmax(dim=1)
        predicts_seg_map = one_hot_changer(predicts_idx, self.n_class, dim=1)
        predict_edge = self.edge_decision(predicts_seg_map)#生成边界的内外层
        targets_edge = self.edge_decision(targets)#生成边界的内外层

        preds = preds * predict_edge
        targets = targets * targets_edge

        intersectoin = (preds * targets).sum(dim=(2, 3))
        union = targets.sum(dim=(2, 3))
        edge_IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        edge_IoU_loss = 1 - edge_IoU_loss
        edge_IoU_loss = self.weight.to(edge_IoU_loss)[None] * edge_IoU_loss

        return IoU_loss.mean() + edge_IoU_loss.mean() * self.lamda

def avg_max_pool(bounds,thickness):
    return F.avg_pool2d(F.max_pool2d(bounds,kernel_size=2*thickness+1,stride=1,padding=thickness),kernel_size=2*thickness+1,stride=1,padding=thickness)

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

CEWLoss = nn.BCEWithLogitsLoss()

def GTSupervision(s1,s2,s3,s4,gt):
    loss = 0.0
    for s in [s1,s2,s3,s4]:
       
        loss += structure_loss(s,gt)#CEWLoss(s, gt) + IOULoss(s, gt)
                                                   
        gt = F.interpolate(gt,scale_factor=0.5,recompute_scale_factor=True)
    return loss

def EdgeSupervision(edge1,bounds):
    return IOULoss(edge1,bounds)#


def NAMLABSupervision(edge2,texs):
    return CEWLoss(edge2,texs) + IOULoss(edge2,texs)

import torch.nn as nn


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1., gamma=1.):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss

if __name__ == "__main__":

    KD_loss = DIST()

    print(KD_loss(torch.randn(4,128,56,56),torch.randn(4,128,56,56)))
    #import cv2
    #import numpy as np
    # eloss = Skin_GTIoULoss(3)
    # gt = torch.tensor([[cv2.imread("./fork/GT.png",0)/255]],dtype=torch.float32)
    # pred = torch.tensor([[cv2.imread("./fork/s1.png",0)/255]],dtype=torch.float32)
    # #y = mask_to_boundary(x,0.01)
    # #cv2.imwrite("eedge.png",y)
    # pred = F.interpolate(pred,size=224)


    # loss = eloss(pred,gt)
    
    # print(loss)
    
    # pred = torch.cat([pred,1-pred],dim=1)
    # gt = torch.cat([gt,1-gt],dim=1)
    

    # gt = eloss.edge_decision(gt)
    # pred = eloss.edge_decision(pred)
    
    # #y = x*y
    # #print(y.shape,y.unique())

    # cv2.imwrite("./smoo_0.png",(gt.numpy()[0][0]*255).astype(np.uint8))
    # cv2.imwrite("./smoo_1.png",(gt.numpy()[0][1]*255).astype(np.uint8))
    # cv2.imwrite("./ep_0.png",(pred.numpy()[0][0]*255).astype(np.uint8))
    # cv2.imwrite("./ep_1.png",(pred.numpy()[0][1]*255).astype(np.uint8))
