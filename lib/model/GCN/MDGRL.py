import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCN.models import GCN

import numpy as np
np.set_printoptions(threshold=np.inf)

class discreator(nn.Module):
  def __init__(self, nin=128, nhid=256, nlast=512):
    super(discreator, self).__init__()
    self.fc1 = nn.Linear(nin, nhid)
    self.fc2 = nn.Linear(nhid, nlast)
    self.fc3 = nn.Linear(nlast, 21)

  def forward(self, x):
    x = F.dropout(F.relu(self.fc1(x)), training=self.training)
    x = F.dropout(F.relu(self.fc2(x)), training=self.training)
    x = self.fc3(x)
    return x

class PLGRL(nn.Module):
    def __init__(self):
        super(PLGRL, self).__init__()
        self.GCN = GCN(512, 256, 128, 0.1)
        self.downsample = nn.AdaptiveAvgPool2d((20, 20))
        self.criterion = nn.CrossEntropyLoss()
        self.classifier = discreator()

    def normal(self, A, sym=True):
        D = A.sum(1)
        if sym is True:
            D = torch.diag(torch.pow(D, -0.5))
            A = D.mm(A).mm(D)
        else:
            D = torch.diag(torch.pow(D, -1))
            A = D.mm(A)
        return A

    def forward(self, x, im_info, gt_boxes, num_boxes):
        if not self.training or x.size(0) is not 1:
            return torch.Tensor([0.0]).cuda()

        x = self.downsample(x)
        H, W = im_info[0][0], im_info[0][1]
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        dis_y = torch.zeros((h * w)).cuda()
        for i in range(h):
            dis_y[i * h:(i+1) * h] = i
        dis_x = torch.zeros((h * w)).cuda()
        for i in range(h):
            dis_x[i * h:(i+1) * h] = torch.Tensor(range(h)).cuda()
        dis_y = ((dis_y.unsqueeze(-1).repeat(1, h * w)) - (dis_y.unsqueeze(0).repeat(h * w, 1))) ** 2
        dis_x = ((dis_x.unsqueeze(-1).repeat(1, h * w)) - (dis_x.unsqueeze(0).repeat(h * w, 1))) ** 2
        dis = dis_y + dis_x

        isboxes = torch.zeros((h, w)).cuda()
        label = torch.zeros((h, w)).cuda()
        for j in range(num_boxes[0]):
            x1, y1, x2, y2, cl = gt_boxes[0][j][0], gt_boxes[0][j][1], gt_boxes[0][j][2], gt_boxes[0][j][3], gt_boxes[0][j][4]
            x1, y1, x2, y2 = int(x1*w//W), int(y1*h//H), int(x2*w//W), int(y2*h//H)
            isboxes[y1:y2, x1:x2] = 1.0
            label[y1:y2, x1:x2] = cl

        isboxes = isboxes.reshape(h*w)
        isboxes_1 = isboxes.unsqueeze(-1).repeat(1, h*w)
        isboxes_2 = isboxes.unsqueeze(0).repeat(h*w, 1)
        isboxes = isboxes_1 * isboxes_2

        dis = torch.where(dis < 50, torch.full_like(dis, 1.), torch.full_like(dis, 0.))
        dis = dis * isboxes
        del isboxes, isboxes_2, isboxes_1, dis_x, dis_y

        x = x.transpose(1, 3).reshape(b * w * h, c)
        simi_matrix = torch.abs(torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2))
        simi_matrix = simi_matrix - torch.eye(simi_matrix.size(0)).cuda()
        simi_matrix = torch.where(simi_matrix < 0.5, torch.full_like(simi_matrix, 0.), simi_matrix)
        simi_matrix = simi_matrix * dis
        simi_matrix = simi_matrix + torch.eye(simi_matrix.size(0)).cuda()
        simi_matrix = self.normal(simi_matrix)
        simi_matrix = simi_matrix.detach()

        x = self.GCN(x, simi_matrix)
        label = label.reshape(h*w)
        label = label.type(torch.LongTensor).cuda()
        x = self.classifier(x)
        loss = self.criterion(x, label)

        return loss

class ILGRL(nn.Module):
    def __init__(self):
        super(ILGRL, self).__init__()
        self.GCN = GCN(4096, 2048, 1024, 0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.classifier = discreator(nin=1024, nhid=512, nlast=256)

    def normal(self, A, sym=True):
        D = A.sum(1)
        if sym is True:
            D = torch.diag(torch.pow(D, -0.5))
            A = D.mm(A).mm(D)
        else:
            D = torch.diag(torch.pow(D, -1))
            A = D.mm(A)
        return A

    def forward(self, x, label):
        if not self.training:
            return torch.Tensor([0.0]).cuda()

        N, F = x.size(0), x.size(1)

        mask = torch.eq(label.unsqueeze(-1).repeat(1, N), label.unsqueeze(0).repeat(N, 1)).float().cuda()
        simi_matrix = torch.abs(torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2))
        simi_matrix = 1.0 - simi_matrix
        simi_matrix[label==0] = 0.0
        simi_matrix = simi_matrix + torch.eye(simi_matrix.size(0)).cuda()
        simi_matrix = simi_matrix * mask
        simi_matrix = self.normal(simi_matrix)
        simi_matrix = simi_matrix.detach()

        x = self.GCN(x, simi_matrix)
        x = self.classifier(x)
        loss = self.criterion(x, label)

        return loss

class CDGRL(nn.Module):
    def __init__(self):
        super(CDGRL, self).__init__()
        self.GCN = GCN(4096, 2048, 1024, 0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.classifier = discreator(nin=1024, nhid=512, nlast=256)

    def normal(self, A, sym=True):
        D = A.sum(1)
        if sym is True:
            D = torch.diag(torch.pow(D, -0.5))
            A = D.mm(A).mm(D)
        else:
            D = torch.diag(torch.pow(D, -1))
            A = D.mm(A)
        return A

    def weighted(self, x, label):
        x, label = x.cpu(), label.cpu()
        N, K = x.size(0), x.size(1)
        lab = label.unsqueeze(-1)
        onehot = torch.zeros(N, 21).scatter_(1, lab, 1)
        centertensor = torch.matmul(x.t(), onehot) #K x C
        su = torch.zeros(21)
        for i in range(N):
            y = int(label[i])
            su[y] = su[y] + 1.0
        su = torch.clamp(su, min=1.0)
        centertensor = centertensor // (su.unsqueeze(0).repeat(K, 1))
        centertensor = torch.matmul(centertensor, onehot.t()) #KxN
        centertensor = centertensor.t() #NxK
        simi_matrix = torch.abs(torch.cosine_similarity(x, centertensor)) #Nx1
        max_simi = torch.zeros(N)
        for i in range(N):
            max_simi[i] = torch.max(simi_matrix[label==label[i]])
            if max_simi[i] == 0.0:
                max_simi[i] = 1.0
        simi_matrix = simi_matrix / max_simi
        del lab, onehot, centertensor, su, max_simi

        return simi_matrix



    def forward(self, x1, x2, label1, label2):
        if not self.training:
            return torch.Tensor([0.0]).cuda()

        n = 0
        for i in range(label1.size(0)):
            if label1[i].item() is not 0:
                n = n + 1
            else:
                break

        m = 0
        for i in range(label2.size(0)):
            if label2[i].item() is not 0:
                m = m + 1
            else:
                break

        weight1 = self.weighted(x1[0:n, :], label1[0:n]).cuda()
        weight2 = self.weighted(x2[0:m, :], label2[0:m]).cuda()

        x = torch.cat((x1[0:n, :], x2[0:m, :]), dim=0)
        label = torch.cat((label1[0:n], label2[0:m]), dim=0)
        weight = torch.cat((weight1, weight2), dim=0)

        weight = 1.0 - torch.abs(weight.unsqueeze(-1).repeat(1, n+m) - weight.unsqueeze(0).repeat(n+m, 1))
        simi_matrix = torch.abs(torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2))
        simi_matrix = simi_matrix * weight
        simi_matrix[0:n, 0:n] = 0.0
        simi_matrix[n:n+m, n:n+m] = 0.0
        simi_matrix = simi_matrix + torch.eye(n+m).cuda()
        simi_matrix = self.normal(simi_matrix)
        simi_matrix = simi_matrix.detach()

        x = self.GCN(x, simi_matrix)
        x = self.classifier(x)
        loss = self.criterion(x, label)
        del n,m,weight1,weight2,weight,label,simi_matrix

        return loss





