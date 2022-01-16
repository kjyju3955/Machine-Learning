import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_averages=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):  # alpha가 숫자로 들어오는 경우 easy, hard negarive일때의 alpha 값을 tensor로 저장
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):  # easy, hard negative 일때의 alpha값이 리스트 형태로 만들어져서 들어온 경우
            self.alpha = torch.Tensor(alpha)
        self.size_averages = size_averages

        def forward(self, input, target):
            if input.dim() > 2:  # input의 디멘젼이 2차원일 경우에 2차원으로 만들어주는 과정 (why? input과 target의 dim을 같게해주려고)
                input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
            target = target.view(-1, 1)

            logpt = F.log_softmax(input)
            logpt = logpt.gather(1, target)  # logpt에서 각 행의 특정 index를 뽑아내는 과정 (이때, target과 logpt의 차원이 같아야 함)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            # alpha가 존재할 경우 처리
            if self.alpha is not None:
                if self.alpha.type() != input.data.type():  # 두 데이터의 데이터 타입이 다를 경우 맞춰줌
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, target.data.view(-1))
                logpt = logpt * Variable(at)

            loss = -1 * (1 - pt) ** self.gamma * logpt

            if self.size_average:  # True일 경우 배치의 각 손실 요소에 대해서 평균화해서 사용, 주로 이걸로 사용
                return loss.mean()
            else:  # False일 경우 배치의 각 손실 요소에 대해서 합산을 사용
                return loss.sum()
