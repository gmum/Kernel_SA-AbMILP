import torch
import torch.nn as nn
import torch.nn.functional as F
from models.distributions import log_Bernoulli


class Attention(nn.Module):
    def __init__(self, self_att):
        super(Attention, self).__init__()
        self.self_att = self_att
        self.L = 512
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        if self.self_att:
            self.self_att = SelfAttention(self.L)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        if self.self_att:
            H, self_attention, gamma, gamma_kernel = self.self_att(H)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        log_likelihood = -log_Bernoulli(Y, Y_prob)

        return log_likelihood, A


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter((torch.ones(1)).cuda())
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter((torch.ones(1)).cuda())

    def forward(self, x):
        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x
        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att