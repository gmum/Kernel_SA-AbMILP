import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import log_Bernoulli


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.L = self.args.L
        self.D = self.args.D
        self.K = self.args.K

        first_conv = 5 if args.out_loc else 3

        if self.args.loc_info:
            self.add = 2
        else:
            self.add = 0

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(first_conv, 36, 4),
            self.args.activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(36, 48, 3),
            self.args.activation,
            nn.MaxPool2d(2, 2),
        )

        torch.nn.init.xavier_uniform_(self.feature_extractor[0].weight)
        self.feature_extractor[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.feature_extractor[3].weight)
        self.feature_extractor[3].bias.data.zero_()

        self.fc = nn.Sequential(
            nn.Linear(48 * 5 * 5, self.L),
            self.args.activation,
            nn.Dropout(0.2),
            nn.Linear(self.L, self.L),
            self.args.activation,
            nn.Dropout(0.2),
        )

        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        self.fc[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc[3].weight)
        self.fc[3].bias.data.zero_()

        if self.args.self_att:
            self.self_att = SelfAttention(self.L, self.args)

        self.attention = nn.Sequential(  # first layer
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            # second layer
            nn.Linear(self.D, self.K)
            # outputs A: NxK
        )

        torch.nn.init.xavier_uniform_(self.attention[0].weight)
        self.attention[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.attention[2].weight)
        self.attention[2].bias.data.zero_()

        self.classifier = nn.Sequential(
            nn.Linear((self.L) * self.K, 1),
            nn.Sigmoid()
        )

        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.zero_()

    def forward(self, x):
        # Trash first dimension
        x = x.squeeze(0)
        if not self.args.out_loc:
            loc = x[:, 3:]
            x = x[:, :3]

        # Extract features
        H = self.feature_extractor(x)  # NxL
        H = H.view(-1, 48 * 5 * 5)
        
        H = self.fc(H)

        if self.args.loc_info:
            pos_x = loc[:, 0, 0, 0].view(-1, 1)
            pos_y = loc[:, 1, 0, 0].view(-1, 1)
            H = torch.cat((H, pos_x, pos_y), dim=1)
        
        gamma, gamma_kernel = (0, 0)
        if self.args.self_att:
            H, self_attention, gamma, gamma_kernel = self.self_att(H)

        # attention
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        z = F.softmax(A)  # softmax over N

        M = torch.mm(z, H)  # KxL

        M = M.view(1, -1)  # (K*L)x1

        # classification
        y_prob = self.classifier(M)

        y_hat = torch.ge(y_prob, self.args.classification_threshold).float()

        if self.args.self_att:
            return y_prob, y_hat, z, (A, self_attention), gamma, gamma_kernel
        else:
            return y_prob, y_hat, z, A, gamma, gamma_kernel

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, y_hat, _, _, gamma, gamma_kernel = self.forward(X)
        error = 1. - y_hat.eq(Y).cpu().float().mean()
        return error, gamma, gamma_kernel

    def calculate_objective(self, X, Y):
        Y = Y.float()
        y_prob, _, _, _, gamma, gamma_kernel = self.forward(X)
        log_likelihood = -log_Bernoulli(Y, y_prob)
        return log_likelihood, gamma, gamma_kernel


class SelfAttention(nn.Module):
    def __init__(self, in_dim, args):
        super(SelfAttention, self).__init__()
        self.args = args
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter((torch.ones(1)).cuda())
        self.gamma_in = nn.Parameter((torch.ones(1)).cuda())
        self.softmax = nn.Softmax(dim=-1)
        self.alfa = nn.Parameter((torch.ones(1)).cuda())
        self.gamma_att = nn.Parameter((torch.ones(1)).cuda())

    def forward(self, x):
        if self.args.loc_info:
            loc = x[:, -2:]
            x = x[:, :-2]

        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        if self.args.att_gauss_spatial:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(length):
                gauss = torch.pow(proj_query - proj_key[:, :, i].t(), 2).sum(dim=1)
                proj[:, i] = torch.exp(-F.relu(self.gamma_att) * gauss)
            energy = proj.view((1, length, length))
        elif self.args.att_inv_q_spatial:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(length):
                gauss = torch.pow(proj_query - proj_key[:, :, i].t(), 2).sum(dim=1)
                proj[:, i] = 1 / (F.relu(self.gamma_att) * gauss + torch.ones(1).cuda())
            energy = proj.view((1, length, length))
        elif self.args.att_module:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(length):
                proj[:, i] = (torch.abs(proj_query - proj_key[:, :, i].t()) -
                         torch.abs(proj_query) -
                         torch.abs(proj_key[:, :, i].t())).sum(dim=1)
            energy = proj.view((1, length, length))

        elif self.args.att_gauss_abnormal:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(int(C//8)):
                gauss = proj_query[0, i, :] - proj_key[0, i, :].view(-1, 1)
                proj += torch.exp(-F.relu(self.gamma_att) * torch.abs(torch.pow(gauss, 2)))
            energy = proj.view((1, length, length))

        elif self.args.att_inv_q_abnormal:
            proj = torch.zeros((length, length)).cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(int(C//8)):
                gauss = proj_query[0, i, :] - proj_key[0, i, :].view(-1, 1)
                proj += torch.exp(F.relu(1 / (torch.pow(gauss, 2) + torch.tensor(1).cuda())))
            energy = proj.view((1, length, length))

        else:
            energy = torch.bmm(proj_query, proj_key)  # transpose check

        if self.args.loc_info:
            if self.args.loc_gauss:
                loc_energy_x = torch.exp(-F.relu(self.gamma_in) * torch.abs(torch.pow(loc[:, 0] - loc[:, 0].view(-1, 1), 2)))
                loc_energy_y = torch.exp(-F.relu(self.gamma_in) * torch.abs(torch.pow(loc[:, 1] - loc[:, 1].view(-1, 1), 2)))
                energy_pos = self.alfa * (loc_energy_x + loc_energy_y)
                energy = energy + energy_pos
            elif self.args.loc_inv_q:
                loc_energy_x = torch.exp(1 / (torch.abs(torch.pow(loc[:, 0] - loc[:, 0].view(-1, 1), 2) + torch.tensor(1).cuda())))
                loc_energy_y = torch.exp(1 / (torch.abs(torch.pow(loc[:, 1] - loc[:, 1].view(-1, 1), 2) + torch.tensor(1).cuda())))
                energy_pos = self.alfa * loc_energy_x + loc_energy_y
                energy = energy + energy_pos

            elif self.args.loc_att:
                loc_proj = torch.zeros((length, length))
                if self.args.cuda:
                    loc_proj = loc_proj.cuda()
                #proj_query = proj_query.permute(0, 2, 1)
                rel_loc_x = loc[:, 0] - loc[:, 0].view(-1, 1)
                rel_loc_y = loc[:, 1] - loc[:, 1].view(-1, 1)
                for i in range(length):
                    rel_loc_at = torch.sum(proj_query[0] * rel_loc_x[:, i].view(-1) * rel_loc_y[i, :].view(-1), dim=0)
                    loc_proj[:, i] = rel_loc_at
                energy += loc_proj.view((1, length, length))
        
        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x
        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att
