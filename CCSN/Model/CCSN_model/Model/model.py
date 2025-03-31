import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=32):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1, padding=0)
        #self.fc1 = nn.Linear(576, 256)
        #self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(216, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.tanh(self.fc3(x))#serelu
        #x = F.relu(self.fc2(x))

        #x = self.fc3(x)

        return x

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
        self.shift = nn.Parameter(torch.log(torch.tensor([2.])), requires_grad=False)

    def forward(self, x):
        return self.sp(x) - self.shift


class ClusterCalculator(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len,state_fea_len,middle_size=64):
        super(ClusterCalculator, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.state_fea_len =  state_fea_len
        self.embedding_len=32
        self.middle_size = middle_size

        embed_size_e = (atom_fea_len*2+nbr_fea_len+state_fea_len+state_fea_len)
        embed_size_v = atom_fea_len + nbr_fea_len
        embed_size_u = state_fea_len


        self.MLP_e = nn.Sequential(     
            nn.Linear( embed_size_e, middle_size),
            nn.Tanh(),
            nn.Linear(middle_size, nbr_fea_len*2),
            #ShiftedSoftplus(),
            #nn.Tanh(),

        )

        self.MLP_v = nn.Sequential(
            nn.Linear(embed_size_v, middle_size),
            nn.Tanh(),
            nn.Linear(middle_size, atom_fea_len*2),
            #ShiftedSoftplus(),
            nn.Tanh(),
        )

        self.MLP_u = nn.Sequential(
            nn.Linear(embed_size_u, state_fea_len),
            #nn.Tanh(),
            #nn.Linear(middle_size, state_fea_len),
            #ShiftedSoftplus(),
            #nn.Tanh(),
        )
        '''
        self.MLP_e = nn.Sequential(
            nn.Linear(embed_size_e, nbr_fea_len * 2),
            #nn.Tanh(),

        )

        self.MLP_v = nn.Sequential(
            nn.Linear(embed_size_v, atom_fea_len * 2),
            #nn.Tanh(),

        )

        self.MLP_u = nn.Sequential(
            nn.Linear(embed_size_u, state_fea_len),
            #nn.Tanh(),

        )
        '''





        self.bn1 = nn.BatchNorm1d(2 * self.nbr_fea_len)
        self.bn2 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn3 = nn.BatchNorm1d(self.state_fea_len)
        #self.fc1 = nn.Linear(self.nbr_fea_len,self.nbr_fea_len)
        #self.sigmoid = nn.Sigmoid()



    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx,state_fea, crystal_atom_idx):

        ori_atom_fea=atom_in_fea
        ori_nbr_fea = nbr_fea
        ori_state_fea = state_fea
        N, M = nbr_fea_idx.shape



        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        state_nbr_fea = state_fea[nbr_fea_idx, :]

        state_fea_expand2 = state_fea.unsqueeze(1).expand(N, M, self.state_fea_len)#(total_atom_num,12,128)
        total_edge_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),state_fea_expand2,
              nbr_fea,atom_nbr_fea,state_nbr_fea], dim=2)
        total_edge_fea_out1 = self.MLP_e(total_edge_fea)
        N, M ,t= total_edge_fea_out1.shape
        total_edge_fea_out1 = self.bn1(total_edge_fea_out1.view(
            -1, self.nbr_fea_len*2)).view(N, M, self.nbr_fea_len*2)
        edge_filter,edge_core = total_edge_fea_out1.chunk(2,dim=2)
        total_edge_fea_out = edge_filter*edge_core+ori_nbr_fea



        total_edge_mean = torch.mean(total_edge_fea_out,dim=1)
        total_atom_fea = torch.cat(
            [total_edge_mean,
             atom_in_fea], dim=1)
        total_atom_fea_out1 = self.bn2(self.MLP_v(total_atom_fea))
        atom_filter, atom_core = total_atom_fea_out1.chunk(2, dim=1)
        total_atom_fea_out = atom_filter*atom_core+ori_atom_fea


        total_state_fea = torch.cat(
            [state_fea], dim=1)
        total_state_fea_out = self.bn3(self.MLP_u(total_state_fea))+ori_state_fea



        return  total_edge_fea_out, total_atom_fea_out,total_state_fea_out


    def pooling(self, atom_fea, crystal_atom_idx):

        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)




'''###------------------------------------------------------------------------------------------------------------###'''


class ClusterModel(nn.Module):
    def __init__(self, orig_atom_fea_len, orig_nbr_fea_len,
                 atom_fea_len=32, n_block=3, h_fea_len=128,orig_state_fea_len=41,state_fea_len=32,surface_fea_len=12,
                 nbr_fea_len=32):


        super(ClusterModel, self).__init__()
        self.embedding_atom = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embedding_state = nn.Linear(orig_state_fea_len, state_fea_len)
        self.embedding_nbr = nn.Linear(orig_nbr_fea_len, nbr_fea_len)
        self.ClusterNet = nn.ModuleList([ClusterCalculator(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len,state_fea_len=state_fea_len)
                                    for _ in range(n_block)])

        self.Fc1 = nn.Linear(atom_fea_len+nbr_fea_len+state_fea_len+surface_fea_len, h_fea_len)
        self.tanh = nn.Tanh()



        self.fc_out = nn.Linear(h_fea_len, 1)
        self.Surface_conv = LeNet(surface_fea_len)


    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, state_fea, surface_fea,crystal_atom_idx):



        atom_fea = self.embedding_atom(atom_fea)
        state_fea = self.embedding_state(state_fea)
        nbr_fea = self.embedding_nbr(nbr_fea)
        for block in self.ClusterNet:
            nbr_fea,atom_fea,state_fea = block( atom_fea, nbr_fea, nbr_fea_idx,state_fea, crystal_atom_idx)

        nbr_fea= torch.sum(nbr_fea,dim=1)
        crys_fea = torch.cat([nbr_fea, atom_fea, state_fea], dim=1)
        crys_fea = self.pooling( crys_fea ,crystal_atom_idx)

        surface_fea = self.Surface_conv(surface_fea)
        crys_fea = torch.cat([crys_fea, surface_fea], dim=1)

        crys_fea = self.Fc1(crys_fea)
        crys_fea = self.tanh(crys_fea)



        out = self.fc_out(crys_fea)

        return out


    def pooling(self, atom_fea, crystal_atom_idx):

        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


'''###------------------------------------------------------------------------------------------------------------###'''