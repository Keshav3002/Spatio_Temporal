import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import BASE_DATA_DIR
from lib.models.spin import Regressor

from torch.autograd import Variable   ## 

class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)
        # breakpoint()

        return scores
    
class Temporal_Feature_Aggregation(nn.Module):
    def __init__(self):
        super(Temporal_Feature_Aggregation, self).__init__()
        self.conv_H3 = nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_H4 = nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_H5 = nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_W1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_W2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1,padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')
        self.flatten = nn.Flatten()
        self.linear_T1 = nn.Linear(in_features=154, out_features=512)
        self.linear_T2 = nn.Linear(in_features=3, out_features=512)
        self.linear_T3 = nn.Linear(in_features=112*112*16, out_features=2048)
        self.relu = nn.ReLU()
        self.conv_phi6 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_phi7 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1,padding=0, bias=False)


    def forward(self, H, bp, w, is_train=False):
        # breakpoint()
        # B x T x F
        # B: Batch Size
        # T: Window Size
        # F: Feature dimensions
        # H = self.flatten(H)                             # 32*112*112*16 ---> 32 x 200704
        H = self.linear_T3(H)                           # 32 x 200704 ----> 32 x 16 x 2048
        H = self.relu(H)
        
        bp = self.linear_T1(bp)                         #3 ---> 16 x 512
        bp = self.relu(bp)

        w = self.linear_T2(w)                           #3 ---> 16 x 512
        w = self.relu(w)
        H_ = H.permute(0,2,1)
        
        NSSM_h = torch.matmul(H, H_)
        NSSM_h = self.softmax(NSSM_h)                   # 16 x 16
        
        bp_ = bp.permute(0,2,1)
        NSSM_bp = torch.matmul(bp, bp_)
        NSSM_bp = self.softmax(NSSM_bp)                 # 16 x 16

        w_ = w.permute(0,2,1)
        NSSM_w = torch.matmul(w, w_)
        NSSM_w = self.softmax(NSSM_w)                   # 16 x 16
        # breakpoint()
        H_phi3 = self.conv_H3(H_)                        #2048 ---> n x 16 x 1024
        H_phi4 = self.conv_H4(H_)                        #2048 ---> n x 16 x 1024

        H_phi3_ = H_phi3.permute(0,2,1)
        mul_phi3_4 = torch.matmul(H_phi3_, H_phi4)
        
        # AM_H = self.attention(mul_phi3_4)               #16 x 16
        AM_H = mul_phi3_4

        w_phi1 = self.conv_W1(w_)                        #16 x 512---> 16 x 512
        w_phi2 = self.conv_W2(w_)                        #16 x 512---> 16 x 512

        w_phi1_ = w_phi1.permute(0,2,1)
        mul_phi1_2 = torch.matmul(w_phi1_, w_phi2)

        # AM_W = self.attention(mul_phi1_2)               # 16 x 16
        AM_W = mul_phi1_2
        # breakpoint()
        # concatenated_AM_NSSM = torch.stack((NSSM_h, NSSM_bp, NSSM_w, AM_W, AM_H), dim=2)
        concatenated_AM_NSSM = torch.cat((NSSM_h, NSSM_bp, NSSM_w, AM_W, AM_H),dim=0).view(NSSM_h.size(0),-1,NSSM_h.size(1),NSSM_h.size(2)) 

        
        # concatenated_AM_NSSM = concatenated_AM_NSSM.permute(0, 2, 1)  # (16, 5, 16)

        phi6 = self.conv_phi6(concatenated_AM_NSSM)     # B x 1 x T x T (B=32; T=16)
        phi6 = torch.squeeze(phi6)
        H = H.permute(0,2,1)
        H_phi5 = self.conv_H5(H)

        H_phi5_ = H_phi5.permute(0,2,1)
        mul_phi5_6 = torch.matmul(phi6, H_phi5_)       # B X T x 1024 (B=32; T=16)
        
        mul_phi5_6 = mul_phi5_6.permute(0,2,1)         # B x 1024 X T (B=32; T=16)

        phi7 = self.conv_phi7(mul_phi5_6)              # B x 2048 x T (B=32; T=16)

        Z = H + phi7                                    # 16 x 2048

        return Z, torch.randn((Z.shape[0], 3)), Z


class sta(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(sta, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.nonlocalblock = Temporal_Feature_Aggregation()	#nonlocalblock --> real name: Spatio-Temporal Feature Aggregation

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    # def forward(self, input1=torch.randn(112, 112, 16),input2= torch.randn(3,16),input3= torch.randn(3,16), is_train=False, J_regressor=None):     
    def forward(self, input1,input2,input3, is_train=False, J_regressor=None):                     
                
        # input size NTF
        # input1 = torch.randn(112, 112, 16)
        # input2 = torch.randn(3,16)
        # input3 = torch.randn(3,16)
        batch_size, seqlen = input1.shape[:2]
        print("\n")
        print(batch_size,seqlen)
        feature, scores, feature_seqlen = self.nonlocalblock(input1,input2,input3, is_train=is_train)
        # feature = Variable(feature.reshape(-1, feature.size(-1)))                    
        # feature_seqlen = Variable(feature_seqlen.reshape(-1, feature_seqlen.size(1)))    # 

        # smpl_output = self.regressor(feature, is_train=is_train, J_regressor=J_regressor)
        # smpl_output_Dm = self.regressor(feature_seqlen, is_train=is_train, J_regressor=J_regressor)     #

        # if not is_train:
        #     for s in smpl_output:
        #         s['theta'] = s['theta'].reshape(batch_size, -1)
        #         s['verts'] = s['verts'].reshape(batch_size, -1, 3)
        #         s['kp_2d'] = s['kp_2d'].reshape(batch_size, -1, 2)
        #         s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3)
        #         s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3)
        #         s['scores'] = scores

        # else:
        #     repeat_num = 3  
        #     for s in smpl_output:                
        #         s['theta'] = s['theta'].reshape(batch_size, repeat_num, -1)               
        #         s['verts'] = s['verts'].reshape(batch_size, repeat_num, -1, 3)    
        #         s['kp_2d'] = s['kp_2d'].reshape(batch_size, repeat_num, -1, 2)   
        #         s['kp_3d'] = s['kp_3d'].reshape(batch_size, repeat_num, -1, 3)   
        #         s['rotmat'] = s['rotmat'].reshape(batch_size, repeat_num, -1, 3, 3) 
        #         s['scores'] = scores

        #     for s_Dm in smpl_output_Dm:                 #
        #         s_Dm['theta_forDM'] = s_Dm['theta'].reshape(batch_size, seqlen, -1)   

        # return smpl_output, scores, smpl_output_Dm      #

        feature_seqlen = Variable(feature_seqlen.reshape(-1, feature_seqlen.size(1)))    # 
        smpl_output_Dm = self.regressor(feature_seqlen, is_train=is_train, J_regressor=J_regressor)     #
        for s_Dm in smpl_output_Dm:                 #
            s_Dm['theta_forDM'] = s_Dm['theta'].reshape(batch_size, seqlen, -1)   
        return smpl_output_Dm, scores, smpl_output_Dm      #
