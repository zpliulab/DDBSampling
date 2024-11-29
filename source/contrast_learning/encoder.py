from torch import nn
import torch
class PPI_encoder(nn.Module):
    def __init__(self, pro_channels=343, lnc_channels=256, out_channels=100):  # imagenet数量
        super().__init__()
        self.layer_pro = nn.Sequential(nn.Linear(in_features=pro_channels, out_features=220),
                                       nn.Sigmoid(),
                                       nn.Linear(in_features=220, out_features=150),
                                       nn.Sigmoid(),
                                       nn.Linear(in_features=150, out_features=out_channels))

        self.act = nn.Softmax(dim=1)


    def embedding(self, feature, mol=1):
        if mol ==1:
            embed_fea  = self.layer_pro(feature)
        if mol ==2:
            embed_fea = self.layer_lnc(feature)
        return embed_fea

class RPI_encoder(nn.Module):
    def __init__(self, pro_channels=343, lnc_channels=256, out_channels=100):  # imagenet数量
        super().__init__()
        self.layer_pro = nn.Sequential(nn.Linear(in_features=pro_channels, out_features=220),
                                       nn.Sigmoid(),
                                       nn.Linear(in_features=220, out_features=150),
                                       nn.Sigmoid(),
                                       nn.Linear(in_features=150, out_features=out_channels))
        self.layer_pro = nn.Sequential(nn.Linear(in_features=lnc_channels, out_features=220),
                                       nn.Sigmoid(),
                                       nn.Linear(in_features=220, out_features=150),
                                       nn.Sigmoid(),
                                       nn.Linear(in_features=150, out_features=out_channels))

        self.act = nn.Softmax(dim=1)

    def forward(self, feature_batch):
        # feature_batch = torch.stack(feature_batch, dim=0).cuda()
        embedding = self.layer_pro(feature_batch)
        source = embedding[:,0,:]
        target = embedding[:,1:,:]
        score = torch.bmm(target, source[:,:,None]).squeeze()
        return score
        output_all = []
        for feature in feature_batch:
            data = feature[0]
            data_type = feature[1]
            out_feature = []
            for index, item in enumerate(data_type):
                source = data[index].cuda()
                if item==0:
                    out_feature.append(self.layer_lnc(source))
                else:
                    out_feature.append(self.layer_pro(source))
            input = out_feature[0]
            output = torch.cat(out_feature[1:], dim=0)
            output_proba = torch.mm(input, output.T)
            # output_proba = self.act(output_proba)
            output_all.append(output_proba)
        output_proba = torch.cat(output_all,dim=0)
        return output_proba

    def embedding(self, feature, mol=1):
        if mol == 1:
            embed_fea  = self.layer_lnc(feature)
        if mol == 2:
            embed_fea = self.layer_pro(feature)
        return embed_fea
