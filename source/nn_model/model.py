import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm
import numpy as np
from .modules import TransformerLayer
from torch_geometric.data import Batch
from torch_geometric.nn.conv import GCNConv
from torch_scatter import scatter_mean
import torch.nn.functional as F
class cnn_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pooling_size):
        super(cnn_block, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=(kernel_size,),padding='same')
        # self.batch = nn.LayerNorm(out_channel)
        self.act = nn.ReLU(inplace=False)
        self.pool = nn.AvgPool1d(kernel_size=pooling_size)

    def forward(self,x):
        x = self.cnn(x)
        # x = self.batch(x)
        x = self.act(x)
        # x = self.pool(x)
        return x

class seqCNN(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(seqCNN, self).__init__()
        self.pooling_size = [1,1,1,1]
        dims = [in_channel, 64, 64, 128]
        kernel_size = [5,5,5]
        pool_size = [1,1,1]
        self.cnn = nn.ModuleList(cnn_block(in_channel=dims[i], out_channel=dims[i+1], kernel_size=kernel_size[i],pooling_size=pool_size[i]) for i in range(3))
    def forward(self, x, mask):
        max_length = x.shape[-1]
        for cnn in self.cnn:
            x = cnn(x)
        # x = x * (~mask).unsqueeze(1)
        # x = self.gp(x).squeeze(-1)
        # x = x * max_length / (~mask).sum(1).unsqueeze(-1)
        # x = self.batch(x)
        return x

class seq_readout(nn.Module):

    def __init__(self,out_channel):
        super(seq_readout, self).__init__()
        self.batch = nn.LayerNorm(out_channel)
        self.gp = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, mask):
        max_length = x.shape[-1]
        x = x * (~mask).unsqueeze(1)
        x = self.gp(x).squeeze(-1)
        x = x * max_length / (~mask).sum(1).unsqueeze(-1)
        x = self.batch(x)
        return x


class gcn_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(gcn_block, self).__init__()
        self.cnn = GCNConv(in_channels=in_channel, out_channels=out_channel)
        # self.batch=nn.LayerNorm(out_channel)
        self.act = nn.ReLU(inplace=False)

    def forward(self,x, edge_index):
        x = self.cnn(x, edge_index)
        # x = self.batch(x)
        x = self.act(x)
        return x

class graphCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(graphCNN, self).__init__()
        dims = [in_channel, out_channel, out_channel, out_channel, out_channel]
        self.cnn = nn.ModuleList(gcn_block(in_channel=dims[i], out_channel=dims[i+1]) for i in range(3))
        self.gp = nn.AdaptiveAvgPool1d(1)
    def forward(self, x, edge_index):
        for cnn in self.cnn:
            x = cnn(x, edge_index)
        return x

class graph_readout(nn.Module):
    def __init__(self, out_channel):
        super(graph_readout, self).__init__()
        self.batch = nn.LayerNorm(out_channel)

    def forward(self, x, batch):
        x = scatter_mean(x, batch, dim=0)
        return self.batch(x)
class esm(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(esm, self).__init__()
        self.cnn1 = TransformerLayer(embed_dim=in_channel,ffn_embed_dim=out_channel, attention_heads=1,)
        self.cnn2 = TransformerLayer(embed_dim=out_channel,ffn_embed_dim=out_channel, attention_heads=1,)
        self.cnn3 = TransformerLayer(embed_dim=out_channel,ffn_embed_dim=out_channel, attention_heads=1,)
        self.cnn4 = TransformerLayer(embed_dim=out_channel,ffn_embed_dim=out_channel, attention_heads=1,)

        # self.gru1 = nn.LSTM(out_channel, out_channel, 2)
        self.avepooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(2,1)
        x = self.cnn1(x)
        x = self.cnn2(x[0])
        x = self.cnn3(x[0])
        x = self.cnn4(x[0])
        x = x[0].transpose(2,1)
        # x = self.gru1(x)

        x = self.avepooling(x)
        x = x.squeeze()
        return x

class EarlyStopping:
    def __init__(self, patience_stop=10, patience_lr=5, verbose=False, delta=0.001, path='check1.pth'):
        self.stop_patience = patience_stop
        self.lr_patience = patience_lr
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, auc, model):
        if self.best_score is None:
            self.best_score = auc
            self.save_checkpoint(model)
            print('saving best model...')
        elif auc <= self.best_score+self.delta:
            self.counter += 1
            if self.counter == self.lr_patience:
                self.adjust_lr(model)
                # self.counter = 0

            if self.counter >= self.stop_patience:
                self.early_stop = True
        else:
            self.best_score = auc
            self.save_checkpoint(model)
            self.counter = 0
            print('saving best model...')

    def adjust_lr(self, model):
        lr = model.optimizer.param_groups[0]['lr']
        lr = lr/10
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = lr
        model.load_state_dict(torch.load(self.path))
        print('loading best model, changing learning rate to %.7f' % lr)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        # Additional geometric features: mean and Gauss curvatures computed at different scales.
        self.args = args
        out_dim = 128
        if args.task == 'PPI':
            self.embedding1 = nn.Embedding(22, 64)
            self.net1 = seqCNN(in_channel=64, out_channel=out_dim)
            self.embedding2 = self.embedding1
            self.net2 = self.net1
            self.net1_readout = seq_readout(out_channel=out_dim)
            self.net2_readout = self.net1_readout

        elif args.task == 'RPI':
            self.embedding1 = nn.Embedding(6, 64)
            self.net1 = seqCNN(in_channel=64, out_channel=out_dim)
            self.embedding2 = nn.Embedding(22, 64)
            self.net2 = seqCNN(in_channel=64, out_channel=out_dim)
            self.net1_readout = seq_readout(out_channel=out_dim)
            self.net2_readout = seq_readout(out_channel=out_dim)

        elif args.task == 'DTI':
            self.net1 = graphCNN(in_channel=34, out_channel=out_dim)
            self.embedding2 = nn.Embedding(22, 64)
            self.net2 = seqCNN(in_channel=64, out_channel=out_dim)
            self.net1_readout = graph_readout(out_channel=out_dim)
            self.net2_readout = seq_readout(out_channel=out_dim)

        from source.nn_model.DIT_model import Decoder
        self.decoder = Decoder(out_dim, 128, 1, 4, 64, device=args.device)
        self.decision_net = nn.Sequential(
                nn.ReLU(),
                nn.Linear(out_dim*2, 64*2),
                nn.ReLU(),
                nn.Linear(64*2, 2),
            )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, amsgrad=True)
        # seq = nn.Sequential(self.decision_net)
        # self.optimizer = torch.optim.Adam(seq.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

        data_dir = 'dataset/{}'.format(args.dataset_name)

        model_dir = os.path.join(data_dir, 'model_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy,
                                                                    args.classifier, args.validation_strategy))

        self.early_stop = EarlyStopping(patience_stop=16, patience_lr=10, verbose=False, delta=0.0001, path=model_dir)

    def forward(self, batch):
        # Compute embeddings of the point clouds:
        # f1, f2 = batch['token_1'], batch['token_2']
        if self.args.task =='DTI':
            f1, f2, f2_mask = batch['token_1'], batch['token_2'], batch['token_2_mask']
        else:
            f1, f2, f1_mask, f2_mask = batch['token_1'], batch['token_2'],  batch['token_1_mask'],  batch['token_2_mask']
            f1_mask = f1_mask.to(self.args.device)
        f1 = f1.to(self.args.device)
        f2 = f2.to(self.args.device)
        f2_mask = f2_mask.to(self.args.device)
        if self.args.task == 'DTI':
            P1_encoder = self.net1(f1.x, f1.edge_index)
            f2 = self.embedding2(f2).transpose(2, 1)
            P2_encoder = self.net2(f2, f2_mask)
            if self.args.fusion_type == 'CAT':
                P1_encoder = self.net1_readout(P1_encoder, f1.batch)
                P2_encoder = self.net2_readout(P2_encoder, f2_mask)
                cat = torch.cat([P1_encoder, P2_encoder], dim=1)
                # cat = P1_encoder * P2_encoder
                out = self.decision_net(cat)
            elif self.args.fusion_type == 'attention':
                # f2 = self.embedding2(f2).transpose(2, 1)
                out = self.decoder(P1_encoder, P2_encoder.transpose(2,1), f1.ptr, f2_mask)
            else:
                print('Fusion type: {}, The fusion type should be chosen from CAT or attention.'.format(self.args.fusion_type))
                exit()
        else:
            f1 = self.embedding1(f1).transpose(2, 1)
            f2 = self.embedding2(f2).transpose(2, 1)
            P1_encoder = self.net1(f1, f1_mask)
            P2_encoder = self.net2(f2, f2_mask)
            if self.args.fusion_type == 'CAT':
                P1_encoder = self.net1_readout(P1_encoder, f1_mask)
                P2_encoder = self.net2_readout(P2_encoder, f2_mask)
                cat = torch.cat([P1_encoder, P2_encoder], dim=1)
                # cat = P1_encoder * P2_encoder
                out = self.decision_net(cat)
            elif self.args.fusion_type == 'attention':
                # f2 = self.embedding2(f2).transpose(2, 1)
                print('PPI and RPI task do not support attention fusion type. Please change the fusion_type to "CAT"')
                out = self.decoder(P1_encoder, P2_encoder.transpose(2,1), f1.ptr, f2_mask)
            else:
                print('Fusion type: {}, The fusion type should be chosen from CAT or attention.'.format(self.args.fusion_type))
                exit()
        return out

    def compute_loss(self, true, proba):
        true = true.to(self.args.device)
        loss = self.criterion(proba, true)
        return loss

    def optimize_parameters(self, true, proba):
        # self.eval()
        loss = self.compute_loss(true, proba)
        loss.backward()
        self.optimizer.step()
        return loss

    def collate_seq(self, batch, key):
        aa = [d[key] for d in batch]
        bb = pad_sequence(aa, batch_first=False, padding_value=0.0).transpose(1, 0)
        mask = bb == 0
        return bb, mask

    def collate_graph(self, batch, key):
        data_list = [d[key] for d in batch]
        return Batch.from_data_list(data_list, follow_batch=['x'])

    def collate_fn(self, batch):
        """Creates mini-batch tensors
        We should build custom collate_fn rather than using default collate_fn
        """
        meta = {}
        meta['token_2'], meta['token_2_mask'] = self.collate_seq(batch, 'token_2')
        if self.args.task != 'DTI':
            meta['token_1'],  meta['token_1_mask'] = self.collate_seq(batch, 'token_1')
        else:
            meta['token_1'] = self.collate_graph(batch, 'token_1')
        meta['label'] = torch.stack([d['label'] for d in batch]).squeeze(-1)
        meta['len_1'] = torch.tensor([len(d['token_1']) for d in batch])
        meta['len_2'] = torch.tensor([len(d['token_2']) for d in batch])
        meta['sample_available'] = [d['mol_1']+'\t'+d['mol_2'] for d in batch]
        return meta

    def loader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn)

    def fit(self, train_dataset, epoch=100):
        pos = train_dataset[:int(len(train_dataset)*0.5)]
        neg = train_dataset[int(len(train_dataset)*0.5):]
        train_loader = pos[:int(len(pos)*0.8)] + neg[:int(len(pos)*0.8)]
        val_loader = pos[int(len(pos)*0.8):] + neg[int(len(pos)*0.8):]

        # train_loader, val_loader = random_split(
        #     train_dataset, [int(len(train_dataset)*0.8), len(train_dataset)-int(len(train_dataset)*0.8)])
        train_loader = self.loader(train_loader)
        val_loader = self.loader(val_loader)
        for i in range(epoch):
            self.train()
            train_loss = []
            for index, data in tqdm(enumerate(train_loader)):
                pre = self.forward(data)
                # loss = self.optimize_parameters(batch['label'], pre)
                loss = self.compute_loss(data['label'], pre)
                train_loss.append(loss.detach().cpu().numpy())
                # loss /= self.args.batch_size
                loss.backward()
                # if index % self.args.batch  == 0 or index == len(train_loader)-1:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.eval()
            with torch.no_grad():
                val_loss = []; val_true = []; val_pre = []
                for batch in val_loader:
                    pre = self.forward(batch)
                    loss = self.compute_loss(batch['label'], pre)
                    val_loss.append(loss.detach().cpu().numpy())
                    val_true.append(batch['label'].detach().cpu().numpy())
                    val_pre.append(F.softmax(pre).detach().cpu().numpy())
            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)
            val_true = np.hstack(val_true)
            val_pre = np.vstack(val_pre)[:,1]
            auc = roc_auc_score(val_true, val_pre)
            print('Epoch: {}, train_loss: {}, val_loss: {}, val_AUC: {}'.format(i, train_loss, val_loss, auc))
            self.early_stop(auc, self)

    def predict(self, val_dataset):
        val_loader = self.loader(val_dataset)

        self.eval()
        with torch.no_grad():
            val_loss = []
            val_true = []
            val_pre = []
            sample_name = []
            for batch in val_loader:
                sample_name.extend(batch['sample_available'])
                pre = self.forward(batch)
                loss = self.compute_loss(batch['label'], pre)
                val_loss.append(loss.detach().cpu().numpy())
                val_true.append(batch['label'].detach().cpu().numpy())
                val_pre.append(F.softmax(pre, dim=1).detach().cpu().numpy())
        val_true = np.hstack(val_true)
        val_pre = np.vstack(val_pre)[:, 1]
        return sample_name, val_true, val_pre