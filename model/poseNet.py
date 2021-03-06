import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model.layers import Conv, Hourglass, Pool, Residual
from loss.calc_loss import Calc_loss
from data import MPII_dataLoader, MPII_test_dataLoader
from model import config
from utils.inference import do_inference, mpii_eval

from model import test_save_model


class poseNet(pl.LightningModule):      #not pl.core.LightningModule!!!

    def __init__(self, bn=False, **kwargs):
        super(poseNet, self).__init__()

        self.hparams.num_workers = 8    #workers number
        self.this_config = config.__config__
        inp_dim = self.this_config['inp_dim']
        oup_dim = self.this_config['oup_dim']
        self.hparams.batch_size = self.this_config['batch_size']
        self.nstack = self.this_config['nstack']
        self.threshold = self.this_config['threshold']
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),  #res: 128*128 if input_res=256
            Residual(64, 128),
            Pool(2, 2), #res:64*64 the max res in the network
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        #it has to use ModuleList, because different module has different parameters
        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn)
            )for num in range(self.nstack)
        ])

        '''
        self.afterHg = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            )for num in range(self.nstack)
        ])

        self.pred_heatmaps = nn.ModuleList([
            Conv(inp_dim, oup_dim, 1, bn=False, relu=False) for num in range(self.nstack)
        ])

        self.module_outs = nn.ModuleList([
            Conv(inp_dim, inp_dim, 1, bn=False, relu=False) for num in range(self.nstack-1)
        ])

        self.merge_pred_heatmaps = nn.ModuleList([
            Conv(oup_dim, inp_dim, 1, bn=False, relu=False) for num in range(self.nstack-1)
        ])
        '''

        self.features = nn.ModuleList([
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(self.nstack)])
        from model.layers import Merge
        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(self.nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(self.nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(self.nstack - 1)])


        self.calc_loss = Calc_loss(self.nstack)

        #for test phase
        self.all_gt_kps = []
        self.all_preds = []
        self.all_norm = []

    def forward(self, imgs):
        input = imgs.permute(0, 3, 1, 2) ##swap order [batch_size,channel,size,size]
        input = self.pre(input)
        '''
        all_pred_heatmaps = []  # nstack*16*size*size
        for i in range(self.nstack):
            hg = self.hgs[i](input)
            afterHg = self.afterHg[i](hg)
            pred_heatmaps = self.pred_heatmaps[i](afterHg) # 16*size*size
            all_pred_heatmaps.append(pred_heatmaps)
            if i < self.nstack - 1:
                input += self.module_outs[i](afterHg) + self.merge_pred_heatmaps[i](pred_heatmaps)

        return torch.stack(all_pred_heatmaps, dim=1) #lists are stacked to become a tensor; dim=0:所有个tensor直接拼起来；dim=1，所有tensor第一个维度拼起来作为第一个维度...
        '''
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](input)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)  # [batch_size, njoints, out_size, out_size]
            # print('pred:',preds.shape)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                input = input + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)


    def training_step(self, batch, batch_idx):
        """
        step == batch
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        batch_imgs, heatmaps_gt = batch   #[batch_size, channel(3), size, size] [batch_size, n_joints, size, size]

        #visiualization
        #cv2.imshow('img',batch_imgs[0].cpu().numpy())
        #cv2.waitKey(0)

        #get prediction
        combined_heatmap_preds = self(batch_imgs)   #[batch_size, nstack, n_joints, size, size]
        #debug pl.core.LightningModule is not correct
        #print(combined_heatmap_preds.shape)
        #print(torch.sum(combined_heatmap_preds))
        #print(self.state_dict()['pre.0.conv.weight'])
        #print(self.state_dict()['pre.0.conv.weight'].grad)
        #calculate loss
        train_loss = self.calc_loss(combined_heatmap_preds, heatmaps_gt)
        train_result = pl.TrainResult(minimize=train_loss) #minimize: what metrics to do bp learning
        train_result.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return train_result


    def validation_step(self, batch, batch_idx):
        """
        Called every batch
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        batch_imgs, heatmaps_gt = batch
        combined_heatmap_preds = self(batch_imgs)

        #self.on_save_checkpoint()

        val_loss = self.calc_loss(combined_heatmap_preds, heatmaps_gt)
        val_result = pl.EvalResult(early_stop_on=val_loss, checkpoint_on=val_loss)
        val_result.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        
#        val_result.log('acc')
        return val_result
        #return {'val_loss':val_loss}

    def test_step(self, batch, batch_idx):
        batch_imgs, batch_gt_kps, heatmaps_gt, batch_center, batch_scale, batch_norm = batch    #[batch_size, size, size, channel]
        combined_heatmap_preds = self(batch_imgs)
        batch_size = self.hparams.batch_size
        for i in range(batch_size):
            if (batch_idx)*batch_size+i>=self.this_config['num_eval']:
                break
            img = batch_imgs[i,:,:,:]   #[size,size,channel]
            img = img.view([1, self.this_config['input_res'], self.this_config['input_res'], 3])
            gt_kps = batch_gt_kps[i]
            center = batch_center[i]
            scale = batch_scale[i]
            norm = batch_norm[i]

            pred = do_inference(img, self, self.this_config, center, scale)

            self.all_gt_kps.append(gt_kps.cpu().numpy())
            self.all_preds.append(pred)
            self.all_norm.append(norm.cpu().numpy())


        test_loss = self.calc_loss(combined_heatmap_preds, heatmaps_gt)

        tensorboard_logs = {'test_loss': test_loss}
        return {'test_loss': test_loss, 'test_log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        dict = {
            'state_dict': self.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'epoch': 1
        }
        test_save_model.save_checkpoint(dict ,False, self.this_config['checkpoint_path']+'/checkpoint.pt')
        print('=> save checkpoint')
        return {}



    #If don't use EvalResult 
   # def validation_epoch_end(self, outputs):
        """
        Called every epoch
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
      #  avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
       # tensorboard_logs = {'val_loss': avg_loss}
      #  return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}

        mpii_eval(self.all_preds, self.all_gt_kps, self.all_norm, self.this_config['threshold'])

        #return {}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.this_config['lr'])
        scheduler = {'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.this_config['lr_decay'], patience=2, verbose=True),
                     'interval':'epoch',
                     'monitor':'val_checkpoint_on',
                     'frequency': 1
                     }
        return [self.optimizer], [scheduler]


    def prepare_data(self):
        #download data
       pass

    def setup(self, stage):
        #split train and valid dataset
        self.mpii_train, self.mpii_valid = MPII_dataLoader.init(self.this_config)
        self.mpii_test = MPII_test_dataLoader.init(self.this_config)

    def train_dataloader(self):
        return DataLoader(self.mpii_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mpii_valid, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mpii_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)
