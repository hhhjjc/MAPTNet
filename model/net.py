import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

import model.resnet as models
import model.vgg as vgg_models
from model.multiscale import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from model.transformer import PositionEmbeddingSine
#from model.ASPP_v3plus import ASPP
class net(nn.Module):
    def __init__(self, layers=50, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True, shot=1, vgg=False):
        super(net, self).__init__()
        # assert layers in [50, 101, 152]
        # assert classes == 2
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = criterion
        self.shot = shot
        self.vgg = vgg

        models.BatchNorm = BatchNorm

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
            self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('>>>>>>>>> Using ResNet {}<<<<<<<<<'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        reduce_dim = 256

        self.pool = nn.AdaptiveMaxPool2d(1)
        
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, 2, kernel_size=3, padding=1, bias=False),
        )
        ################
        self.down = nn.ModuleList([nn.Sequential(
            nn.Conv2d(reduce_dim * 2 ** (3 - i), reduce_dim, kernel_size=1, padding=0, bias=False)) for i in range(4)])
        
        self.smooth = nn.ModuleList([nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False)) for i in range(3)])
        ################
        self.fuse = nn.ModuleList([nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        ) for _ in range(3)])
        
        self.fuse_up = nn.ModuleList([nn.Sequential(
            nn.Conv2d(reduce_dim + 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        ) for _ in range(3)])
        ################
        self.supervise_s = nn.ModuleList([nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, classes, kernel_size=3, padding=1, bias=False)
        ) for _ in range(4)])
        
        self.supervise_q = nn.ModuleList([nn.Sequential(
            nn.Conv2d(reduce_dim, classes, kernel_size=3, padding=1, bias=False)
        ) for _ in range(4)])
        ################
        self.num_subclass = 1
        num_heads = 4
        self.pro_embed = nn.ModuleList([nn.Embedding(self.num_subclass, reduce_dim) for _ in range(4)])
        self.pro_layer = nn.ModuleList([PositionEmbeddingSine(reduce_dim // 2, normalize=True) for _ in range(4)])
        self.transformer_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model=reduce_dim, nhead=num_heads, dropout=0.0, normalize_before=False) for _ in range(4)
        ])
        self.transformer_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(d_model=reduce_dim, nhead=num_heads, dropout=0.0, normalize_before=False) for _ in range(4)
        ])
        self.transformer_ffn_layers = nn.ModuleList([
            FFNLayer(d_model=reduce_dim, dim_feedforward=reduce_dim * 2, dropout=0.0, normalize_before=False) for _ in range(4)
        ])
        ################
        #self.aspp = ASPP()

    def forward(self, x, s_x=torch.FloatTensor(4, 1, 3, 200, 200).cuda(), s_y=torch.FloatTensor(4, 1, 200, 200).cuda(), y=None):
        x_size = x.size()  # [4,3,200,200]
        b, h, w = x_size[0], int(x_size[-1]), int(x_size[-2])


        with torch.no_grad():
            query_feat_0 = self.layer0(x)  # [2, 128, 50, 50] 224/112/56/28/14/7
            query_feat_1 = self.layer1(query_feat_0)  # [2, 256, 50, 50]
            query_feat_2 = self.layer2(query_feat_1)  # [2, 512, 25, 25]
            query_feat_3 = self.layer3(query_feat_2)  # [2, 1024, 13, 13]
            query_feat_4 = self.layer4(query_feat_3)  # [2, 2048, 7, 7]

        
        query_feat_4 = self.down[0](query_feat_4)
        query_feat_3 = self.down[1](query_feat_3)
        query_feat_2 = self.down[2](query_feat_2)
        query_feat_1 = self.down[3](query_feat_1)
        query_4 = query_feat_4
        query_3 = _upsample_add(query_4, query_feat_3)
        query_2 = _upsample_add(query_3, query_feat_2)
        query_1 = _upsample_add(query_2, query_feat_1)
        query_3 = self.smooth[0](query_3)
        query_2 = self.smooth[1](query_2)
        query_1 = self.smooth[2](query_1)
        
        pro_s4, pro_s3, pro_s2, pro_s1 = [], [], [], []
        #################### ----- 5shot ----- ####################
        for i in range(self.shot):
            supp_gt = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            # gt_list.append(supp_gt)

            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                supp_feat_4 = self.layer4(supp_feat_3)
            
            supp_feat_4 = self.down[0](supp_feat_4)
            supp_feat_3 = self.down[1](supp_feat_3)
            supp_feat_2 = self.down[2](supp_feat_2)
            supp_feat_1 = self.down[3](supp_feat_1)
            
            supp_4 = supp_feat_4
            supp_3 = _upsample_add(supp_4, supp_feat_3)
            supp_2 = _upsample_add(supp_3, supp_feat_2)
            supp_1 = _upsample_add(supp_2, supp_feat_1)
            supp_3 = self.smooth[0](supp_3)
            supp_2 = self.smooth[1](supp_2)
            supp_1 = self.smooth[2](supp_1)

            supp_gt_4 = F.interpolate(supp_gt, size=supp_4.shape[-2:], mode='bilinear',align_corners=True)
            supp_gt_3 = F.interpolate(supp_gt, size=supp_3.shape[-2:], mode='bilinear',align_corners=True)
            supp_gt_2 = F.interpolate(supp_gt, size=supp_2.shape[-2:], mode='bilinear',align_corners=True)
            supp_gt_1 = F.interpolate(supp_gt, size=supp_1.shape[-2:], mode='bilinear',align_corners=True)

            supp_mask_4 = supp_4 * supp_gt_4
            supp_mask_3 = supp_3 * supp_gt_3
            supp_mask_2 = supp_2 * supp_gt_2
            supp_mask_1 = supp_1 * supp_gt_1
            
            pro_s_4 = self.pool(supp_mask_4)
            pro_s_3 = self.pool(supp_mask_3)
            pro_s_2 = self.pool(supp_mask_2)
            pro_s_1 = self.pool(supp_mask_1)   
            
            if self.training:
                out_4_s = self.supervise_s[0](torch.cat([supp_4, pro_s_4.expand(b, 256, *supp_4.shape[-2:])], dim=1))
                out_3_s = self.supervise_s[1](torch.cat([supp_3, pro_s_3.expand(b, 256, *supp_3.shape[-2:])], dim=1))
                out_2_s = self.supervise_s[2](torch.cat([supp_2, pro_s_2.expand(b, 256, *supp_2.shape[-2:])], dim=1))
                out_1_s = self.supervise_s[3](torch.cat([supp_1, pro_s_1.expand(b, 256, *supp_1.shape[-2:])], dim=1))
            
            
            if i == 0:
                pro_s4 = pro_s_4.squeeze(-1).permute(0, 2, 1)
                pro_s3 = pro_s_3.squeeze(-1).permute(0, 2, 1)
                pro_s2 = pro_s_2.squeeze(-1).permute(0, 2, 1)
                pro_s1 = pro_s_1.squeeze(-1).permute(0, 2, 1)
            else:
                pro_s4 = torch.cat([pro_s4, pro_s_4.squeeze(-1).permute(0, 2, 1)], dim=1)
                pro_s3 = torch.cat([pro_s3, pro_s_3.squeeze(-1).permute(0, 2, 1)], dim=1)
                pro_s2 = torch.cat([pro_s2, pro_s_2.squeeze(-1).permute(0, 2, 1)], dim=1)
                pro_s1 = torch.cat([pro_s1, pro_s_1.squeeze(-1).permute(0, 2, 1)], dim=1)
        ################################################# prototype transform
        def prototype_transform(query_feat, pro_s, pro_layer, embed_weight, transformer_cross, transformer_self, ffn_layer):
            kv = pro_layer(query_feat)
            patches = query_feat.permute(0, 2, 3, 1).contiguous().view(b, -1, 256)
            pro = pro_s + transformer_cross(tgt=pro_s, query_pos=embed_weight, memory=patches, pos=kv)
            pro = transformer_self(tgt=pro)
            pro = ffn_layer(pro)
            return torch.mean(pro.permute(0, 2, 1), dim=-1, keepdim=True)
        
        pro_4 = prototype_transform(query_4, pro_s4, self.pro_layer[0], self.pro_embed[0].weight, self.transformer_cross_attention_layers[0], self.transformer_self_attention_layers[0], self.transformer_ffn_layers[0])
        pro_3 = prototype_transform(query_3, pro_s3, self.pro_layer[1], self.pro_embed[1].weight, self.transformer_cross_attention_layers[1], self.transformer_self_attention_layers[1], self.transformer_ffn_layers[1])
        pro_2 = prototype_transform(query_2, pro_s2, self.pro_layer[2], self.pro_embed[2].weight, self.transformer_cross_attention_layers[2], self.transformer_self_attention_layers[2], self.transformer_ffn_layers[2])
        pro_1 = prototype_transform(query_1, pro_s1, self.pro_layer[1], self.pro_embed[1].weight, self.transformer_cross_attention_layers[3], self.transformer_self_attention_layers[3], self.transformer_ffn_layers[3])
        
        ################################################# prototype decoder
        def prototype_decoder(query_feat, pro_feat, fuse_layer, upsample_q=None):
            pro_feat = pro_feat.unsqueeze(-1).expand(b, 256, *query_feat.shape[-2:])
            fuse_feat = torch.cat([query_feat, pro_feat], dim=1)
            fuse_feat = fuse_layer(fuse_feat)
            if upsample_q is not None:
                fuse_feat = torch.cat([fuse_feat, upsample_q], dim=1)
            return fuse_feat
        
        fuse_4 = prototype_decoder(query_4, pro_4, self.fuse[0])
        out_4_q = self.supervise_q[0](F.interpolate(fuse_4, size=(h, w), mode='bilinear', align_corners=True))
        up_4 = F.interpolate(out_4_q, size=query_3.shape[-2:], mode='bilinear', align_corners=True)

        fuse_3 = self.fuse_up[0](prototype_decoder(query_3, pro_3, self.fuse[0], up_4))
        out_3_q = self.supervise_q[1](F.interpolate(fuse_3, size=(h, w), mode='bilinear', align_corners=True))
        up_3 = F.interpolate(out_3_q, size=query_2.shape[-2:], mode='bilinear', align_corners=True)

        fuse_2 = self.fuse_up[1](prototype_decoder(query_2, pro_2, self.fuse[1], up_3))
        out_2_q = self.supervise_q[2](F.interpolate(fuse_2, size=(h, w), mode='bilinear', align_corners=True))
        up_2 = F.interpolate(out_2_q, size=query_1.shape[-2:], mode='bilinear', align_corners=True)

        fuse_1 = self.fuse_up[2](prototype_decoder(query_1, pro_1, self.fuse[2], up_2))
        out_1_q = self.supervise_q[3](F.interpolate(fuse_1, size=(h, w), mode='bilinear', align_corners=True))
        ################################################# final prediction
        #fuse = self.aspp(fuse_1)
        final = F.interpolate(fuse_1, size=(h, w), mode='bilinear', align_corners=True)
        query_pred_mask = self.cls(final)
        query_pred_mask_save = torch.argmax(query_pred_mask[0].squeeze(0).permute(1, 2, 0), axis=-1).detach().cpu().numpy()
        query_pred_mask_save[query_pred_mask_save!=0] = 255
        query_pred_mask_save[query_pred_mask_save==0] = 0
    
        if self.training:
            main_loss = self.criterion(query_pred_mask, y.long())
            aux_loss_s = self.criterion(out_4_s, supp_gt_4.squeeze(1).long()) +\
                        self.criterion(out_3_s, supp_gt_3.squeeze(1).long()) +\
                        self.criterion(out_2_s, supp_gt_2.squeeze(1).long()) +\
                        self.criterion(out_1_s, supp_gt_1.squeeze(1).long())
            aux_loss_q = self.criterion(out_4_q, y.long()) +\
                        self.criterion(out_3_q, y.long()) +\
                        self.criterion(out_2_q, y.long()) +\
                        self.criterion(out_1_q, y.long())
            return query_pred_mask.max(1)[1], main_loss + aux_loss_s + aux_loss_q
        else:
            return query_pred_mask, query_pred_mask_save
        
def _upsample_add(x, y):
    _,_,H,W = y.shape
    return F.interpolate(x, size=(H,W), mode='bilinear') + y

def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4
