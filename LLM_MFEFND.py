# from math import dist
import os
from cn_clip.clip import load_from_name
import tqdm
from utils import models_mae
from utils.layers import TokenAttention, clip_fuion
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder, metrics1, visualize_tsne
from .pivot import TransformerLayer, MLP_trans


class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, dataset):
        super(MultiDomainFENDModel, self).__init__()

        self.bert_content = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        self.bert_comment = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        '''self.bert_content = BertModel.from_pretrained('hfl/bert-base-uncased').requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained('hfl/bert-base-uncased').requires_grad_(False)
        self.bert_comment = BertModel.from_pretrained('hfl/bert-base-uncased').requires_grad_(False)'''
        for name, param in self.bert_comment.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.bert_FTR.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.model_size = "base"
        self.score_mapper_ftr_2 = nn.Sequential(nn.Linear(768, 384),
                                                nn.BatchNorm1d(384),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(384, 64),
                                                nn.BatchNorm1d(64),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(64, 1),
                                                nn.Sigmoid()
                                                )
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('hfl/mae_pretrain_vit_base.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False
        self.ClipModel, _ = load_from_name("ViT-B-16", device="cuda", download_root='./')
        self.mlp = MLP(768, mlp_dims, dropout)
        self.transformers_list = torch.nn.ModuleList()
        self.layers = 18
        self.feature_num = 5
        self.content_attention = MaskAttention(768)
        self.background_attention = MaskAttention(768)
        self.comments_attention = MaskAttention(768)
        self.image_fc = nn.Linear(2048, 768)
        self.image_attention = TokenAttention(768)
        self.fc_1 = nn.Linear(768*5,768)
        self.hard_mlp_ftr_2 = nn.Sequential(nn.Linear(768, 384),
                                            nn.ReLU(),
                                            nn.Linear(384, 1),
                                            nn.Sigmoid()
                                            )

        self.clip_fusion = clip_fuion(1024, 768, [348], 0.1)

        feature_num = 5
        self.mlp_img = torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                            range(feature_num)])

        self.mlp_text = torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                             range(feature_num)])
        self.pivot_mlp_fusion = torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                                     range(feature_num)])
        self.pivot_background_fusion = torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                                            range(feature_num)])
        self.pivot_comments_fusion = torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                                          range(feature_num)])
        self.transformers_list = torch.nn.ModuleList()
        self.mlp_img_list = torch.nn.ModuleList()
        self.mlp_text_list = torch.nn.ModuleList()
        self.pivot_mlp_fusion_list = torch.nn.ModuleList()
        self.pivot_background_fusion_list = torch.nn.ModuleList()
        self.pivot_comments_fusion_list = torch.nn.ModuleList()
        self.content_attention_rationale = SelfAttentionFeatureExtract(1, 768)
        self.content_attention_comments = SelfAttentionFeatureExtract(1, 768)
        self.rationale_attention_content = SelfAttentionFeatureExtract(1, 768)
        self.comments_attention_content = SelfAttentionFeatureExtract(1, 768)
        self.fusion_fc1 =nn.Linear(768*5,768)
        self.fusion_fc2 = nn.Linear(768 * 2, 768 * 1)

        self.transformers_list.append(
            torch.nn.ModuleList([TransformerLayer(768, head_num=4, dropout=0.6,
                                                  attention_dropout=0,
                                                  initializer_range=0.02) for _ in
                                 range(18)]))
        self.mlp_img_list.append(
            torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                 range(5)]))
        self.mlp_text_list.append(
            torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                 range(5)]))
        self.pivot_mlp_fusion_list.append(
            torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                 range(5)]))
        self.pivot_background_fusion_list.append(
            torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in
                                 range(5)]))
        self.pivot_comments_fusion_list.append(
            torch.nn.ModuleList([MLP_trans(768, 768, dropout=0.6) for _ in

                                 range(5)]))
        emb_size = 768
        self.active = nn.SiLU()
        self.dropout2 = nn.Dropout(0.2)
        self.mlp_star_f1 = nn.Linear(emb_size * 4, 2 * emb_size)
        self.mlp_star_f2 = nn.Linear(2 * emb_size, emb_size)
        self.fc_re = nn.Linear(768 * 2, 768)
        self.fc_comments = nn.Linear(768 * 2, 768)
        self.mlp_star_f1_list = torch.nn.ModuleList()
        self.mlp_star_f2_list = torch.nn.ModuleList()
        self.hard_mlp_ftr_3 = nn.Sequential(nn.Linear(768, 384),
                                            nn.ReLU(),
                                            nn.Linear(384, 1),
                                            nn.Sigmoid()
                                            )
        self.score_mapper_ftr_3 = nn.Sequential(nn.Linear(768, 384),
                                                nn.BatchNorm1d(384),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(384, 64),
                                                nn.BatchNorm1d(64),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(64, 1),
                                                nn.Sigmoid()
                                               )
        for i in range(9):
            self.mlp_star_f1_list.append(nn.Linear(emb_size * 4, 2 * emb_size))
            self.mlp_star_f2_list.append(nn.Linear(2 * emb_size, emb_size))

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        background, background_masks = kwargs['background'], kwargs['background_masks']
        comment, comment_masks = kwargs['comment'], kwargs['comment_masks']
        CILP_image_feature = kwargs['image_features']
        clip_content_features = kwargs['clip_content_features']
        image_tensor = CILP_image_feature.squeeze(1)
        with torch.no_grad():
            clip_image_feature = self.ClipModel.encode_image(image_tensor)  # ([64, 512])
            clip_text_feature = self.ClipModel.encode_text(clip_content_features)  # ([64, 512])
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
            clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)

        bert_background_feature = self.bert_content(background, attention_mask=background_masks)[0]
        bert_content_feature = self.bert_FTR(content, attention_mask=content_masks)[0]

        bert_comment_feature = self.bert_comment(comment, attention_mask=comment_masks)[0]

        expert_content_background_1, _ = self.content_attention_rationale(bert_content_feature, bert_background_feature,
                                                                          content_masks)
        mutual_content_background_2, _ = self.rationale_attention_content(bert_background_feature, bert_content_feature,
                                                                          background_masks)
        expert_content_background_1 = torch.mean(expert_content_background_1, dim=1)
        mutual_content_background_2 = torch.mean(mutual_content_background_2, dim=1)
        #background_hard_ftr_2_pred = self.hard_mlp_ftr_2(mutual_content_background_2).squeeze(1)
        reweight_score_background = self.score_mapper_ftr_2(mutual_content_background_2)
        expert_content_background_1 = expert_content_background_1*reweight_score_background
        #mutual_content_background = torch.cat((mutual_content_background_1, mutual_content_background_2), dim=1)
        expert_content_comments_1, _ = self.content_attention_comments(bert_content_feature, bert_comment_feature,
                                                                       content_masks)
        mutual_content_comments_2, _ = self.comments_attention_content(bert_comment_feature, bert_content_feature,
                                                                       comment_masks)
        mutual_content_comments_1 = torch.mean(expert_content_comments_1, dim=1)
        mutual_content_comments_2 = torch.mean(mutual_content_comments_2, dim=1)
        #hard_mlp_ftr_3 = self.hard_mlp_ftr_3(mutual_content_comments_2).squeeze(1)
        reweight_score_comments = self.score_mapper_ftr_3(mutual_content_comments_2)
        expert_content_comments_1 = reweight_score_comments*mutual_content_comments_1
        #mutual_content_comments = torch.cat((mutual_content_comments_1, mutual_content_comments_2), dim=1)
        #mutual_content_comments = self.fc_comments(mutual_content_comments)
        bert_content_feature, _ = self.content_attention(bert_content_feature, mask=content_masks)
        bert_background_feature, _  = self.background_attention(bert_background_feature)
        bert_comments_feature, _ = self.comments_attention(bert_comment_feature)
        image_feature = self.image_model.forward_ying(image_tensor)  # ([64, 197, 768])
        image_atn_feature, _ = self.image_attention(image_feature)
        clip_fusion_feature = torch.cat((clip_text_feature, clip_image_feature), dim=1)
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())  # torch.Size([64, 768])

        clip_fusion_feature = clip_fusion_feature.to(torch.float32)

        final_fusion_feature = self.fusion_img_text(image_atn_feature, bert_content_feature, clip_fusion_feature,
                                                    bert_background_feature,
                                                    bert_comments_feature, self.mlp_img_list[0],
                                                    self.mlp_text_list[0],
                                                    self.pivot_mlp_fusion_list[0],
                                                    self.pivot_background_fusion_list[0],
                                                    self.pivot_comments_fusion_list[0],
                                                    self.transformers_list[0],
                                                    self.mlp_star_f1_list[0],
                                                    self.mlp_star_f2_list[0],num_layers=4)
        # final_fusion_feature = torch.cat((final_fusion_feature, domain_embedding), dim=1)
        '''final_fusion_feature = torch.cat((image_atn_feature, bert_content_feature, clip_fusion_feature,
                                          mutual_content_background,
                                          mutual_content_comments), dim=1)'''
        #final_fusion_feature = self.fc_1(final_fusion_feature)
        #final_fusion_feature = self.fusion_fc1(final_fusion_feature)

        label_pred = self.mlp(final_fusion_feature)
        res = {'classify_pred': torch.sigmoid(label_pred.squeeze(1)), 'final_fusion_feature': final_fusion_feature}
        return res


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 dataset,
                 early_stop=5,
                 epoches=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.dataset = dataset

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def train(self, logger=None):
        if logger:
            logger.info('start training......')

        self.model = MultiDomainFENDModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout,
                                          self.dataset)
        if self.use_cuda:
            self.model = self.model.cuda()

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                optimizer.zero_grad()  # 只调用一次
                res = self.model(**batch_data)
                loss = loss_fn(res['classify_pred'], label.float())
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()  # 通常在每个 epoch 结束后调用

                avg_loss.add(loss.item())

            print(f'Training Epoch {epoch + 1}; Loss {avg_loss.item()}')
            status = f'[{epoch}] lr = {self.lr}; batch_loss = {loss.item()}; average_loss = {avg_loss.item()}'
            if logger:
                logger.info(status)  # 记录日志信息

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, 'parameter_mdfend.pkl'))
                logger.info("Model saved successfully.")
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue

        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')))
        results = self.test(self.test_loader)
        if logger:
            logger.info("start testing......")
            logger.info(f"test score: {results}\n\n")
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')
    def test(self, dataloader):
        pred = []
        label = []
        #category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                #batch_category = batch_data['category']
                res = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(res['classify_pred'].detach().cpu().numpy().tolist())
                #category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics1(label, pred)


