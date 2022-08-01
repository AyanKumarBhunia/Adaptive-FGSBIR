import torch.nn as nn
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Networks import VGG_Network, Resnet50_Network,  InceptionV3_Network
device = "cpu"
import os
from torchvision.utils import save_image

class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()
        self.backbone_feature = eval(hp.backbone_name + '_Network(hp)')
        self.classifier = nn.Linear( hp.latent_embdedding_dim, 125)
        self.embedding = nn.Linear( hp.latent_embdedding_dim, 64)
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.train_params = list(self.backbone_feature.parameters()) + list(self.embedding.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.hp = hp


    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()

        positive_feature = self.backbone_feature(batch['positive'].to(device))
        negative_feature = self.backbone_feature(batch['negative'].to(device))
        sample_feature = self.backbone_feature(batch['sketch'].to(device))

        triplet_loss = self.loss(F.normalize(self.embedding(sample_feature)),
                                 F.normalize(self.embedding(positive_feature)), F.normalize(self.embedding(negative_feature)))

        cls_loss_sketch = F.cross_entropy(self.classifier(sample_feature), batch['class_label'].to(device))
        cls_loss_positive = F.cross_entropy(self.classifier(positive_feature), batch['class_label'].to(device))
        cls_loss_negetive = F.cross_entropy(self.classifier(negative_feature), batch['class_label'].to(device))

        loss = self.hp.lambda_1 * triplet_loss + self.hp.lambda_2 * (cls_loss_sketch + cls_loss_positive + cls_loss_negetive)


        loss.backward()
        self.optimizer.step()

        return loss.item() 

    def evaluate(self, datloader_Test):
        start_time = time.time()

        self.eval()
        Total_Top1, Total_Top10 = 0, 0
        Total_Sketch = 0
        accuracy_list = []
        for i_batch, sanpled_batch in enumerate(datloader_Test):

            Image_Feature_ALL = []
            Image_Name = []
            Sketch_Feature_ALL = []
            Sketch_Name = []

            batch = sanpled_batch[0]
            sketch_feature, positive_feature= self.test_forward(batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(batch['meta_batch']['meta_test']['sketch_path'])

            for i_num, positive_name in enumerate(batch['meta_batch']['meta_test']['positive_path']):
                if positive_name not in Image_Name:
                    path = batch['meta_batch']['meta_test']['positive_path'][i_num]
                    positive_path = path.split('/')[-2]+ '/' + path.split('/')[-1].split('.')[0]
                    Image_Name.append(positive_path)
                    Image_Feature_ALL.append(positive_feature[i_num])

            rank = torch.zeros(len(Sketch_Name))
            Image_Feature_ALL = torch.stack(Image_Feature_ALL)

            for num, sketch_feature in enumerate(Sketch_Feature_ALL):
                s_name = Sketch_Name[num]
                sketch_query_name = s_name.split('/')[0] + '/' + s_name.split('/')[1].split('-')[0]
                position_query = Image_Name.index(sketch_query_name)

                distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
                target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                      Image_Feature_ALL[position_query].unsqueeze(0))

                rank[num] = distance.le(target_distance).sum()

            top1 = rank.le(1).sum().numpy() / rank.shape[0]
            top10 = rank.le(10).sum().numpy() / rank.shape[0]

            Total_Top1 = Total_Top1 + rank.le(1).sum().numpy()
            Total_Top10 = Total_Top10 + rank.le(10).sum().numpy()
            Total_Sketch = Total_Sketch + rank.shape[0]

            print('Class_Name: %s ===>>: Top1 Accuracy: %f, Top10 Accuracy: %f' % (batch['class_name'], top1, top10))
            accuracy_list.append('Class_Name: %s ===>>: Top1 Accuracy: %f, Top10 Accuracy: %f' % (batch['class_name'], top1, top10))

        print('Time to EValuate:{}'.format(time.time() - start_time))

        return Total_Top1/Total_Sketch, Total_Top10/Total_Sketch, accuracy_list

    def evaluate_plot(self, datloader_Test):
        start_time = time.time()


        folder = './CVPR_Supplementory/'
        self.eval()
        Total_Top1, Total_Top10 = 0, 0
        Total_Sketch = 0
        accuracy_list = []
        for i_batch, sanpled_batch in enumerate(datloader_Test):

            Image_Feature_ALL = []
            Image_Name = []
            Sketch_Feature_ALL = []
            Sketch_Name = []
            Image_ALL = []

            batch = sanpled_batch[0]
            sketch_feature, positive_feature= self.test_forward(batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(batch['meta_batch']['meta_test']['sketch_path'])

            mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).to(device)
            std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).to(device)
            batch['meta_batch']['meta_test']['positive'].sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

            for i_num, positive_name in enumerate(batch['meta_batch']['meta_test']['positive_path']):
                if positive_name not in Image_Name:
                    path = batch['meta_batch']['meta_test']['positive_path'][i_num]
                    positive_path = path.split('/')[-2]+ '/' + path.split('/')[-1].split('.')[0]
                    Image_Name.append(positive_path)
                    Image_Feature_ALL.append(positive_feature[i_num])
                    Image_ALL.append(batch['meta_batch']['meta_test']['positive'][i_num])

            rank = torch.zeros(len(Sketch_Name))
            Image_Feature_ALL = torch.stack(Image_Feature_ALL)

            for num, sketch_feature in enumerate(Sketch_Feature_ALL):
                s_name = Sketch_Name[num]
                sketch_query_name = s_name.split('/')[0] + '/' + s_name.split('/')[1].split('-')[0]
                position_query = Image_Name.index(sketch_query_name)

                distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
                target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                      Image_Feature_ALL[position_query].unsqueeze(0))

                [_, indices] = distance.sort()

                rank[num] = distance.le(target_distance).sum()

                folder_name = os.path.join(folder, s_name)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                s_name_sketch = os.path.join(folder_name, s_name.split('/')[0] + '_' + s_name.split('/')[-1] + '_Sketch.png')
                s_name_photo = os.path.join(folder_name, s_name.split('/')[0] + '_' + s_name.split('/')[-1] + '_True_Photo.png')

                save_image(1. - batch['meta_batch']['meta_test']['sketch'][num].unsqueeze(0), s_name_sketch, normalize=True)
                save_image(Image_ALL[position_query].unsqueeze(0), s_name_photo, normalize=True)

                for num_k, x_num in enumerate(indices[:10]):
                    s_name_ret = os.path.join(folder_name, s_name.split('/')[0] +
                                              '_' + s_name.split('/')[-1] + '___' + str(num_k)  +'___Photo.png')
                    save_image(batch['meta_batch']['meta_test']['positive'][x_num].unsqueeze(0), s_name_ret)


            top1 = rank.le(1).sum().numpy() / rank.shape[0]
            top10 = rank.le(10).sum().numpy() / rank.shape[0]

            Total_Top1 = Total_Top1 + rank.le(1).sum().numpy()
            Total_Top10 = Total_Top10 + rank.le(10).sum().numpy()
            Total_Sketch = Total_Sketch + rank.shape[0]

            print('Class_Name: %s ===>>: Top1 Accuracy: %f, Top10 Accuracy: %f' % (batch['class_name'], top1, top10))
            accuracy_list.append('Class_Name: %s ===>>: Top1 Accuracy: %f, Top10 Accuracy: %f' % (batch['class_name'], top1, top10))

        print('Time to EValuate:{}'.format(time.time() - start_time))

        return Total_Top1/Total_Sketch, Total_Top10/Total_Sketch, accuracy_list

    def test_forward(self, batch):            #  this is being called only during evaluation
        sketch_feature = self.embedding(self.backbone_feature(batch['meta_batch']['meta_test']['sketch'].to(device)))
        positive_feature = self.embedding(self.backbone_feature(batch['meta_batch']['meta_test']['positive'].to(device)))
        return F.normalize(sketch_feature), F.normalize(positive_feature)




