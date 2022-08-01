import torch.nn as nn
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()
        self.backbone_feature = eval(hp.backbone_name + '_Network(hp)')
        self.classifier = nn.Linear(512, 104)
        self.embedding = nn.Linear(512, 64)
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

        loss = 10 * triplet_loss + (cls_loss_sketch + cls_loss_positive + cls_loss_negetive)


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

    def test_forward(self, batch):            #  this is being called only during evaluation
        sketch_feature = self.embedding(self.backbone_feature(batch['meta_batch']['meta_test']['sketch'].to(device)))
        positive_feature = self.embedding(self.backbone_feature(batch['meta_batch']['meta_test']['positive'].to(device)))
        return F.normalize(sketch_feature), F.normalize(positive_feature)




