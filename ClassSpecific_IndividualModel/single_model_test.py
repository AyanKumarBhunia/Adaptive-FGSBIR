import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import numpy as np
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Meat Learning for Fine-Grained SBIR Model')
    parser.add_argument('--class_id', type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default='sketchy-FGSBIR')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--root_dir', type=str, default='./../Dataset/sketchy/')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--eval_freq_iter', type=int, default=25)
    parser.add_argument('--print_freq_iter', type=int, default=10)



    # for clas_num in range(0, 125):
    hp = parser.parse_args()
    print(hp.class_id)

    model = FGSBIR_Model(hp)
    model.to(device)

    accuracy_list = []
    top1_all, top10_all = [], []

    for id_x in range(125):
        hp.class_id= id_x
        dataloader_Train, dataloader_Test = get_dataloader(hp)
        # print(hp)

        model_folder = os.path.join('./model', hp.current_class)
        print(model_folder)
        model_path = model_folder + '/SingleClass_' + str(hp.current_class) + '_Model_best.pth'
        model.load_state_dict(torch.load(model_path), strict=False)

        with torch.no_grad():
            top1_eval, top10_eval, accuracy_list_ = model.evaluate(dataloader_Test)
            print('Class {} : Top1 Accuracy: {}'.format(hp.current_class, top1_eval))
            accuracy_list.append('Class_Name: %s ===>>: Top1 Accuracy: %f, Top10 Accuracy: %f' % ( hp.current_class, top1_eval, top10_eval))

            top1_all.append(top1_eval)
            top10_all.append(top10_eval)

    with open('./Single_Model_Results.txt', 'a+') as filehandle:
        np.savetxt(filehandle, np.array(accuracy_list), fmt='%s',
                   newline='\n', header='Iteration Number' + str(0) + ':', footer='\n')

    print(np.mean(top1_all), np.mean(top10_all))