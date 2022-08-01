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
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)


    model = FGSBIR_Model(hp)
    model.to(device)
    model.load_state_dict(torch.load('./../FGSBIR_Baselines_Final/Seen_Class/model/BaselineFGSBIR_Model_best.pth'), strict=False)
    step_count, top1, top10 = -1, 0, 0

    model_path = os.path.join('./model', hp.current_class)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for i_epoch in range(hp.max_epoch):
        start = time.time()
        for batch_data in dataloader_Train:

            step_count = step_count + 1
            model.train()
            loss = model.train_model(batch=batch_data)

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}'.format
                      (i_epoch, step_count, loss, top1, top10))

            if step_count % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    top1_eval, top10_eval, accuracy_list = model.evaluate(dataloader_Test)
                    print('results : Top1 Accuracy: {}'.format(top1_eval))

                if top1_eval > top1:
                    torch.save(model.state_dict(), model_path + '/SingleClass_' + str(hp.current_class) + '_Model_best.pth')
                    top1, top10 = top1_eval, top10_eval

                    with open(model_path + '/Results.txt', 'a') as filehandle:
                        np.savetxt(filehandle, np.array(accuracy_list), fmt='%s',
                                   newline='\n', header='Iteration Number' + str(step_count) + ':', footer='\n')
                    print('Model Updated')

        print ('Epoch time : ', time.time()-start)

