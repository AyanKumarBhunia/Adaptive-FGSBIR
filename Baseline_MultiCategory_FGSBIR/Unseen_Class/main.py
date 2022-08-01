import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Meat Learning for Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='sketchy-FGSBIR')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--root_dir', type=str, default='./../../Dataset/sketchy/')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=200)
    parser.add_argument('--print_freq_iter', type=int, default=10)

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)


    model = FGSBIR_Model(hp)
    model.to(device)
    step_count, top1, top10 = -1, 0, 0

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
                    torch.save(model.state_dict(), './model/metaFGSBIR_Model_best.pth')
                    top1, top10 = top1_eval, top10_eval

                    with open('./Results.txt', 'a') as filehandle:
                        np.savetxt(filehandle, np.array(accuracy_list), fmt='%s',
                                   newline='\n', header='Iteration Number' + str(step_count) + ':', footer='\n')
                    print('Model Updated')

        print ('Epoch time : ', time.time()-start)