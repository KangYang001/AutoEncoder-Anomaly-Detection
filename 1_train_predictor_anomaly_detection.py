import time
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from pathlib import Path
# import create_data_for_deeplearning
import numpy as np
import math
from tqdm import tqdm
import os
import sys
import pickle
import glob
import sklearn.model_selection
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/PCA/autoencoder/code')

import simpleModel as Model
import configuration
import preprocess_data
import process_data


args = configuration.args
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


'''-------------------------------------------------------------------------'''
'''--------------------------- hyperparameters -----------------------------'''
'''-------------------------------------------------------------------------'''
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''-------------------------------------------------------------------------'''
'''---------------------------- build model --------------------------------'''
'''-------------------------------------------------------------------------'''

model = Model.AutoEncoder(args)

model = nn.DataParallel(model, device_ids = list(eval(args.cuda_visible_devices))).cuda()
model.to(device)
#optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
optimizer = optim.Adam(model.parameters(), lr= args.lr)
criterion = nn.MSELoss()

# Loop over epochs.
if args.resume or args.pretrained:
    print("=> loading checkpoint ")
    checkpoint_dir = Path(args.direction_workspace, 'save', 'checkpoint')
    checkpoint_dir.mkdir(parents = True, exist_ok = True)
    checkpoint = checkpoint_dir.joinpath(args.model).with_suffix('.pth')
    checkpoint = torch.load(checkpoint)
    args, start_epoch, best_val_loss = model.load_checkpoint(args, checkpoint)
    optimizer.load_state_dict((checkpoint['optimizer']))
    del checkpoint
    epoch = start_epoch
    print("=> loaded checkpoint")
else:
    epoch = 1
    start_epoch = 1
    best_val_loss = math.inf
    print("=> Start training from scratch")
print('-' * 89)
print(args)
print('-' * 89)

'''-------------------------------------------------------------------------'''
'''---------------------------- Train model --------------------------------'''
'''-------------------------------------------------------------------------'''

direction_ultrasonic_original = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", "sensors_" + str(args.sensors)).joinpath(str(args.batch_file))
list_pickle_file = glob.glob(str(direction_ultrasonic_original) + '/*.pickle')
list_pickle_file.sort()    

list_var_loss = []
for n in range(len(list_pickle_file)):
    
    with open(list_pickle_file[n] , 'rb') as handle:
        plate_ultrasonic_dataset = pickle.load(handle) 
    print(list_pickle_file[n])
    gen_dataset = plate_ultrasonic_dataset
    #process_data.generate_output(args, epoch, model, gen_dataset, criterion, device)

    dataset = plate_ultrasonic_dataset['train_data']
    num_data_T = int(len(dataset)/args.cuda_devices) * args.cuda_devices
    dataset = dataset[:num_data_T][:, 500:1500]

    dataset = preprocess_data.normalize_ultrasonic_data(dataset =  dataset, normalization_type = 0)

    data_input = dataset
    data_label = plate_ultrasonic_dataset['tag']
    
    train_input, val_input, train_label, val_label = sklearn.model_selection.train_test_split(data_input, data_label, test_size = 0.1, random_state=1)
    
    train_input = data_input
    train_label = data_label
    
    train_input = torch.from_numpy(train_input).to(device) 
    train_label  = torch.from_numpy(train_label).to(device)
                   
    train_data = torch.utils.data.TensorDataset(train_input, train_label)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = False)

    val_input = torch.from_numpy(val_input).to(device) 
    val_label  = torch.from_numpy(val_label).to(device)
                   
    val_data = torch.utils.data.TensorDataset(val_input, val_label)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, shuffle = False)

    print("\n---------------------- complete loading data ---------------------------\n")
    #validation_data.tensors[1].shape

    try:
        for epoch in range(start_epoch, args.epochs + 1):

            epoch_start_time = time.time()
            
            ''' input mode 1 '''
            # train(args, model, train_input.float().to(device), epoch)
            # val_loss = evaluate(args,model,validation_input.float().to(device))
            
            '''input mode 2 '''
            train_loss = 0
            process_control = tqdm(list(enumerate(train_loader)))
            for i, batch in process_control:

                #print("*******8\n traindataset: batch[0].transpose_(0, 1).shape",batch[0].transpose_(0, 1).shape,"\n****** i: ", len(list(enumerate(train_loader)) ) )
                cur_loss = process_data.train(args, model, optimizer, criterion, batch)
                #print("\r | current epoch ", epo, "| cur_loss ", cur_loss )
                train_loss = train_loss + cur_loss
            train_loss = train_loss / (i+1)
            process_control.set_description("| current epoch: %i | current loss %8.8f " %(epoch, train_loss))
                          
            val_loss = 0
            for i, batch in tqdm(list(enumerate(val_loader))):          
                cur_val_loss = process_data.evaluate(args, model, criterion, batch)
                val_loss = val_loss + cur_val_loss
            
            val_loss = val_loss / (i+1)
            if epoch == 1 or epoch == 21:
                list_var_loss.append(val_loss)
                      
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time), val_loss))
            print('-' * 89)
            
            if (epoch % args.save_interval) == 0:
                # Save the model if the validation loss is the best we've seen so far.
                # is_best = val_loss > best_val_loss
                # best_val_loss = max(val_loss, best_val_loss)
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)
                model_dictionary = {'epoch': epoch,
                                    'best_loss': best_val_loss,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'args':args
                                    }
                model.module.save_checkpoint(model_dictionary, is_best)
                
        process_data.generate_output(args, epoch, model, gen_dataset, criterion, device)
        start_epoch = args.epochs_resume
            
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

#args.save_fig = False
    
    list_var_loss = np.array(list_var_loss)    
    var_loss_dataset = {'list_var_loss': list_var_loss}
    
    filename_save = 'list_var_loss.pickle'
    save_dir = Path(args.direction_workspace, 'dataset', "autoencoder_corrcoeff", \
                    args.sensors + " sensors", str(args.batch_file)).joinpath(str(batch[0].shape[1])).joinpath(str("list_var_loss"))
    save_dir.mkdir(parents = True, exist_ok = True)
    path_save = Path(save_dir).joinpath(filename_save)
    
    with open(path_save, 'wb') as handle:
        pickle.dump(var_loss_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
    print(path_save, "has been created\n")  