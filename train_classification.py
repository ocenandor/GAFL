import os

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from utils_clf import get_params, parse_args

from data.dataset import ClassificationDataset, data_loaders

from models.classification import ResNet

from training.utils import make_reproducible, print_model_info
from training.model_training_clf import train
from training.metrics import *

from tqdm import tqdm

#from nip import load



# for printing
torch.set_printoptions(precision=2)

# for reproducibility
make_reproducible(seed=0)


def main(args, params):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    #params = get_params()
    
    params.update({'random_seed': args.random_seed})

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=params['image_size']),
                                                torchvision.transforms.ToTensor()])
    train_dataloader, val_dataloader = data_loaders(dataset=ClassificationDataset,
                                                    train_transform=transform,
                                                    val_transform=transform,
                                                    params=params)

    model = ResNet(n_channels=1, n_classes=params['n_classes'],
                   blocks=params['blocks'], filters=params['filters'],
                   image_size=params['image_size'], adaptive_layer_type=params['adaptive_layer'],
                   multi_gafl=params['multi_gafl'],
                   ).to(device)
    print_model_info(model, params)

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1/0.56472, 1/0.26649, 1/0.168781]).to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    #metrics = {'F1Score': F1Score()}
    metrics = {
        #'F1Score': F1Score(),
        #'F1Score_nb': F1Score_nb(),
        'F1Score_NvsB': F1Score_NvsB(),
        'F1Score_BvsM': F1Score_BvsM(),
        #'F1Score_NvsM': F1Score_NvsM(),
    }

    writer = None
    if not args.nolog:
        print(params['log_dir'])
        writer = SummaryWriter(log_dir=os.path.join(params['log_dir'], model.name))
        print("To see the learning process, use command in the new terminal:\n" +
              "tensorboard --logdir <path to log directory>")
        #print()

    train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metrics,
          n_epochs=params['n_epochs'],
          device=device,
          writer=writer)
   

if __name__ == "__main__":
    params = get_params()
    
    #ADAPTIVE_LAYER_ITER = ('spectrum', 'spectrum_log', 'phase' , 'general_spectrum')
    #MULTI_GAFL_ITER = ([0], [1], [2], [3],  [0,1], [0,2], [0,3], [1,2], [1,3], [2,3],  [0,1,2], [0,1,3], [1,2,3], [0,2,3],  [0,1,2,3])
    
    ADAPTIVE_LAYER_ITER = ('None',)
    MULTI_GAFL_ITER = ([],)
    
    # override params['adaptive_layer'] and params['multi_gafl']
    for adaptive_layer_ in tqdm(ADAPTIVE_LAYER_ITER) :
        for multi_gafl_ in tqdm(MULTI_GAFL_ITER):
    
            params.update({'multi_gafl': [multi_gafl_, []] })

            params.update({'adaptive_layer': adaptive_layer_})
            if params['adaptive_layer'] == 'None':
                params['adaptive_layer'] = None
                
            #print(params)
        
            main(parse_args(), params)
