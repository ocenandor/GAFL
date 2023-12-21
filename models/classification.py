import torch
import torch.nn.functional as TF

from models.adaptive_layer import AdaptiveLayer, GeneralAdaptiveLayer
from models.resnet_blocks import ResBlock


class ResNet(torch.nn.Module):

    @staticmethod
    def get_adaptive_layer(n_channels, image_size, adjustment):
        if adjustment in ('spectrum', 'spectrum_log', 'phase'):
            return AdaptiveLayer((n_channels, ) + image_size,
                                 adjustment=adjustment)
        elif adjustment == 'general_spectrum':
            return GeneralAdaptiveLayer((n_channels, ) + image_size,
                                        adjustment=adjustment,
                                        activation_function_name='relu')
                                        
    def __init__(self, n_channels, n_classes, 
        blocks, filters, image_size, 
        adaptive_layer_type=None,
        multi_gafl=[[0],]
        ):
        super(ResNet, self).__init__()

        image_size_flow = image_size
        
        self.name = 'ResNet'
        if adaptive_layer_type is not None:
            #self.name = '_'.join([self.name, 'adaptive', adaptive_layer_type])
            self.name = '_'.join([self.name, 'adaptive', str(multi_gafl), adaptive_layer_type])

        
        self.adaptive_layer_type = adaptive_layer_type
        
        # init adaptive_layer
        self.adaptive_layer = None
        if ( 0 in multi_gafl[0]) and (self.adaptive_layer_type is not None):
            self.adaptive_layer = self.get_adaptive_layer(
                n_channels,
                image_size_flow,
                adjustment=self.adaptive_layer_type,
            )
            """
            if self.adaptive_layer_type in ('spectrum', 'spectrum_log', 'phase'):
                self.adaptive_layer = AdaptiveLayer(
                    (n_channels, ) + image_size,  # model = ResNet(n_channels=1, -> (1,256,256)
                    adjustment=self.adaptive_layer_type
                )
            elif self.adaptive_layer_type == 'general_spectrum':
                self.adaptive_layer = GeneralAdaptiveLayer(
                    (n_channels, ) + image_size,
                    adjustment=self.adaptive_layer_type,
                    activation_function_name='relu'
                )
            """

        # image size 256->64 
        self.init_conv = torch.nn.Sequential(torch.nn.Conv2d(n_channels, filters[0],
                                                             kernel_size=7, stride=2, padding=3, bias=False),
                                             torch.nn.BatchNorm2d(filters[0]),
                                             torch.nn.ReLU(inplace=True),
                                             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        image_size_flow = tuple(map(lambda x: x//4, image_size_flow))
            
        """
        multi_gafl is added here
        """
        self.encoder = torch.nn.ModuleList()
        for i, num_layers in enumerate(blocks):
            if i == 0:
                if ( 1 in multi_gafl[0]) and (self.adaptive_layer_type is not None):
                    self.encoder.append(self.get_adaptive_layer(
                        filters[i],
                        image_size_flow,
                        adjustment=self.adaptive_layer_type,
                    ))
                # no change in image size
                self.encoder.append(ResBlock(num_layers=num_layers,
                                             num_input_features=filters[i], num_features=filters[i],
                                             downsampling=False))
            else:
                if ( (i+1) in multi_gafl[0]) and (self.adaptive_layer_type is not None):
                    self.encoder.append(self.get_adaptive_layer(
                        filters[i - 1],
                        image_size_flow,
                        adjustment=self.adaptive_layer_type,
                    ))
                
                # image size down x2 
                self.encoder.append(ResBlock(num_layers=num_layers,
                                             num_input_features=filters[i - 1], num_features=filters[i],
                                             downsampling=True))
                image_size_flow = tuple(map(lambda x: x//2, image_size_flow))

        self.fc = torch.nn.Linear(filters[-1], n_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print('image')
        #print(x.shape)
        #print(x)
        
        if (self.adaptive_layer is not None) and (self.adaptive_layer_type is not None):
            x = self.adaptive_layer(x)
        #print('adaptive_layer')
        #print(x.shape)
        #print(x)

        x = self.init_conv(x)
        #print('init_conv')
        #print(x.shape)
        #print(x)

        for layer in self.encoder:
            x = layer(x)
            #print('layer in self.encoder')
            #print(x.shape)
            #print(x)


        x = TF.avg_pool2d(x, x.size()[2:])
        #print("TF.avg_pool2d(x, x.size()[2:])")
        #print(x.shape)
        #print(x)

        x = self.fc(x.view(x.size(0), -1))
        #print('Linear(filters[-1], n_classes)')
        #print(x.shape,'\n')
        #print(x)

        return x
        

