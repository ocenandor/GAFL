import torch
from models.adaptive_layer import AdaptiveLayer, GeneralAdaptiveLayer
from models.unet_blocks import double_conv, out_conv, down_step, up_step


class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, init_features, depth, image_size, adaptive_layer_type=None, multi_gafl=[[], []]):
        super(UNet, self).__init__()
        
        """
        To use GAFL choose adaptive_layer_type from ['spectrum', 'spectrum_log', 'phase', 'general_spectrum']
        And place where to insert it (from 0 to depth - 1): multi_gafl = [[...], []] first list for downstream, second for upstream (in development)
        """

        self.name = 'UNet'
        if adaptive_layer_type is not None:
            self.name = '_'.join([self.name, 'adaptive', str(multi_gafl), adaptive_layer_type])

        self.features = init_features
        self.depth = depth

        self.adaptive_layer_type = adaptive_layer_type

        self.down_path = torch.nn.ModuleList()
        if 0 in multi_gafl[0] and self.adaptive_layer_type is not None:
            self.down_path.append(self.get_adaptive_layer(n_channels, image_size, self.adaptive_layer_type))
        self.down_path.append(double_conv(n_channels, self.features, self.features))
        for i in range(1, self.depth):
            if 0 in multi_gafl[0] and self.adaptive_layer_type is not None:
                self.down_path.append(self.get_adaptive_layer(self.features,
                                                              (image_size[0] // 2 ** (i - 1), image_size[1] // 2 ** (i - 1)),
                                                              self.adaptive_layer_type))  
            self.down_path.append(down_step(self.features, 2 * self.features))
            self.features *= 2

        self.up_path = torch.nn.ModuleList()
        for i in range(1, self.depth):
            self.up_path.append(up_step(self.features, self.features // 2))
            self.features //= 2
        self.out_conv = out_conv(self.features, n_classes)

    def forward_down(self, input):
        downs = [input]
        idx_conv = []
        for down_step in self.down_path:
            if type(down_step) != AdaptiveLayer and type(down_step) != GeneralAdaptiveLayer:
                idx_conv.append(len(downs))
            downs.append(down_step(downs[-1]))

        return downs, idx_conv

    def forward_up(self, downs, idx_conv):
        current_up = downs[-1]
        for i, up_step in enumerate(self.up_path):
            current_up = up_step(current_up, downs[idx_conv[-i - 2]])

        return current_up

    def forward(self, x):
        downs, idx_conv = self.forward_down(x)
        up = self.forward_up(downs, idx_conv)

        return self.out_conv(up)
    
    @staticmethod
    def get_adaptive_layer(n_channels, image_size, adjustment):
        if adjustment in ('spectrum', 'spectrum_log', 'phase'):
            return AdaptiveLayer((n_channels, ) + image_size,
                                 adjustment=adjustment)
        elif adjustment == 'general_spectrum':
            return GeneralAdaptiveLayer((n_channels, ) + image_size,
                                        adjustment=adjustment,
                                        activation_function_name='relu')