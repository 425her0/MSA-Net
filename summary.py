import torch
from thop import clever_format, profile
from torchsummary import summary
from models.MSANet import MSANet

if __name__ == "__main__":
    input_shape     = [512, 512]

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSANet(in_channels=3, n_classes=21, feature_scale=1, ESCC=True, MSIC=True).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)

    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")

    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
