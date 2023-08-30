import torch as t
import torch.nn as nn
import torchvision

def calculateConvDims(layer_count, input_dim, stride_dim=1, kernel_dim=3, padding=0):
    next_dim = input_dim
    for i in range(layer_count):
        next_dim = int((next_dim - kernel_dim + 2*padding) / stride_dim) + 1
        next_dim = int((next_dim - 2 + 2*padding) / 2) + 1 # Max pooling
    return next_dim

class Encoder(nn.Module):
    def __init__(self, layer_count, latent_dim, input_dim, batch_norm=False, leak=False, stride_dim=1, kernel_dim=3, flat_start=1, padding=0, dropout_odds=0.5):
        super().__init__()

        flat = calculateConvDims(layer_count=layer_count, input_dim=input_dim, stride_dim=stride_dim, kernel_dim=kernel_dim, padding=padding)
        
        blocks = []
        in_ch = 0
        out_ch = 0
        for i in range(layer_count):
            if(i > 0):
                in_ch = out_ch
                out_ch = out_ch * 2
                block = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_dim, stride=stride_dim),
                    nn.LeakyReLU if leak else nn.ReLU(),
                    nn.Dropout2d(p=dropout_odds),
                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=2, stride=2),
                    nn.LeakyReLU if leak else nn.ReLU(),
                    #nn.BatchNorm2d(out_ch),
                )
                blocks.append(block)
                if batch_norm:
                    block = nn.Sequential(
                        nn.BatchNorm2d(out_ch)
                    )
                    blocks.append(block)
                
            else: # First layer
                in_ch = 3
                out_ch = 16
                block = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_dim, stride=stride_dim),
                    nn.LeakyReLU if leak else nn.ReLU(),
                    nn.Dropout2d(p=dropout_odds),
                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=2, stride=2),
                    nn.LeakyReLU if leak else nn.ReLU(),
                    #nn.BatchNorm2d(out_ch),
                )
                blocks.append(block)

        self.model = nn.Sequential(*blocks)
        self.flatten = nn.Flatten(start_dim=flat_start)
        self.encoder_lin = nn.Sequential(
            nn.Linear(out_ch*flat*flat, out_ch*flat*flat // 4),
            nn.LeakyReLU if leak else nn.ReLU(),
            nn.Linear(out_ch*flat*flat // 4, latent_dim),
            nn.LeakyReLU if leak else nn.ReLU(),
        )

        self.out = out_ch

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
    # TODO: Verify
    def getOut(self):
        return self.out


class Decoder(nn.Module):
    def __init__(self, layer_count, latent_dim, input_dim, encoder_out_ch, unflat_start=1, batch_norm=False, leak=False, stride_dim=1, kernel_dim=3, padding=0, dropout_odds=0.5):
        super().__init__()

        #self.encoder = encoder
        flat = calculateConvDims(layer_count=layer_count, input_dim=input_dim, stride_dim=stride_dim, kernel_dim=kernel_dim, padding=padding)

        self.input_dim = input_dim
        blocks = []
        in_ch = encoder_out_ch #encoder.getOut()
        out_ch = in_ch // 2

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, flat*flat*in_ch // 4),
            nn.LeakyReLU if leak else nn.ReLU(),
            nn.Linear(flat*flat*in_ch // 4, flat*flat*in_ch),
            nn.LeakyReLU if leak else nn.ReLU(),
        )
        self.unflatten = nn.Unflatten(dim=unflat_start, unflattened_size=(in_ch, flat, flat)) # TODO: Create function that calculates image dimensions after convolutions. Take layer_count as a parameter.

        for i in range(layer_count):
            if(i != layer_count - 1):
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=2, stride=2), # FAKE MAX POOLING
                    nn.LeakyReLU if leak else nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_dim, stride=stride_dim),
                    nn.LeakyReLU if leak else nn.ReLU(),
                    nn.Dropout2d(p=dropout_odds),
                )
                blocks.append(block)
                if batch_norm:
                    block = nn.Sequential(
                        nn.BatchNorm2d(out_ch)
                    )
                    blocks.append(block)
                
                in_ch = out_ch
                out_ch = in_ch // 2
            else:
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=2, stride=2), # FAKE MAX POOLING
                    nn.LeakyReLU if leak else nn.ReLU(),
                    #nn.BatchNorm2d(in_ch),
                    nn.ConvTranspose2d(in_channels=in_ch, out_channels=3, kernel_size=kernel_dim, stride=stride_dim),
                    nn.LeakyReLU if leak else nn.ReLU(),
                    nn.Dropout2d(p=dropout_odds),
                )
                blocks.append(block)

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.model(x)
        x = torchvision.transforms.Resize((self.input_dim, self.input_dim))(x)
        
        return x
    
if __name__ == "__main__":
    print(calculateConvDims(3, 256))