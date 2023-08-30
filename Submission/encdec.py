import torch as t
import torch.nn as nn
import torchvision

def calculateConvDims(layer_count, input_dim, stride, kernel_dim):
    """
    In the event that all convolutional layers have the same parameters, calculates the dimensions of the final output. 
    """
    next_dim = input_dim
    pad = []

    for i in range(layer_count):
        padding = 0
        numerator = (next_dim - kernel_dim + 2*padding)
        if numerator % stride != 0:
            padding += 1
            numerator = (next_dim - kernel_dim + 2*padding)
            pad.append(padding)
        next_dim = int(numerator / stride) + 1
    return next_dim, pad

class Encoder(nn.Module):
    def __init__(self, layer_count, latent_dim, input_dim, leak=True, stride_dim=2, kernel_dim=3, dropout_odds=0.5):
        super().__init__()

        flat, pad = calculateConvDims(layer_count, input_dim, stride_dim, kernel_dim)
        padding = 0
        blocks = []
        in_ch = 0
        out_ch = 0
        for i in range(layer_count):
            padding = pad[i]
            if(i > 0):
                in_ch = out_ch
                out_ch = out_ch * 2
                block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_dim, stride_dim, padding),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU() if leak else nn.ReLU(),
                    nn.Dropout2d(p=dropout_odds),
                )
                blocks.append(block)
                
            else: # First layer
                in_ch = 3
                out_ch = 64
                block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_dim, stride_dim, padding),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU() if leak else nn.ReLU(),
                    nn.Dropout2d(p=dropout_odds),
                    #nn.BatchNorm2d(out_ch),
                )
                blocks.append(block)

        # Used for downsampling channels
        blocks.append(nn.Sequential(
            nn.Conv2d(out_ch, out_ch//4, 1, 1),
            nn.LeakyReLU(),
        ))
        out_ch = out_ch//4

        self.model = nn.Sequential(*blocks)
        self.flatten = nn.Flatten() # x.view(x.size(0), -1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(out_ch*flat*flat, out_ch*flat*flat // 4),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Linear(out_ch*flat*flat // 4, latent_dim),
            nn.LeakyReLU() if leak else nn.ReLU(),
        )

        self.out = out_ch # The number of output channels of the encoder's final convolutional layer

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
    # TODO: Verify
    def getOut(self):
        return self.out


class Decoder(nn.Module):
    def __init__(self, layer_count, latent_dim, input_dim, encoder_out_ch, leak=True, stride_dim=2, kernel_dim=3, dropout_odds=0.5):
        super().__init__()

        #self.encoder = encoder
        flat, pad = calculateConvDims(layer_count, input_dim, stride_dim, kernel_dim)

        self.input_dim = input_dim
        blocks = []
        in_ch = encoder_out_ch #encoder.getOut()
        out_ch = in_ch // 2

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, flat*flat*in_ch // 4),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Linear(flat*flat*in_ch // 4, flat*flat*in_ch),
            nn.LeakyReLU() if leak else nn.ReLU(),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(in_ch, flat, flat)) # TODO: Create function that calculates image dimensions after convolutions. Take layer_count as a parameter.

        blocks.append(nn.Sequential(
            nn.Conv2d(in_ch, in_ch*4, 1, 1),
            nn.LeakyReLU(),
        ))
        in_ch = in_ch * 4
        out_ch = in_ch // 2

        for i in range(layer_count):
            #print(pad)
            padding = 1#pad[i]
            if(i != layer_count - 1):

                block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_dim, stride_dim, padding),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU() if leak else nn.ReLU(),
                    nn.Dropout2d(p=dropout_odds),
                )
                blocks.append(block)
                in_ch = out_ch
                out_ch = in_ch // 2

            else:
                
                block = nn.Sequential(
                    #nn.BatchNorm2d(in_ch),
                    nn.ConvTranspose2d(in_ch, 3, kernel_dim, stride_dim, padding),
                    nn.BatchNorm2d(3),
                    nn.Sigmoid(),
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
    LAYER_COUNT = 6
    INPUT = 256
    STRIDE = 2
    KERNEL = 3
    print(calculateConvDims(LAYER_COUNT, INPUT, STRIDE, KERNEL))