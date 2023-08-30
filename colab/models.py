import torch.nn as nn
import encdec as ed
import torchvision

class AutoEncoder(nn.Module):
    def __init__(self, layer_count, latent_dim, input_dim, dropout_odds=0.5):
        super().__init__()
        self.encoder = ed.Encoder(layer_count, latent_dim, input_dim, dropout_odds=dropout_odds)
        self.decoder = ed.Decoder(layer_count, latent_dim, input_dim, self.encoder.getOut(), dropout_odds=dropout_odds)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y

    def encode(self, x):
        output = self.encoder(x)
        return output

    def decode(self, input):
        y = self.decoder(input)
        return y
    
class DeepFaker(nn.Module):
    def __init__(self, layer_count, latent_dim, stride_dim=1, kernel_dim=3, dropout_odds=0.5, flat_start=1):
        super().__init__()
        self.encoderA = ed.Encoder(layer_count, latent_dim, input_dim = 256, stride_dim=stride_dim, kernel_dim=kernel_dim, flat_start=flat_start, dropout_odds=dropout_odds)
        self.encoderB = ed.Encoder(layer_count, latent_dim, input_dim = 256, stride_dim=stride_dim, kernel_dim=kernel_dim, flat_start=flat_start, dropout_odds=dropout_odds)
        self.decoderA = ed.Decoder(layer_count, latent_dim, input_dim = 256, encoder_out_ch = self.encoder.getOut(), stride_dim=stride_dim, kernel_dim=kernel_dim, dropout_odds=dropout_odds)
        self.decoderB = ed.Decoder(layer_count, latent_dim, input_dim = 256, encoder_out_ch = self.encoder.getOut(), stride_dim=stride_dim, kernel_dim=kernel_dim, dropout_odds=dropout_odds)

    def forward(self, x):
        zA = self.encodeA(x)
        yA = self.decodeA(zA)

        zB = self.encodeA(x)
        yB = self.decodeA(zB)
        return yA, yB
    
    def encodeA(self, x):
        return self.encoderA(x)
    def encodeB(self, x):
        return self.encoderB(x)
    def decodeA(self, x):
        return self.decoderA(x)
    def decodeB(self, x):
        return self.decoderB(x)
    
class SingleEnc(nn.Module):
    def __init__(self, latent_dim, leak=True, dropout_odds=0.5, discriminator=False):
        super().__init__()
        self.discrim = discriminator
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 9, 4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(64, 128, 5, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(128, 256, 5, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(1024, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Flatten(),
        )

        self.inter = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Linear(4096, latent_dim),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Linear(latent_dim, 4096),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.LeakyReLU() if leak else nn.ReLU(),
        )

        self.decoderA = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),
            nn.ConvTranspose2d(512, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(1024, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(256, 128, 5, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(128, 64, 5, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(64, 3, 9, 4),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            nn.Dropout2d(p=dropout_odds),
        )
        self.decoderB = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),
            nn.ConvTranspose2d(512, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(1024, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(256, 128, 5, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(128, 64, 5, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(64, 3, 9, 4),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            nn.Dropout2d(p=dropout_odds),
        )
        if(discriminator):
            self.discriminatorA = nn.Sequential(
                nn.Conv2d(3, 64, 4, 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 128, 4, 4),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, 4, 4),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 512, 4, 1),
                nn.Sigmoid(),
            )
            self.discriminatorB = nn.Sequential(
                nn.Conv2d(3, 64, 4, 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 128, 4, 4),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, 4, 4),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 512, 4, 1),
                nn.Sigmoid(),
            )

    def forward(self, x, type='a'):
        x = self.encoder(x)
        x = self.inter(x)
        if type == 'a':
            x = self.decoderA(x)
            x = torchvision.transforms.Resize((256, 256))(x)
            if self.discrim:
                x = self.discriminatorA
                return x
        else:
            x = self.decoderB(x)
            x = torchvision.transforms.Resize((256, 256))(x)
            if self.discrim:
                x = self.discriminatorB
                return x
        return x
    
    def discriminator(self, x, type='a'):
        if type == 'a':
            x = self.discriminatorA(x)
        else:
            x = self.discriminatorB(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.inter(x)
        return x
    
    def decode(self, z, type='a'):
        if type == 'a':
            x = self.decoderA(z)
        else:
            x = self.decoderB(z)
        x = torchvision.transforms.Resize((256, 256))(x)
        return x

class ModifiedSingleEnc(nn.Module):
    def __init__(self, latent_dim, leak=True, dropout_odds=0.5, discriminator=False):
        super().__init__()
        self.discrim = discriminator
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(64, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(128, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(256, 512, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(512, 1024, 2, 2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(1024, 2048, 2, 2),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.Flatten(),
        )

        self.inter = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Linear(4096, latent_dim),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Linear(latent_dim, 4096),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.LeakyReLU() if leak else nn.ReLU(),
        )

        self.decoderA = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),
            nn.ConvTranspose2d(512, 2048, 1, 1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(2048, 1024, 2, 2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(64, 3, 2, 2),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            nn.Dropout2d(p=dropout_odds),
        )
        self.decoderB = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),
            nn.ConvTranspose2d(512, 2048, 1, 1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(2048, 1024, 2, 2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU() if leak else nn.ReLU(),
            nn.Dropout2d(p=dropout_odds),

            nn.ConvTranspose2d(64, 3, 2, 2),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            nn.Dropout2d(p=dropout_odds),
        )
        if(discriminator):
            self.discriminatorA = nn.Sequential(
                nn.Conv2d(3, 64, 4, 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 128, 4, 4),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, 4, 4),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 512, 4, 1),
                nn.Sigmoid(),
            )
            self.discriminatorB = nn.Sequential(
                nn.Conv2d(3, 64, 4, 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 128, 4, 4),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, 4, 4),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 512, 4, 1),
                nn.Sigmoid(),
            )

    def forward(self, x, type='a'):
        x = self.encoder(x)
        x = self.inter(x)
        if type == 'a':
            x = self.decoderA(x)
            x = torchvision.transforms.Resize((256, 256))(x)
            if self.discrim:
                x = self.discriminatorA
                return x
        else:
            x = self.decoderB(x)
            x = torchvision.transforms.Resize((256, 256))(x)
            if self.discrim:
                x = self.discriminatorB
                return x
        return x
    
    def discriminator(self, x, type='a'):
        if type == 'a':
            x = self.discriminatorA(x)
        else:
            x = self.discriminatorB(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.inter(x)
        return x
    
    def decode(self, z, type='a'):
        if type == 'a':
            x = self.decoderA(z)
        else:
            x = self.decoderB(z)
        return x

class Discriminator(nn.Module):
    '''
    Discriminator class for testing.
    '''
    def __init__(self):
        super().__init__()

    
if __name__ == "__main__":
    deepfake = SingleEnc(True)
