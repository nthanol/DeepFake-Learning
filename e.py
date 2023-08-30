import torch
from torch import nn

class AutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            #nn.LeakyReLU(),
            #nn.BatchNorm2d(128),
            #nn.MaxPool2d(3, stride=2),

            #nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            #nn.LeakyReLU(),
            #nn.BatchNorm2d(256),
            #nn.MaxPool2d(3, stride=2, return_indices=True),

        )

        self.flatten = nn.Flatten()

        self.encoder_lin = nn.Sequential(
            nn.Linear(1032256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )

        self.decoder_lin = nn.Sequential(    
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1032256),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 127, 127))

        self.unpool = nn.MaxUnpool2d(2, stride=2)

        self.decoder = nn.Sequential(
            #nn.MaxUnpool2d(3, stride=2),
            #nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
            #nn.BatchNorm2d(256),
            #nn.ReLU(),

            #nn.MaxUnpool2d(3, stride=2),
            #nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(),
            #nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1),
            #nn.BatchNorm2d(),
            nn.ReLU(),
        )

    def forward(self, x):
        x, indices = self.encoder(x)

        x = self.flatten(x) # Flatten
        z = self.encoder_lin(x)
        z = self.decoder_lin(z)
        z = self.unflatten(z)

        z = self.unpool(z, indices)
        z = self.decoder(z)
        #z = z.view(-1, self.fc_output_dim, 1, 1) # hopefully reshapes it back into images
        z = torch.sigmoid(z)
        return z
    
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoEncoder(100).to(device)

    loss_fn = torch.nn.MSELoss()
    lr= 0.001
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)

    model.train()

