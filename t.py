import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# Gerät konfigurieren
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
batch_size = 64
lr = 0.0002
latent_dim = 100
num_epochs = 50
image_size = 28 * 28

# MNIST Datensatz
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, image_size),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Diskriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Instanzen und Optimierer
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Verlustfunktion
criterion = nn.BCELoss()

# Verzeichnis für generierte Bilder
if not os.path.exists('images_gan'):
    os.makedirs('images_gan')

# Hilfsfunktionen für Speichern und Laden
def save_checkpoint(generator, discriminator, epoch):
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'epoch': epoch,
    }, f'checkpoint_{epoch}.pth')

def load_checkpoint(generator, discriminator, filepath):
    if os.path.isfile(filepath):
        print(f"Lade Checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        return checkpoint['epoch']
    else:
        print("Kein Checkpoint gefunden. Starte das Training von vorne.")
        return 0

# Trainingsfunktion
def train(generator, discriminator, dataloader, epochs, save_interval):
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            # Trainiere Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Trainiere Diskriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if epoch % save_interval == 0 or epoch == epochs-1:
            save_image(gen_imgs.data[:25], f'images_gan/{epoch}.png', nrow=5, normalize=True)
            save_checkpoint(generator, discriminator, epoch)

# Startpunkt des Trainings
start_epoch = load_checkpoint(generator, discriminator, 'latest_checkpoint.pth')
train(generator, discriminator, dataloader, num_epochs - start_epoch, 10)
