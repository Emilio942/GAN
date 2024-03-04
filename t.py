import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# Überprüfen, ob ein CUDA-fähiges Gerät verfügbar ist, um das Training zu beschleunigen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
latent_dim = 100
batch_size = 64
learning_rate = 0.0002
image_size = 28*28
num_epochs = 50

# MNIST Datensatz laden
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Generator definieren
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# Diskriminator definieren
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Instanzen von Generator und Diskriminator erstellen
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Zustand des Generators und Diskriminators laden, falls vorhanden
if os.path.isfile('generator_state.pth'):
    generator.load_state_dict(torch.load('generator_state.pth'))
if os.path.isfile('discriminator_state.pth'):
    discriminator.load_state_dict(torch.load('discriminator_state.pth'))

# Verlustfunktion und Optimierer
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Verzeichnis für die generierten Bilder erstellen, falls nicht vorhanden
if not os.path.exists('images'):
    os.makedirs('images')

# Trainingsschleife
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # Echte und generierte Bilder vorbereiten
        real_imgs = imgs.to(device)
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)

        # Diskriminator trainieren
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), torch.ones(imgs.size(0), 1).to(device))
        fake_loss = criterion(discriminator(fake_imgs.detach()), torch.zeros(batch_size, 1).to(device))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Generator trainieren
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_imgs), torch.ones(batch_size, 1).to(device))
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{num_epochs} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

    # Generierte Bilder nach jeder Epoche speichern
    if epoch % 10 == 0:
        save_image(fake_imgs.data[:25], f"images/{epoch}_fake.png", nrow=5, normalize=True)

    # Modellzustände speichern
    if (epoch + 1) % 50 == 0:
        torch.save(generator.state_dict(), 'generator_state.pth')
        torch.save(discriminator.state_dict(), 'discriminator_state.pth')
