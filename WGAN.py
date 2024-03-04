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
lr = 0.00005
latent_dim = 100
n_critic = 5
clip_value = 0.01  # Weight clipping value
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

# Kritiker (Diskriminator im WGAN)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Instanzen und Optimierer
generator = Generator().to(device)
critic = Critic().to(device)

optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_C = optim.RMSprop(critic.parameters(), lr=lr)

# Verzeichnis für generierte Bilder
if not os.path.exists('images_wgan'):
    os.makedirs('images_wgan')

# Trainingsfunktion
def train():
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Echte Bilder
            real_imgs = imgs.to(device)

            # ---------------------
            #  Train Critic
            # ---------------------
            optimizer_C.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.size(0), latent_dim).to(device)

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_C = -torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))
            loss_C.backward()
            optimizer_C.step()

            # Clip weights of critic
            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(critic(gen_imgs))
                loss_G.backward()
                optimizer_G.step()

        # Save Images
        if epoch % 10 == 0:
            save_image(gen_imgs.data[:25], f'images_wgan/{epoch}_fake.png', nrow=5, normalize=True)

        # Save model checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), 'generator_wgan.pth')
            torch.save(critic.state_dict(), 'critic_wgan.pth')

        print(f"Epoch {epoch+1}/{num_epochs} | D Loss: {loss_C.item()} | G Loss: {loss_G.item()}")

if __name__ == '__main__':
    train()
