import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import Loader
from tqdm import tqdm
from model import EncoderLR, EncoderHR, DecoderHR, VGGLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 10
latent_dim = 128

data_set = Loader(r"data")
train_loader = DataLoader(dataset=data_set, num_workers=0, batch_size=32, shuffle=True)

encoder_lr = EncoderLR(latent_dim).to(device)
encoder_hr = EncoderHR(latent_dim).to(device)
inference = DecoderHR(latent_dim).to(device)
vgg = VGGLoss().to(device)
mse_loss = nn.MSELoss().to(device)
optimizer = optim.Adam(list(encoder_lr.parameters()) + list(encoder_hr.parameters()) + list(inference.parameters()), lr=0.001)

def gram_matrix(input):

        a, b, c, d = input.size()
        features = input.view(a * b, c * d) 

        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)


for epoch in range(1, num_epochs + 1):
    progress_bar = tqdm(enumerate(train_loader), total = len(train_loader))
    epoch_loss = 0.0
    for i, batch in progress_bar:

        img_normal, img_degraded = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        z_lr, mu, log_sigma = encoder_lr(img_degraded)
        z_hr = encoder_hr(img_normal)
        reconstructed_img = inference(z_hr, z_lr)

        vgg_reconstructed = vgg(reconstructed_img)
        vgg_original = vgg(img_normal)

        style_loss = 0.0
        for feature_a, feature_b in zip(vgg_original, vgg_reconstructed):

            gram_a = gram_matrix(feature_a)
            gram_b = gram_matrix(feature_b)

            style_loss += mse_loss(gram_a, gram_b)

        content_loss = mse_loss(vgg_original[-1], vgg_reconstructed[-1])

        reconstruction_loss = content_loss + 0.1 * style_loss
        kl_divergence_loss = -0.5 * torch.sum(1 + log_sigma - torch.pow(mu, 2) - torch.exp(log_sigma))
        total_loss = reconstruction_loss + 0.01 * kl_divergence_loss

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        progress_bar.set_description(f"[{epoch}/{num_epochs}][{i + 1}/{len(train_loader)}] "
                                     f"Reconstruction loss: {reconstruction_loss.item():.4f} "
                                     f"KL loss: {kl_divergence_loss.item():.4f}") 
    
    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(train_loader)}") 

    torch.save({
        'encoder_lr_state_dict': encoder_lr.state_dict(),
        'encoder_hr_state_dict': encoder_hr.state_dict(),
        'inference_state_dict': inference.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'trained_model{epoch}.pth')
