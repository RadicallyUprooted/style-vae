import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import Loader
from tqdm import tqdm
from model import EncoderLR, EncoderHR, Decoder, VGG, MINE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 30
latent_dim = 128
batch_size = 8

data_set = Loader(r"data")
train_loader = DataLoader(dataset=data_set, num_workers=0, batch_size=batch_size, shuffle=True)

encoder_lr = EncoderLR(latent_dim).to(device)
encoder_hr = EncoderHR(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
mine = MINE().to(device)
vgg = VGG().to(device)

vae_optim = optim.Adam(list(encoder_lr.parameters()) + list(encoder_hr.parameters()) 
                        + list(decoder.parameters()) + list(mine.parameters()), lr=1e-4)

def gram_matrix(input):

        a, b, c, d = input.size()
        features = input.view(a * b, c * d) 

        G = torch.mm(features, features.t())

        return G

for epoch in range(1, num_epochs + 1):
    progress_bar = tqdm(enumerate(train_loader), total = len(train_loader))
    epoch_loss = 0.0
    for i, batch in progress_bar:

        img_normal, img_degraded = batch[0].to(device), batch[1].to(device)
        
        z_hr = encoder_hr(img_normal)
        z_lr, mean, log_var = encoder_lr(img_degraded)
        img_reconstructed = decoder(z_hr, z_lr)

        vgg_normal = vgg(img_normal)
        vgg_degraded = vgg(img_degraded)
        vgg_reconstructed = vgg(img_reconstructed)

        style_loss = 0.0
        for feature_a, feature_b in zip(vgg_reconstructed, vgg_degraded):

            b, c, h, w = feature_a.size()
            
            gram_a = gram_matrix(feature_a)
            gram_b = gram_matrix(feature_b)

            style_loss += torch.mean((gram_a - gram_b)**2) / (c * h * w)

        content_loss = torch.mean((vgg_normal[1] - vgg_reconstructed[1])**2)
        reconstruction_loss = content_loss + 1e-2 * style_loss
        kl_divergence = -0.5 * torch.sum(1 + log_var - torch.pow(mean, 2) - torch.exp(log_var))
            
        marginal = img_reconstructed[torch.randperm(img_degraded.shape[0])]
        t = torch.mean(mine(img_degraded, img_reconstructed))
        et = torch.mean(torch.exp(mine(img_degraded, marginal) - 1))
        mi = t - et
        
        total_loss = reconstruction_loss + 1e-4 * kl_divergence - 1e-1 * mi

        vae_optim.zero_grad()
        total_loss.backward()
        vae_optim.step()

        epoch_loss += total_loss.item()
        progress_bar.set_description(f"[{epoch}/{num_epochs}][{i + 1}/{len(train_loader)}] "
                                     f"Content loss: {content_loss.item():.3f} "
                                     f"Style loss: {style_loss.item():.3f} "
                                     f"KL: {kl_divergence.item():.3f} "
                                     f"MI: {mi.item():.3f} ") 
    
    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(train_loader)}") 
    torch.save({
        'encoder_hr_state_dict': encoder_hr.state_dict(),
        'encoder_lr_state_dict': encoder_lr.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'mine_state_dict': mine.state_dict(),
        'optim_state_dict': vae_optim.state_dict()
    }, f'trained_model_{epoch}.pth')
