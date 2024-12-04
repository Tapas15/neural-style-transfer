# -*- coding: utf-8 -*-

# Importing required libraries
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image

# Define the VGG model for extracting features
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:29]
    
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.req_features:
                features.append(x)
        return features

# Define helper functions
def image_loader(path):
    image = Image.open(path).convert("RGB")
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def calc_content_loss(gen_feat, orig_feat):
    return torch.mean((gen_feat - orig_feat) ** 2)

def calc_style_loss(gen, style):
    B, C, H, W = gen.shape
    G = torch.mm(gen.view(C, H * W), gen.view(C, H * W).t())
    A = torch.mm(style.view(C, H * W), style.view(C, H * W).t())
    return torch.mean((G - A) ** 2)

def calculate_loss(gen_features, orig_features, style_features):
    style_loss = content_loss = 0
    for gen, orig, style in zip(gen_features, orig_features, style_features):
        content_loss += calc_content_loss(gen, orig)
        style_loss += calc_style_loss(gen, style)
    return alpha * content_loss + beta * style_loss

# Initialize variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_image = image_loader('Tajmahal.jpg')
style_image = image_loader('style_image.jpg')
generated_image = original_image.clone().requires_grad_(True)

model = VGG().to(device).eval()
optimizer = optim.Adam([generated_image], lr=0.004)

# Hyperparameters
alpha = 8
beta = 70
epochs = 7000

# Training Loop
for epoch in range(epochs):
    gen_features = model(generated_image)
    orig_features = model(original_image)
    style_features = model(style_image)
    
    total_loss = calculate_loss(gen_features, orig_features, style_features)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item()}")
        save_image(generated_image, f"generated_image_epoch_{epoch}.png")

# Save the final image
save_image(generated_image, "final_generated_image.png")
