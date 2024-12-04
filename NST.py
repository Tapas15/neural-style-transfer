import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class NeuralStyleTransfer:
    def __init__(self, content_image_path, style_image_path):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image transformations
        self.loader = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess images
        self.content_img = self.load_image(content_image_path)
        self.style_img = self.load_image(style_image_path)
        
        # Initialize VGG model
        self.model = self.get_vgg_model()
        
    def load_image(self, image_path):
        """Load and transform image"""
        image = Image.open(image_path).convert('RGB')
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device)
    
    def get_vgg_model(self):
        """Prepare VGG model for style transfer"""
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device)
        
        # Freeze model parameters
        for param in vgg.parameters():
            param.requires_grad_(False)
        
        return vgg
    
    def gram_matrix(self, input):
        """Compute Gram matrix for style representation"""
        batch_size, n_channels, height, width = input.size()
        features = input.view(batch_size * n_channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * n_channels * height * width)
    
    def content_loss(self, generated_features, content_features):
        """Compute content loss"""
        return torch.nn.functional.mse_loss(generated_features, content_features)
    
    def style_loss(self, generated_features, style_features):
        """Compute style loss using Gram matrix"""
        G_gen = self.gram_matrix(generated_features)
        G_style = self.gram_matrix(style_features)
        return torch.nn.functional.mse_loss(G_gen, G_style)
    
    def transfer_style(self, 
                       content_weight=1e3, 
                       style_weight=1e6, 
                       num_steps=300):
        """Main style transfer algorithm"""
        # Clone content image as our generated image
        generated_img = self.content_img.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = optim.LBFGS([generated_img])
        
        # Content and style layers (as in the paper)
        content_layers = ['4']  # Equivalent to conv4_2 in the feature map
        style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, ..., conv5_1
        
        # Extract features for content and style images
        content_features = {}
        style_features = {}
        
        x = self.content_img
        y = self.style_img
        for name, layer in self.model._modules.items():
            x = layer(x)
            y = layer(y)
            
            if name in content_layers:
                content_features[name] = x
            
            if name in style_layers:
                style_features[name] = y
        
        def style_transfer_step():
            """Closure function for LBFGS optimizer"""
            # Zero gradients
            optimizer.zero_grad()
            
            # Reset feature storage
            generated_features = {}
            
            # Forward pass through the model
            x = generated_img
            for name, layer in self.model._modules.items():
                x = layer(x)
                
                if name in content_layers or name in style_layers:
                    generated_features[name] = x
            
            # Compute losses
            total_content_loss = 0
            total_style_loss = 0
            
            # Content loss
            for layer in content_layers:
                total_content_loss += self.content_loss(
                    generated_features[layer], content_features[layer]
                )
            
            # Style loss
            for layer in style_layers:
                total_style_loss += self.style_loss(
                    generated_features[layer], style_features[layer]
                )
            
            # Total loss
            total_loss = content_weight * total_content_loss + style_weight * total_style_loss
            
            # Backpropagate
            total_loss.backward()
            
            return total_loss
        
        # Optimization loop
        for _ in range(num_steps):
            optimizer.step(style_transfer_step)
        
        return generated_img
    
    def display_result(self, generated_img):
        """Denormalize and display result"""
        img = generated_img.squeeze(0).cpu().detach()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.clamp(0, 1)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(131)
        plt.title('Content Image')
        plt.imshow(self.content_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        plt.subplot(132)
        plt.title('Style Image')
        plt.imshow(self.style_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        plt.subplot(133)
        plt.title('Generated Image')
        plt.imshow(img.permute(1, 2, 0).numpy())
        
        plt.show()

# Example usage
def main():
    content_path = 'content_image.jpg'
    style_path = 'style_image.jpg'
    
    style_transfer = NeuralStyleTransfer(content_path, style_path)
    generated_img = style_transfer.transfer_style()
    style_transfer.display_result(generated_img)

if __name__ == '__main__':
    main()
