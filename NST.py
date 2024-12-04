import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class NeuralStyleTransfer:
    def __init__(self, content_image_path, style_image_path):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image transformations
        self.loader = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
        # Load images
        self.content_image = self.load_image(content_image_path)
        self.style_image = self.load_image(style_image_path)
        self.generated_image = self.content_image.clone().requires_grad_(True)
        
        # Initialize VGG model
        self.model = self.VGG().to(self.device).eval()
        
        # Hyperparameters
        self.alpha = 8  # Content weight
        self.beta = 70  # Style weight
        self.lr = 0.004  # Learning rate
        self.epochs = 7000

    def load_image(self, path):
        """Load and preprocess an image"""
        image = Image.open(path).convert('RGB')
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    class VGG(nn.Module):
        """Custom VGG model for extracting specific layers' features"""
        def __init__(self):
            super().__init__()
            self.req_features = ['0', '5', '10', '19', '28']  # Selected layers
            self.model = models.vgg19(pretrained=True).features[:29]

        def forward(self, x):
            features = []
            for layer_num, layer in enumerate(self.model):
                x = layer(x)
                if str(layer_num) in self.req_features:
                    features.append(x)
            return features
    
    def calc_content_loss(self, gen_feat, orig_feat):
        """Calculate content loss"""
        return torch.mean((gen_feat - orig_feat) ** 2)

    def calc_style_loss(self, gen, style):
        """Calculate style loss using Gram matrix"""
        batch_size, channel, height, width = gen.shape
        G = torch.mm(gen.view(channel, height * width), gen.view(channel, height * width).t())
        A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())
        return torch.mean((G - A) ** 2)

    def calculate_loss(self, gen_features, orig_features, style_features):
        """Calculate total loss"""
        style_loss = content_loss = 0
        for gen, orig, style in zip(gen_features, orig_features, style_features):
            content_loss += self.calc_content_loss(gen, orig)
            style_loss += self.calc_style_loss(gen, style)
        return self.alpha * content_loss + self.beta * style_loss

    def transfer_style(self):
        """Perform style transfer"""
        optimizer = optim.Adam([self.generated_image], lr=self.lr)

        for e in range(self.epochs):
            # Extract features
            gen_features = self.model(self.generated_image)
            orig_features = self.model(self.content_image)
            style_features = self.model(self.style_image)
            
            # Calculate loss
            total_loss = self.calculate_loss(gen_features, orig_features, style_features)
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Print and save progress
            if e % 100 == 0:
                print(f"Epoch {e}, Loss: {total_loss.item()}")
                save_image(self.generated_image, f"generated_epoch_{e}.png")
        
        return self.generated_image
    
    def display_result(self, generated_img):
        """Display the content, style, and generated images"""
        img = generated_img.squeeze(0).cpu().detach()
        img = img.clamp(0, 1)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(131)
        plt.title("Content Image")
        plt.imshow(self.content_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        plt.subplot(132)
        plt.title("Style Image")
        plt.imshow(self.style_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        plt.subplot(133)
        plt.title("Generated Image")
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
