import torchvision.models as models
from collections import Counter

# Load VGG-19 model
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Extract layers from features and classifier
feature_layer_types = [type(layer).__name__ for layer in vgg.features]
classifier_layer_types = [type(layer).__name__ for layer in vgg.classifier]

# Combine all layers and count types
all_layers = feature_layer_types + classifier_layer_types
layer_counts = Counter(all_layers)

# Total number of layers
total_layers = len(feature_layer_types) + len(classifier_layer_types)

# Print the details
print("VGG-19 Layer Breakdown:")
for layer_type, count in layer_counts.items():
    print(f"{layer_type}: {count}")
print(f"Total Layers: {total_layers}")

# import torchvision.models as models

# # Load the VGG-19 model
# vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# # Print the layers
# print("VGG-19 Layers:")
# for idx, layer in enumerate(vgg.features):
#     print(f"Layer {idx}: {layer}")
