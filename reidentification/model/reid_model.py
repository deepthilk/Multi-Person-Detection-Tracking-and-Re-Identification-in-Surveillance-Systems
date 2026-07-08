import torch
import torchreid
from torchvision import transforms
from PIL import Image

# Load model
def load_model():
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        pretrained=True
    )
    model.eval()
    return model

# Transform image
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Extract embedding
def extract_feature(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model(img)

    return feat.numpy()


# TEST
if __name__ == "__main__":
    model = load_model()
    img_path = "reidentification/dataset/Market-1501/bounding_box_train/1338_c2s3_032357_01.jpg"
    feat = extract_feature(img_path, model)

    print("✅ Feature extracted!")
    print("Feature shape:", feat.shape)