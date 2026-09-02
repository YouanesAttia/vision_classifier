from PIL import Image
import torch
from torchvision import transforms

def predict_image(image_path: str, model, class_names: list) -> dict:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    return {
        "class": class_names[pred_idx.item()],
        "confidence": confidence.item(),
        "all_probs": {cls: prob.item() for cls, prob in zip(class_names, probs[0])}
    }