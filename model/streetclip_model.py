import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("geolocal/StreetCLIP").half()
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

model = model.to("cpu")
model.eval()
with open("model/location_tree.json", "r") as f:
    LOCATION_TREE = json.load(f)


def predict_label(image, labels):
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image.squeeze(0)
        probs = logits.softmax(dim=0)
    best_idx = probs.argmax().item()
    return labels[best_idx], probs[best_idx].item()


def predict_location_hierarchical(image_path):
    image = Image.open(image_path).convert("RGB")

    country_labels = list(LOCATION_TREE.keys())
    country, conf_country = predict_label(image, country_labels)

    city_labels = LOCATION_TREE[country]["cities"]
    city, conf_city = predict_label(image, city_labels)

    landmark = None
    conf_landmark = None
    if city in LOCATION_TREE[country]["landmarks"]:
        landmark_labels = LOCATION_TREE[country]["landmarks"][city]
        landmark, conf_landmark = predict_label(image, landmark_labels)

    return {
        "country": country,
        "country_conf": conf_country,
        "city": city,
        "city_conf": conf_city,
        "landmark": landmark,
        "landmark_conf": conf_landmark
    }
