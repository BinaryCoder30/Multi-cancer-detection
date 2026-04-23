import torch
import torchvision.models as models
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_model(path, num_classes, hidden_units=256):
    print(f"Checking {path}")
    model = models.efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, num_classes)
    )
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print("Success for num_classes =", num_classes, "hidden_units =", hidden_units)
    except Exception as e:
        print("Failed:", e)

check_model('/Users/krishdhamecha/Downloads/multi_cancer_webapp/organ_model.pth', 2, 128)
check_model('/Users/krishdhamecha/Downloads/multi_cancer_webapp/colon_model.pth', 2, 256)
check_model('/Users/krishdhamecha/Downloads/multi_cancer_webapp/lung_model.pth', 3, 256)

