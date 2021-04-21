import io
import torch
from PIL import Image
from torchvision import transforms
from model.octmodel import OCTModel
from argparse import Namespace


oct_class_index = {
    0: 'NORMAL',
    1: 'CNV',
    2: 'DME',
    3: 'DRUSEN'
}
num_of_classes = int(len(oct_class_index))
frac_train_images = 1  
batch_size = 32 
optimizer = 'Adam' 

def get_classifier():
    suggested_lr = 0.0019054607179632484
    weight_path = 'simclr_imagenet.ckpt'
    hparams = Namespace(
        learning_rate=suggested_lr,
        freeze_base=False,
        tune=False,
        max_epochs=15,
        steps_per_epoch=1000,
        n_classes=num_of_classes,
        embeddings_path=weight_path,
        batch_size=batch_size,
        optimizer=optimizer,
        arch='resnet50',
        frac_train_images=frac_train_images
    )
    model = OCTModel(hparams)
    model.load_state_dict(torch.load('octmodel'))
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        # ImageNet Normalization
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes): 
    tensor = transform_image(image_bytes=image_bytes)
    outputs = get_classifier().forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    return predicted_idx, oct_class_index[predicted_idx]
