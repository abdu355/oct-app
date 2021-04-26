import io
import torch
from PIL import Image
from torchvision import transforms
from model.octmodel import OCTModel
from argparse import Namespace

#gradcam
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
import numpy as np


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

#gradcam
vis_types = ["original_image", "blended_heat_map"]
vis_signs = ["all", "all"]  # "positive", "negative", or "all" to show both

# center_crop = transforms.Compose([
#  transforms.Resize((244, 244)),
#  transforms.Grayscale(num_output_channels=3),
# ])

def get_classifier():
    suggested_lr = 0.0019054607179632484
    weight_path = 'model_files/simclr_imagenet.ckpt'
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
    model.load_state_dict(torch.load('model_files/octmodel'))
    model.eval()
    return model


def get_model():
    suggested_lr = 0.0019054607179632484
    weight_path = 'model_files/simclr_imagenet.ckpt'
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
    model.load_state_dict(torch.load('model_files/octmodel'))
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

def crop_image(image_bytes):
    center_crop = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.Grayscale(num_output_channels=3),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return center_crop(image)



def get_prediction(image_bytes): 
    tensor = transform_image(image_bytes=image_bytes)
    outputs = get_classifier().forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    return predicted_idx, oct_class_index[predicted_idx]

def get_gradcam(image_bytes):
    class_id, _ = get_prediction(image_bytes=image_bytes)
    print(class_id)
    # start layer gradcam class
    model = get_classifier()
    layer_gc = LayerGradCam(model,
                            model.base_model.encoder.layer4[-1])
    
    # print(get_classifier().base_model.encoder.layer4[-1])
    # pass image through layer gradcam
    attribution = layer_gc.attribute(transform_image(image_bytes=image_bytes), int(class_id),
                                     attribute_to_layer_input=False,
                                     relu_attributions=True)
   
    # print(transform_image(image_bytes=image_bytes))
    
    # interpolate attributions from layer gradcam
    upsampled_attr = LayerAttribution.interpolate(attribution, (244, 244))
    upsampled_attr = np.transpose(upsampled_attr.squeeze(0).detach().numpy(),
                                  (1, 2, 0))
    # visualize image and attibutes
    gradcam_result = viz.visualize_image_attr_multiple(upsampled_attr,
                                          np.array(crop_image(image_bytes)),
                                          vis_types,
                                          vis_signs,
                                          ['original image: ',
                                              "attribution for " + oct_class_index[class_id]],
                                          show_colorbar=True
                                          )
    return gradcam_result