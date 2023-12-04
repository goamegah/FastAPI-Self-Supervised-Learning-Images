import io
import pickle

from fastapi import FastAPI, UploadFile, File, Form
import importlib
import torch
import numpy as np
from PIL import Image
import torchvision
from pydantic import BaseModel
import base64

num_classes = 10

# instanciate API class
app = FastAPI()


def read_image_as_pil(encoded_image):
    pil_image = Image.open(io.BytesIO(encoded_image))
    return pil_image


def load_model(model_name):
    model_artefact_path = f'artefacts/{model_name}/model.pt'
    module_name = f'artefacts.{model_name}.model'

    if model_name == "SimCLR" or model_name == "ResNet18":
        # module_name = f'artefacts.ResNet18.model'
        model_module = importlib.import_module(module_name)
        resnet_class = getattr(model_module, 'ResNet18')
        block_class = getattr(model_module, 'BasicBlock')
        model_instance = resnet_class(num_layers=18, block=block_class, num_classes=num_classes, grayscale=True)
    else:
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, model_name)
        model_instance = model_class(num_classes, grayscale=True)

    # Now we instanciate model class
    model_instance.load_state_dict(state_dict=torch.load(f=model_artefact_path))

    return model_instance


def load_eval(model_name):
    model_artefact_path = f'artefacts/{model_name}/{model_name}_summary.pkl'
    with open(file=model_artefact_path, mode='rb') as fp:
        model_summary = pickle.load(fp)
        print(f'{model_name} dictionary |> type: {type(model_summary)}')
    return model_summary


def process_pil_image_to_tensor(image):
    # Resize image to 32x32
    img = image.resize((32, 32))

    # Apply transformation using torchvision modeul
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1), # to be sure to have one channel
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))  # Normalisation (mean and standard deviation)
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # add one dimension for batch dimension
    # print(f'==+> {img_tensor.shape}')

    return img_tensor


def process_image_numpy(image):
    resize_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))])

    # Open image with Pillow
    with Image.open(image) as img:
        transformed_image = resize_transform(img)

        # Convert image to numpy array
        img_array = np.array(img)

        # be sure to have only one channel (gray scale)
        if len(img_array.shape) == 2:  # whether image is already in gray scale format
            img_array = np.expand_dims(img_array, axis=0)
        else:  # Whether image has 3 channels
            # compute mean of all channel
            img_array = np.expand_dims(np.mean(img_array, axis=2), axis=0)  # Transform into gray scale format

        # transform to tensor array
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)  # Ajoutez une dimension pour le batch

        # Normalize value to belong to range (0, 1)
        img_tensor /= 255.0

        return img_tensor


@app.get(path="/")
def welcome():
    return {"message": "welcome to SsIma API"}


@app.post("/evaluation")
def evaluation(model_name: str = Form(...)):
    eval_loaded = load_eval(model_name)
    print("Item: \n ")
    print(eval_loaded["minibatch_loss_list"])

    # Important thing here!!! Json allow only native type inside dict
    # to do so we need to transform confusion matrix (ndarray) to list
    return \
        {
            "minibatch_loss_list": eval_loaded['minibatch_loss_list'],
            "train_acc_list": eval_loaded['train_acc_list'],
            "valid_acc_list": eval_loaded['valid_acc_list'],
            "confusion_matrix": eval_loaded['confusion_matrix'].tolist(),
            "num_epochs": eval_loaded['num_epochs'],
            "iter_per_epoch": eval_loaded['iter_per_epoch'],
            "averaging_iterations": eval_loaded['averaging_iterations']
        }


@app.post("/prediction")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    # get encoded image step
    image_file = await file.read()
    pil_image = read_image_as_pil(image_file)

    # process image step
    image_tensor = process_pil_image_to_tensor(pil_image)

    # model loading step
    model_loaded = load_model(model_name)

    # inference step
    model_loaded.eval()
    logits = model_loaded(image_tensor)
    prediction = torch.argmax(input=logits)

    return {"prediction": prediction.item()}




