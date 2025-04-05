import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import torchvision.transforms as transforms
from PIL import Image
import cv2
import albumentations as A

import os
import shutil
import random

def see_dir(dir_path: str):
    """Walks through a directory and prints the number of directories and files in each level.

    Args:
        dir_path (str): The path to the directory to walk through.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def copy_img(input_dir: str, output_dir: str):
    """Copies all subdirectories from the input directory to the output directory.

    Args:
        input_dir (str): The path to the directory containing the subdirectories to be copied.
        output_dir (str): The path to the directory where the subdirectories will be copied.
                            If the new root directory does not exist, it will be created.
    """
    os.makedirs(output_dir, exist_ok=True)

    for item_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, item_name)
        output_path = os.path.join(output_dir, item_name)

        try:
            if os.path.isdir(input_path):
                shutil.copytree(input_path, output_path)
                print(f"Copied directory '{item_name}' to '{output_dir}'")
            elif os.path.isfile(input_path):
                shutil.copy2(input_path, output_path)
                print(f"Copied file '{item_name}' to '{output_dir}'")
            else:
                print(f"Skipping '{item_name}': it's not a file or directory.")
        except FileExistsError:
            print(f"Warning: Directory '{item_name}' already exists in '{output_dir}'. Skipping.")
        except Exception as e:
            print(f"An error occurred while copying '{item_name}': {e}")

    print("Copying process complete.")

def create_augmented_img(input_dir: str, 
                         output_dir: str, 
                         target_count: int,
                         augmentation_pipeline: A.Compose = None,
                         use_torchvision: bool = False):
    """
    Create new images through image augmentation to balance or create new data for the dataset.

    Parameters:
    - input_dir (str): Path to the original images directory.
    - output_dir (str): Path to save augmented images.
    - target_count (int): Desired number of total images after augmentation.
    - augmentation_pipeline (albumentations.Compose): Custom augmentation pipeline. 
      If None, a default augmentation pipeline will be used.
    - use_torchvision (bool, optional): Whether to use torchvision transforms instead of Albumentations. Default is False.

    Returns:
    - None (Saves augmented images to output_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if augmentation_pipeline is None and not use_torchvision:
        augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2)
        ])
    
    if augmentation_pipeline is None and use_torchvision:
        augmentation_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.ElasticTransform(alpha=50.0)], p=0.5),
            transforms.ToTensor(),
            transforms.ToPILImage(),
        ])
    
    image_filenames = os.listdir(input_dir)
    original_count = len(image_filenames)
    
    if original_count >= target_count:
        print(f"No augmentation needed. Already {original_count} images in {input_dir}.")
        return
    
    num_generated = 0
    image_idx = 0
    print(f"🔄 Generating images until {target_count} is reached...")
    
    while len(os.listdir(output_dir)) < target_count:
        image_name = image_filenames[image_idx]
        img_path = os.path.join(input_dir, image_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Warning: Skipping {image_name} (failed to load).")
            image_idx = (image_idx + 1) % original_count
            continue
        
        if use_torchvision:
            pil_img = Image.fromarray(img)
            augmented = augmentation_pipeline(pil_img)
            augmented = np.array(augmented)
        else:
            augmented = augmentation_pipeline(image=img)['image']
        
        new_img_name = f'aug_{num_generated}_{image_name}'
        new_img_path =os.path.join(output_dir, new_img_name)
        cv2.imwrite(new_img_path, augmented)
        
        num_generated += 1
        image_idx = (image_idx + 1) % original_count
        
    print(f"✅ Augmentation complete! {len(os.listdir(output_dir))} images available in '{output_dir}'.")

def torch_seed(seed: int = 42):
    """Sets the random seed for both PyTorch CPU and CUDA operations.
    
    Args:
        seed (int): The random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_loop(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: optim,
               num_classes: int,
               device: torch.device = 'cpu'):
    """Trains a PyTorch model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): The DataLoader containing the training data.
        loss_fn (nn.Module): The loss function to use for training.
        optimizer (optim): The optimizer to use for updating model parameters.
        num_classes (int): The number of classes in the classification task.
        device (torch.device): The device to train on ('cuda' or 'cpu'). Defaults to 'cpu'.

    Prints:
        str: Training loss, accuracy, precision, recall, and F1-score for the epoch.
    """
    
    accuracy_fn = Accuracy(task="multiclass", num_classes=num_classes).to(device=device)
    precision_fn = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device=device)
    recall_fn = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device=device)
    f1score_fn = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device=device)
    
    train_loss = 0
    train_accuracy = 0
    train_precision = 0
    train_recall = 0
    train_f1score = 0
    total_batches = len(dataloader)
    model.train()
    model = model.to(device)
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        loss = loss_fn(y_pred, y)
        train_loss += loss
        y_pred = y_pred.argmax(dim=1)
        
        train_accuracy += accuracy_fn(y, y_pred).item()
        train_precision += precision_fn(y, y_pred).item()
        train_recall += recall_fn(y, y_pred).item()
        train_f1score += f1score_fn(y, y_pred).item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= total_batches
    train_accuracy /= total_batches
    train_precision /= total_batches
    train_recall /= total_batches
    train_f1score /= total_batches    
    print(f'Train loss: {train_loss:.5f} | Train Accuracy: {train_accuracy:.2f} | Train Precision: {train_precision:.2f} | Train Recall: {train_recall:.2f} | Train F1-Score: {train_f1score:.2f}')
 
def test_model(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               num_classes: int,
               device: torch.device = 'cpu'):
    """Evaluates a PyTorch model on a test dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): The DataLoader containing the test data.
        loss_fn (nn.Module): The loss function to use for evaluation.
        num_classes (int): The number of classes in the classification task.
        device (torch.device): The device to evaluate on ('cuda' or 'cpu'). Defaults to 'cpu'.

    Prints:
        str: Test loss, accuracy, precision, recall, and F1-score.

    Returns:
        tuple: A tuple containing the test loss, accuracy, precision, recall, and F1-score.
    """
    
    accuracy_fn = Accuracy(task="multiclass", num_classes=num_classes).to(device=device)
    precision_fn = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device=device)
    recall_fn = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device=device)
    f1score_fn = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device=device)
    
    test_loss = 0
    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    test_f1score = 0
    total_batches = len(dataloader)
    model.eval()
    model = model.to(device)
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            if isinstance(test_pred, tuple):
                test_pred = test_pred[0]
            test_loss += loss_fn(test_pred, y)
            y_pred = test_pred.argmax(dim=1)
            
            test_accuracy += accuracy_fn(y, y_pred).item()
            test_precision += precision_fn(y, y_pred).item()
            test_recall += recall_fn(y, y_pred).item()
            test_f1score += f1score_fn(y, y_pred).item()
            
        test_loss /= total_batches
        test_accuracy /= total_batches
        test_precision /= total_batches
        test_recall /= total_batches
        test_f1score /= total_batches
        print(f'Test loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f} | Test Precision: {test_precision:.2f} | Test Recall: {test_recall:.2f} | Test F1-Score: {test_f1score:.2f}')
    
    return test_loss, test_accuracy, test_precision, test_recall, test_f1score

def test_demo(model, dataloader, class_names, device, num_samples=9):
    model.eval()
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))  # Display up to 9 images
    axs = axs.ravel()

    with torch.no_grad():
        all_images = []
        all_labels = []
        all_preds = []

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_preds.extend(preds.cpu())

        if not all_images:
            print("No images found in the dataloader.")
            return

        num_available = len(all_images)
        indices = random.sample(range(num_available), min(num_samples, num_available))

        for i, idx in enumerate(indices):
            img = all_images[idx].numpy().squeeze()
            true_label = all_labels[idx].item()
            pred_label = all_preds[idx].item()

            pred_class = class_names[pred_label]
            true_class = class_names[true_label]
            title = f"Pred: {pred_class}\nTrue: {true_class}"
            color = 'green' if pred_class == true_class else 'red'

            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(title, color=color)
            axs[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, dataloader, class_names, device):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({model.__class__.__name__})")
    plt.show()

def main():
    custom_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.ElasticTransform(alpha=50.0)], p=0.5),
    ])

    create_augmented_img('Training/no_tumor', 'Training_augmented/no_tumor', target_count=820, augmentation_pipeline=custom_pipeline, use_torchvision=True)
    # copy_img('Training/glioma_tumor', 'Training_augmented/glioma_tumor')
    # copy_img('Training/meningioma_tumor', 'Training_augmented/meningioma_tumor')
    # copy_img('Training/pituitary_tumor', 'Training_augmented/pituitary_tumor')
    
    see_dir('Training_augmented')
    
if __name__ == '__main__':
    main()