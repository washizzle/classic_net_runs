import torchvision.models as models
import torch
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from datetime import datetime
import time
import os
import sys
from pid_reader import PIDReader
import copy
#dir full omniglot dataset: F:\Users\maurice\Data_afstudeerproject\omniglot\protobuf_datasets\full dataset
    #dir separate alphabets: F:\Users\maurice\Data_afstudeerproject\omniglot\protobuf_datasets\alphabets\[alphabet_name]

def vgg16(num_classes):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model

def resnet34(num_classes, pretrained):
    model = models.resnet34(pretrained=pretrained)
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
    
def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #add logging
    log_subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_path = os.path.join(os.path.expanduser(args.logs_base_path), log_subdir)
    if not os.path.isdir(log_path):  # Create the log directory if it doesn't exist
        os.makedirs(log_path)
    model_path = os.path.join(os.path.expanduser(args.models_base_path), log_subdir)
    if not os.path.isdir(model_path):  # Create the model directory if it doesn't exist
        os.makedirs(model_path)

    #load data
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    datasets_path = os.path.join(os.path.expanduser(args.root_path), "datasets")
    # if I use my own datasets, they are always .pid files, so i want to keep to keep these in a separate folder from the jpeg files loaded from torchvision.
    if args.torchvision_dataset:
        datasets_path = os.path.join(os.path.expanduser(args.root_path), "jpg_datasets")
    dataloaders, dataset_sizes, num_classes = None, None, None
    if args.dataset_name == 'omniglot_1_folder_splits':
        dataset_path = os.path.join(os.path.expanduser(datasets_path), args.dataset_name)
        dataloaders, dataset_sizes, num_classes = load_own_data(dataset_path, args.train_csv_path, args.val_csv_path, 
                                              args.image_count, args.train_format, args.valid_format, 
                                              args.train_dataset_depth, args.val_dataset_depth, data_transforms)
    elif args.torchvision_dataset:
        dataset_path = os.path.join(os.path.expanduser(datasets_path), args.dataset_name)
        dataloaders, dataset_sizes, num_classes = load_torchvision_data(args.dataset_name, dataset_path, data_transforms)
    else:
        raise Exception("This dataset is not known.")
    
    
    #load model
    print("num_classes: ", num_classes)
    model = resnet34(num_classes, args.pretrained_imagenet)
    # print("resnet34 model: ", model)
    # model = vgg16(num_classes)
    model = model.to(device)
    print("model: ", model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    model_ft = train_model(device, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, log_path, model_path, num_epochs=25)


def load_torchvision_data(dataset_name, dataset_path, data_transforms)
    data = None
    if dataset_name == 'mnist':
        data = {x: torchvision.datasets.MNIST(os.path.join(dataset_path, x), train=x=='train', 
                    transform=data_transforms[x], target_transform=None, download=True)
                for x in ['train', 'val']}
    else:
        raise Exception("This torchvision dataset is not known.")
    classes = data['train'].targets.unique()
    num_classes = len(classes)
    dataset_sizes = {x: len(data[x]) for x in ['train','val']}
    dataloaders = {x: torch.utils.data.DataLoader(data[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    return num_classes, dataset_sizes, dataloaders
                   

def load_own_data(dataset_path, train_csv_path, val_csv_path, image_count, train_format, valid_format, train_dataset_depth, val_dataset_depth, data_transforms):
    
    format = {'train': train_format, 'val': valid_format}
    csv_path = {'train': train_csv_path, 'val': val_csv_path}
    dataset_depth = {'train': train_dataset_depth, 'val': val_dataset_depth}
    image_datasets = {x: PIDReader(os.path.join(dataset_path, x), data_transforms[x], csv_path[x], image_count, format[x], dataset_depth[x])
                      for x in ['train', 'val']}
    # image_datasets = {"train": train_dataset, "val": val_dataset}
    num_classes = image_datasets['train'].num_classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    return dataloaders, dataset_sizes, num_classes

def train_model(device, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, log_path, model_path, num_epochs=25):
    since = time.time()
    since_last_epoch = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for index, batch_sample in enumerate(dataloaders[phase]):
                if not 'exception' in batch_sample:
                    print("index: ", index)
                    # print("inputs: ", batch_sample['image'])
                    print("labels: ", batch_sample['class'])
                    labels = batch_sample['class'].to(device)
                    inputs = batch_sample['image'].to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # print("outputs: ", outputs)
                        _, preds = torch.max(outputs, 1)
                        print("preds: ", preds)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            with open('{}/{}_log_epoch{}.txt'.format(log_path, phase, epoch), 'w') as f:
                f.write(str(epoch) + '\t' +
                        str(epoch_acc) + '\t' +
                        str(epoch_loss) + '\t' +
                        str(time.time() - since_last_epoch))
            since_last_epoch = time.time()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save({'state_dict': model.state_dict()},
               '{}/best_model.pth'.format(model_path))
    return model



def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Classic model training and validation')

    parser.add_argument('--root_path', type=str,
                        help='Path where the datasets and jpg_datasets folders are stored.', default='./../datasets/')
    parser.add_argument('--dataset_name', type=str,
                        help='Name of the dataset folder.', default='omniglot')
    parser.add_argument('--logs_base_path', type=str,
                        help='Path where to store logging.', default='./logs/')
    parser.add_argument('--models_base_path', type=str,
                        help='Path where to store resulting models.', default='./models/')
    parser.add_argument('--train_csv_path', type=str,
                        help='Path where to retrieve the csv with the list of all the training images.', default='./../csv/train')
    parser.add_argument('--val_csv_path', type=str,
                        help='Path where to retrieve the csv with the list of all the validation images.', default='./../csv/val')                    
    parser.add_argument('--image_count', type=int,
                        help='Amount of images per epoch per phase (train/valid).', default=10000)
    parser.add_argument('--train_format', type=str,
                        help='Format of images for training set', default='.png')
    parser.add_argument('--valid_format', type=str,
                        help='Format of images for validation set', default='.png')
    parser.add_argument('--train_dataset_depth', default = 3, type = int,
                        help = 'Defines depth of the images in the train dataset. E.g. Grayscale = 1 and rgb = 3 ')
    parser.add_argument('--val_dataset_depth', default = 3, type = int,
                        help = 'Defines depth of the images in the validation dataset. E.g. Grayscale = 1 and rgb = 3 ')
    parser.add_argument('--learning_rate', default = 0.001, type = int,
                        help = 'Learning rate for the optimizer')
    parser.add_argument('--pretrained_imagenet', 
                    help='Defines whether the used model is pretrained on ImageNet or not.', action='store_true')
    parser.add_argument('--torchvision_dataset', 
                    help='Defines whether the dataset has to be downloaded through torchvision or not.', action='store_true')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
