import torchvision.models as models
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import os
from pid_reader import PIDReader

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), log_subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    #add logging

    #load model
    model = models.vgg16(pretrained=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    #load data
    dataset_dir = os.path.join(os.path.expanduser(args.datasets_dir), dataset_name)
    dataloaders = loadData(dataset_dir, args.csv_dir)

    #if not done already, split into train and validate sets.

    train_model(model, criterion, optimizer, scheduler)
    #dir full omniglot dataset: F:\Users\maurice\Data_afstudeerproject\omniglot\protobuf_datasets\full dataset
    #dir separate alphabets: F:\Users\maurice\Data_afstudeerproject\omniglot\protobuf_datasets\alphabets\[alphabet_name]

def loadData(dataset_dir, csv_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: pid_reader(os.path.join(dataset_dir, x), data_transforms[x], csv_dir)
                      for x in ['train', 'val']}
    # image_datasets = {"train": train_dataset, "val": val_dataset}
    return {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

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
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
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
    return model



def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Classic model training and validation')

    parser.add_argument('--datasets_dir', type=str,
                        help='Directory where to take the datasets from.', default='./../datasets/')
    parser.add_argument('--dataset_name', type=str,
                        help='Name of the dataset folder.', default='omniglot')
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to store logging.', default='./logs/')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to store resulting models.', default='./models/')
    parser.add_argument('--csv_dir', type=str,
                        help='Directory where to store resulting models.', default='./../csv/')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
