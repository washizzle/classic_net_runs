import torchvision.models as models
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import os
from pid_reader import PIDReader
#dir full omniglot dataset: F:\Users\maurice\Data_afstudeerproject\omniglot\protobuf_datasets\full dataset
    #dir separate alphabets: F:\Users\maurice\Data_afstudeerproject\omniglot\protobuf_datasets\alphabets\[alphabet_name]

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #add logging
    log_subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_path = os.path.join(os.path.expanduser(args.logs_base_path), log_subdir)
    if not os.path.isdir(log_path):  # Create the log directory if it doesn't exist
        os.makedirs(log_path)
    model_path = os.path.join(os.path.expanduser(args.models_base_path), subdir)
    if not os.path.isdir(model_path):  # Create the model directory if it doesn't exist
        os.makedirs(model_path)


    #load model
    model = models.vgg16(pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    #load data
    dataset_path = os.path.join(os.path.expanduser(args.datasets_path), dataset_name)
    dataloaders = load_data(dataset_path, args.csv_path, args.image_count, args.train_format, args.valid_format)

    model_ft = train_model(model, criterion, optimizer, scheduler, num_epochs=25)


def load_data(dataset_path, csv_path, image_count, train_format, valid_format):
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
    format = {'train': train_format, 'valid': valid_format}
    image_datasets = {x: pid_reader(os.path.join(dataset_path, x), data_transforms[x], csv_path, image_count, format[x])
                      for x in ['train', 'val']}
    # image_datasets = {"train": train_dataset, "val": val_dataset}
    return {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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
            for inputs, labels in dataloaders[phase]:
                print("inputs: ", inputs)
                print("labels: ", labels)
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

    parser.add_argument('--datasets_path', type=str,
                        help='Path where to take the datasets from.', default='./../datasets/')
    parser.add_argument('--dataset_name', type=str,
                        help='Name of the dataset folder.', default='omniglot')
    parser.add_argument('--logs_base_path', type=str,
                        help='Path where to store logging.', default='./logs/')
    parser.add_argument('--models_base_path', type=str,
                        help='Path where to store resulting models.', default='./models/')
    parser.add_argument('--csv_path', type=str,
                        help='Path where to store resulting models.', default='./../csv/')
    parser.add_argument('--image_count', type=int,
                        help='Amount of images per epoch per phase (train/valid).', default=10000)
    parser.add_argument('--train_format', type=str,
                        help='Format of images for training set', default='.png')
    parser.add_argument('--valid_format', type=str,
                        help='Format of images for validation set', default='.png')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
