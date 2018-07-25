import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import _NetG, _NetD
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='face', help='input dataset')
parser.add_argument('--batch_size', type=int, default=7, help='train batch size')
parser.add_argument('--num_epochs', type=str, default="60,100", help='number of train epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for generator')
parser.add_argument("--vgg_loss", type=bool, default=False, help="Use content loss?")
args = parser.parse_args()
print(args)

# Directories for loading data and saving results
data_dir = '../Data/'
model_dir = args.dataset + '_model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
transform = transforms.Compose([transforms.ToTensor()])

# Train data
train_data = DatasetFromFolder(data_dir, subfolder='train', transform=transform, fliplr=True)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True)

test_data = DatasetFromFolder(data_dir, subfolder='test', transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=4)


# Models
if args.vgg_loss:
    class _content_model(nn.Module):
        def __init__(self):
            super(_content_model, self).__init__()
            self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
                
        def forward(self, x):
            out = self.feature(x)
            return out

    netVGG = models.vgg19(pretrained=True).cuda()
    netContent = _content_model().cuda()

model = _NetG().cuda()
criterion = nn.MSELoss().cuda()

# Optimizers
epochs = [int(i) for i in args.num_epochs.split(',')]
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=epochs, gamma=0.1)

# Training GAN
best_epoch = 0
best_loss = 100
for epoch in range(epochs[-1]):
    scheduler.step()

    # training
    test_loss = []
    model.train()
    for i, (_, inputs, targets) in enumerate(train_data_loader):
        # input & target image data
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda(), requires_grad=False)

        # Train generator
        outputs = model(inputs)
        # L1 loss
        loss = criterion(outputs, targets)

        if args.vgg_loss:
            content_input = netContent(output)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)

        optimizer.zero_grad()

        if args.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_graph=True)

        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Step [%d/%d], loss: %.4f' % (epoch+1, epochs[-1], i+1, len(train_data_loader), loss.data[0]))

    model.eval()
    for _, (_, test_inputs, test_targets) in enumerate(test_data_loader):
        inputs = Variable(test_inputs.cuda(), volatile=True)
        targets = Variable(test_targets.cuda(), volatile=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss.append(loss.cpu().data[0])

    test_loss = sum(test_loss) / len(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch+1
    print('Epoch [%d/%d], loss: %.4f, Best Epoch %d, Best Loss %.4f' % (epoch+1, epochs[-1], test_loss, best_epoch, best_loss))

torch.save(model.state_dict(), model_dir + 'generator_param.pkl')
