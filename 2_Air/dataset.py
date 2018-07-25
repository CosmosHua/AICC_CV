from os import listdir
from os.path import join
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from skimage import io

from util import is_image_file, load_img


# pixel values of the inputs images are squeezed to [0,1]
# pixel values of the outputs images are within [0,255]

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.photo_path = join(image_dir, "inputs")
        self.sketch_path = join(image_dir, "outputs")
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x)]

        transform_list_inputs = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list_outputs = []
        transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform_inputs = transforms.Compose(transform_list_inputs)
        self.transform_outputs = transforms.Compose(transform_list_outputs)
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.photo_path, self.image_filenames[index]))
        input = self.transform(input)
        target = load_img(join(self.sketch_path, self.image_filenames[index]))
        target = self.transform(target)
        # target = io.imread(join(self.sketch_path, self.image_filenames[index]))
        # target = torch.Tensor(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)