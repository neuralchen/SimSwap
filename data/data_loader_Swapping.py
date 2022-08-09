import os
import glob
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.num_images = len(loader)
        self.preload()

    def preload(self):
        try:
            self.src_image1, self.src_image2 = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.src_image1, self.src_image2 = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.src_image1  = self.src_image1.cuda(non_blocking=True)
            self.src_image1  = self.src_image1.sub_(self.mean).div_(self.std)
            self.src_image2  = self.src_image2.cuda(non_blocking=True)
            self.src_image2  = self.src_image2.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        src_image1  = self.src_image1
        src_image2  = self.src_image2
        self.preload()
        return src_image1, src_image2
    
    def __len__(self):
        """Return the number of images."""
        return self.num_images

class SwappingDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                    image_dir,
                    img_transform,
                    subffix='jpg',
                    random_seed=1234):
        """Initialize and preprocess the Swapping dataset."""
        self.image_dir      = image_dir
        self.img_transform  = img_transform   
        self.subffix        = subffix
        self.dataset        = []
        self.random_seed    = random_seed
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        """Preprocess the Swapping dataset."""
        print("processing Swapping dataset images...")

        temp_path   = os.path.join(self.image_dir,'*/')
        pathes      = glob.glob(temp_path)
        self.dataset = []
        for dir_item in pathes:
            join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
            print("processing %s"%dir_item,end='\r')
            temp_list = []
            for item in join_path:
                temp_list.append(item)
            self.dataset.append(temp_list)
        random.seed(self.random_seed)
        random.shuffle(self.dataset)
        print('Finished preprocessing the Swapping dataset, total dirs number: %d...'%len(self.dataset))
             
    def __getitem__(self, index):
        """Return two src domain images and two dst domain images."""
        dir_tmp1        = self.dataset[index]
        dir_tmp1_len    = len(dir_tmp1)

        filename1   = dir_tmp1[random.randint(0,dir_tmp1_len-1)]
        filename2   = dir_tmp1[random.randint(0,dir_tmp1_len-1)]
        image1      = self.img_transform(Image.open(filename1))
        image2      = self.img_transform(Image.open(filename2))
        return image1, image2
    
    def __len__(self):
        """Return the number of images."""
        return self.num_images

def GetLoader(  dataset_roots,
                batch_size=16,
                dataloader_workers=8,
                random_seed = 1234
                ):
    """Build and return a data loader."""
        
    num_workers         = dataloader_workers
    data_root           = dataset_roots
    random_seed         = random_seed
    
    c_transforms = []
    
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    content_dataset = SwappingDataset(
                            data_root, 
                            c_transforms,
                            "jpg",
                            random_seed)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=True,shuffle=True,num_workers=num_workers,pin_memory=True)
    prefetcher = data_prefetcher(content_data_loader)
    return prefetcher

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)