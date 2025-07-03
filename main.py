import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import ImageFilter 
from datasets import dataset 
from PIL import Image  


# required preprocessing
base_transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
# augmentation pipeline
augment_transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.RandomHorizontalFlip(p=1.0),
   transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=1))),
   transforms.ToTensor(),
   transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  #noise
   transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
class StanfordCarsDataset(Dataset):
   def __init__(self, hf_dataset, transform):
       self.hf_dataset = hf_dataset
       self.transform = transform
   def __len__(self):
       return len(self.hf_dataset)
   def __getitem__(self, idx):
       example = self.hf_dataset[idx]
       img = example['image'].convert('RGB')  # already a PIL.Image
       label = example['label']
       img = self.transform(img)
       return img, torch.tensor(label)




train_base = StanfordCarsDataset(dataset['train'], base_transform)
train_aug = StanfordCarsDataset(dataset['train'], augment_transform)
train_combined = ConcatDataset([train_base, train_aug])


train_loader = DataLoader(train_combined, batch_size=32, shuffle=True) #arbitrary batching, to be changed later, shuffle to remove dataset patterns
test_loader = DataLoader(dataset["test"], batch_size=32, shuffle=True)


