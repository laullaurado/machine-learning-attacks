from torch.utils.data import Dataset
import torch
import PIL

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UTKFace(Dataset):

    def __init__(self, samples, label, transform=None):
        self.samples = samples
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_array = self.samples.iloc[idx, 3]
        image = PIL.Image.fromarray(image_array)
        
        if self.transform:
            image = self.transform(image)
        
        if self.label =='gender':
            label = int(self.samples.iloc[idx, 1])
            sample =  {'image': image, 'gender': label}
            
        if self.label == 'race':
            label = int(self.samples.iloc[idx, 2])
            sample =  {'image': image, 'race': label}
        
        return sample


class AttackData(Dataset):
    def __init__(self, samples, target_model, transform=None):
        self.samples = samples
        self.target_model = target_model.to(device)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_array = self.samples.iloc[idx, 3]
        image = PIL.Image.fromarray(image_array)
        
        if self.transform:
            image = self.transform(image)

        image = image.to(device)

        z = self.target_model(image.unsqueeze(0))[1]
        z = z.squeeze(0)

        label = int(self.samples.iloc[idx, 2])
        sample = {'z': z, 'race': label}
        
        return sample
