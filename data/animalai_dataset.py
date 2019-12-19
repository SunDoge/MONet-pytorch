from .clevr_dataset import CLEVRDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class AnimalAIDataset(CLEVRDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self._transform = T.Compose([
            T.RandomCrop(64),
            T.ToTensor(),
            T.Normalize([0.5]*self.opt.input_nc, [0.5] * self.opt.input_nc)
        ])
