import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_MASK_FOLDER_NAMES = ['GH', 'ZZ', 'XW']


class IVUSDataset(Dataset):

    def __init__(
        self,
        img_dir,
        mask_dir,
        img_ext='.jpg',
        mask_ext='.png',
        num_classes=4,
        mask_folder_names=None,
        transform=None,
        class_index_format=True
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.class_index_format = class_index_format

        self.mask_folder_names = mask_folder_names or DEFAULT_MASK_FOLDER_NAMES
        assert len(self.mask_folder_names) == num_classes - 1, \
            f"len(mask_folder_names)={len(self.mask_folder_names)} must equal num_classes-1={num_classes-1}"

        self.fg_map = {cls_name: i + 1 for i, cls_name in enumerate(self.mask_folder_names)}

        self.img_names = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(self.img_ext)
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # load grayscale image
        img = Image.open(img_path).convert("L")
        img = np.array(img)

        h, w = img.shape
        multi_mask = np.zeros((h, w), dtype=np.uint8)

        # build multi-class mask from per-class binary masks
        for cls_name in self.mask_folder_names:
            mask_path = os.path.join(self.mask_dir, cls_name, img_name.replace(self.img_ext, self.mask_ext))

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                if mask.size != (w, h):
                    mask = mask.resize((w, h), Image.NEAREST)
                mask = np.array(mask)
                multi_mask[mask > 0] = self.fg_map[cls_name]

        # optional augmentation
        if self.transform is not None:
            augmented = self.transform(image=img, mask=multi_mask)
            img = augmented['image']
            multi_mask = augmented['mask']

        # image to tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)  # [1, H, W]

        # mask to tensor
        if self.class_index_format:
            if not isinstance(multi_mask, torch.Tensor):
                multi_mask = torch.from_numpy(np.array(multi_mask)).long()
            else:
                multi_mask = multi_mask.long()
        else:
            if isinstance(multi_mask, torch.Tensor):
                multi_mask_np = multi_mask.cpu().numpy()
            else:
                multi_mask_np = np.array(multi_mask)

            one_hot = np.zeros((self.num_classes, multi_mask_np.shape[0], multi_mask_np.shape[1]), dtype=np.float32)
            for class_id in range(self.num_classes):
                one_hot[class_id] = (multi_mask_np == class_id).astype(np.float32)
            multi_mask = torch.from_numpy(one_hot).float()

        return img, multi_mask
