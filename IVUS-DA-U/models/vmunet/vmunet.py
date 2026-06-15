from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.input_channels = input_channels

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x):

        
        logits = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            

            if 'patch_embed.proj.weight' in pretrained_dict and 'patch_embed.proj.weight' in model_dict:
                pretrained_weight = pretrained_dict['patch_embed.proj.weight']
                model_weight = model_dict['patch_embed.proj.weight']
                

                if pretrained_weight.shape[1] != model_weight.shape[1]:
                    print(f"调整patch_embed.proj.weight的通道维度从{pretrained_weight.shape}到{model_weight.shape}")

                    if pretrained_weight.shape[1] == 3 and model_weight.shape[1] == 1:

                        pretrained_dict['patch_embed.proj.weight'] = pretrained_weight.mean(dim=1, keepdim=True)
            

            new_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict.keys() and model_dict[k].shape == pretrained_dict[k].shape}
            model_dict.update(new_dict)

            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")


            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
                    

            for k in list(pretrained_dict.keys()):
                if 'proj.weight' in k and k in model_dict:
                    pretrained_weight = pretrained_dict[k]
                    model_weight = model_dict[k]
                    if pretrained_weight.shape[1] != model_weight.shape[1]:
                        print(f"调整{k}的通道维度从{pretrained_weight.shape}到{model_weight.shape}")
                        if pretrained_weight.shape[1] == 3 and model_weight.shape[1] == 1:
                            pretrained_dict[k] = pretrained_weight.mean(dim=1, keepdim=True)

            new_dict = {k: v for k, v in pretrained_dict.items() 
                       if k in model_dict.keys() and model_dict[k].shape == pretrained_dict[k].shape}
            model_dict.update(new_dict)
            

            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
