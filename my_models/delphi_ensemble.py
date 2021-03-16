
import torch
from typing import Optional, Union, List
import segmentation_models_pytorch as smp



class DelphiEnsemble(smp.UnetPlusPlus):
    
    def __init__(self, opt,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None):
        
        super().__init__()
        
       

         
        self.checkpoint_1 = torch.load(opt.model_1_path, map_location=opt.device)
        self.checkpoint_2 = torch.load(opt.model_2_path, map_location=opt.device)
        self.checkpoint_3 = torch.load(opt.model_3_path, map_location=opt.device)
        self.checkpoint_4 = torch.load(opt.model_4_path, map_location=opt.device)
        self.checkpoint_5 = torch.load(opt.model_5_path, map_location=opt.device)

        self.model_1 = self.checkpoint_1["model"]
        self.model_2 = self.checkpoint_2["model"]
        self.model_3 = self.checkpoint_3["model"]
        self.model_4 = self.checkpoint_4["model"]
        self.model_5 = self.checkpoint_5["model"]
        
        
        
    def forward(self,x):

        out_1 = self.model_1(x)
        out_2 = self.model_2(x)
        out_3 = self.model_3(x)
        out_4 = self.model_4(x)
        out_5 = self.model_5(x)

        out_1_c1 = out_1[:,0:1,:,:]
        out_1_c2 = out_1[:,1:,:,:]

        out_2_c1 = out_2[:,0:1,:,:]
        out_2_c2 = out_2[:,1:,:,:]

        out_3_c1 = out_3[:,0:1,:,:]
        out_3_c2 = out_3[:,1:,:,:]

        out_4_c1 = out_4[:,0:1,:,:]
        out_4_c2 = out_4[:,1:,:,:]

        out_5_c1 = out_5[:,0:1,:,:]
        out_5_c2 = out_5[:,1:,:,:]

        #==========
        out_c1 = torch.cat((out_1_c1, out_2_c1, out_3_c1, out_4_c1, out_5_c1), dim=1)
        out_c2 = torch.cat((out_1_c2, out_2_c2, out_3_c2, out_4_c2, out_5_c2), dim=1)

        out_c1_mean = out_c1.mean(dim=1).unsqueeze(dim=0)
        out_c2_mean = out_c2.mean(dim=1).unsqueeze(dim=0)

        mask = torch.cat((out_c1_mean, out_c2_mean), dim=1)

        #print("out_ shape=", mask.shape)
        
    

        return mask 