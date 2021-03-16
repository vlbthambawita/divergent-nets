
import torch
from typing import Optional, Union, List
import segmentation_models_pytorch as smp



class TriUnet(smp.Unet):
    
    def __init__(self, 
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,):
        
        super().__init__()
        
        self.in_model_1 = smp.Unet(
                encoder_name=encoder_name,
                in_channels=in_channels, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
        self.in_model_2 = smp.Unet(
                encoder_name=encoder_name,
                in_channels=in_channels, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
        self.out_model = smp.Unet(
                encoder_name=encoder_name,
                in_channels=4, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
    def forward(self,x):

        mask_1 = self.in_model_1(x)
        mask_2 = self.in_model_2(x)
        
        mask_concat = torch.cat((mask_1, mask_2), 1)
        
        mask = self.out_model(mask_concat)

        return mask 