import torch_pruning as tp
from typing import Sequence

from models.yolo import DiverseBranchBlock
class DiverseBranchBlockPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: DiverseBranchBlock, idxs: Sequence[int]):
        # prune for dbb_origin
        tp.prune_conv_out_channels(layer.dbb_origin.conv, idxs)
        tp.prune_batchnorm_out_channels(layer.dbb_origin.bn, idxs)
        
        # prune for dbb_avg and dbb_1x1
        if hasattr(layer.dbb_avg, 'conv'):
            # dbb_avg
            tp.prune_conv_out_channels(layer.dbb_avg.conv, idxs)
            tp.prune_batchnorm_out_channels(layer.dbb_avg.bn.bn, idxs)
            tp.prune_batchnorm_out_channels(layer.dbb_avg.avgbn, idxs)
            
            # dbb_1x1
            tp.prune_conv_out_channels(layer.dbb_1x1.conv, idxs)
            tp.prune_batchnorm_out_channels(layer.dbb_1x1.bn, idxs)
        
        # prune for dbb_1x1_kxk
        tp.prune_conv_out_channels(layer.dbb_1x1_kxk.conv2, idxs)
        tp.prune_batchnorm_out_channels(layer.dbb_1x1_kxk.bn2, idxs)
        
        # update out_channels
        layer.out_channels = layer.out_channels - len(idxs)
        return layer
            
    def prune_in_channels(self, layer: DiverseBranchBlock, idxs: Sequence[int]):
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        
        # prune for dbb_origin
        tp.prune_conv_in_channels(layer.dbb_origin.conv, idxs)
        
        # prune for dbb_avg and dbb_1x1
        if hasattr(layer.dbb_avg, 'conv'):
            # dbb_avg
            tp.prune_conv_in_channels(layer.dbb_avg.conv, idxs)
            
            # dbb_1x1
            tp.prune_conv_in_channels(layer.dbb_1x1.conv, idxs)
        
        # prune for dbb_1x1_kxk
        if hasattr(layer.dbb_1x1_kxk, 'idconv1'):
            layer.dbb_1x1_kxk.idconv1.id_tensor = self._prune_parameter_and_grad(layer.dbb_1x1_kxk.idconv1.id_tensor, keep_idxs=keep_idxs, pruning_dim=1)
            tp.prune_conv_in_channels(layer.dbb_1x1_kxk.idconv1.conv, idxs)
        elif hasattr(layer.dbb_1x1_kxk, 'conv1'):
            tp.prune_conv_in_channels(layer.dbb_1x1_kxk.conv1, idxs)
        
        # update in_channels
        layer.in_channels = layer.in_channels - len(idxs)
        return layer
        
    def get_out_channels(self, layer: DiverseBranchBlock):
        return layer.out_channels
    
    def get_in_channels(self, layer: DiverseBranchBlock):
        return layer.in_channels
    
    def get_channel_groups(self, layer: DiverseBranchBlock):
        return layer.groups