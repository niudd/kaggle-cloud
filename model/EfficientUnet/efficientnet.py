import sys
sys.path.insert(0,'../..')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


BatchNorm2d = SynchronizedBatchNorm2d


#IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
#IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

IS_PYTORCH_PAD = True  # True  # False
IS_GATHER_EXCITE = False


CONVERSION = [
 'stem.0.conv.weight',	(64, 3, 3, 3),	 '_conv_stem.weight',	(48, 3, 3, 3),
 'stem.0.bn.weight',	(64,),	 '_bn0.weight',	(48,),
 'stem.0.bn.bias',	(64,),	 '_bn0.bias',	(48,),
 'stem.0.bn.running_mean',	(64,),	 '_bn0.running_mean',	(48,),
 'stem.0.bn.running_var',	(64,),	 '_bn0.running_var',	(48,),
 'block1.0.bottleneck.0.conv.weight',	(64, 1, 3, 3),	 '_blocks.0._depthwise_conv.weight',	(48, 1, 3, 3),
 'block1.0.bottleneck.0.bn.weight',	(64,),	 '_blocks.0._bn1.weight',	(48,),
 'block1.0.bottleneck.0.bn.bias',	(64,),	 '_blocks.0._bn1.bias',	(48,),
 'block1.0.bottleneck.0.bn.running_mean',	(64,),	 '_blocks.0._bn1.running_mean',	(48,),
 'block1.0.bottleneck.0.bn.running_var',	(64,),	 '_blocks.0._bn1.running_var',	(48,),
 'block1.0.bottleneck.2.squeeze.weight',	(16, 64, 1, 1),	 '_blocks.0._se_reduce.weight',	(12, 48, 1, 1),
 'block1.0.bottleneck.2.squeeze.bias',	(16,),	 '_blocks.0._se_reduce.bias',	(12,),
 'block1.0.bottleneck.2.excite.weight',	(64, 16, 1, 1),	 '_blocks.0._se_expand.weight',	(48, 12, 1, 1),
 'block1.0.bottleneck.2.excite.bias',	(64,),	 '_blocks.0._se_expand.bias',	(48,),
 'block1.0.bottleneck.3.conv.weight',	(32, 64, 1, 1),	 '_blocks.0._project_conv.weight',	(24, 48, 1, 1),
 'block1.0.bottleneck.3.bn.weight',	(32,),	 '_blocks.0._bn2.weight',	(24,),
 'block1.0.bottleneck.3.bn.bias',	(32,),	 '_blocks.0._bn2.bias',	(24,),
 'block1.0.bottleneck.3.bn.running_mean',	(32,),	 '_blocks.0._bn2.running_mean',	(24,),
 'block1.0.bottleneck.3.bn.running_var',	(32,),	 '_blocks.0._bn2.running_var',	(24,),
 'block1.1.bottleneck.0.conv.weight',	(32, 1, 3, 3),	 '_blocks.1._depthwise_conv.weight',	(24, 1, 3, 3),
 'block1.1.bottleneck.0.bn.weight',	(32,),	 '_blocks.1._bn1.weight',	(24,),
 'block1.1.bottleneck.0.bn.bias',	(32,),	 '_blocks.1._bn1.bias',	(24,),
 'block1.1.bottleneck.0.bn.running_mean',	(32,),	 '_blocks.1._bn1.running_mean',	(24,),
 'block1.1.bottleneck.0.bn.running_var',	(32,),	 '_blocks.1._bn1.running_var',	(24,),
 'block1.1.bottleneck.2.squeeze.weight',	(8, 32, 1, 1),	 '_blocks.1._se_reduce.weight',	(6, 24, 1, 1),
 'block1.1.bottleneck.2.squeeze.bias',	(8,),	 '_blocks.1._se_reduce.bias',	(6,),
 'block1.1.bottleneck.2.excite.weight',	(32, 8, 1, 1),	 '_blocks.1._se_expand.weight',	(24, 6, 1, 1),
 'block1.1.bottleneck.2.excite.bias',	(32,),	 '_blocks.1._se_expand.bias',	(24,),
 'block1.1.bottleneck.3.conv.weight',	(32, 32, 1, 1),	 '_blocks.1._project_conv.weight',	(24, 24, 1, 1),
 'block1.1.bottleneck.3.bn.weight',	(32,),	 '_blocks.1._bn2.weight',	(24,),
 'block1.1.bottleneck.3.bn.bias',	(32,),	 '_blocks.1._bn2.bias',	(24,),
 'block1.1.bottleneck.3.bn.running_mean',	(32,),	 '_blocks.1._bn2.running_mean',	(24,),
 'block1.1.bottleneck.3.bn.running_var',	(32,),	 '_blocks.1._bn2.running_var',	(24,),
 'block1.2.bottleneck.0.conv.weight',	(32, 1, 3, 3),	 '_blocks.2._depthwise_conv.weight',	(24, 1, 3, 3),
 'block1.2.bottleneck.0.bn.weight',	(32,),	 '_blocks.2._bn1.weight',	(24,),
 'block1.2.bottleneck.0.bn.bias',	(32,),	 '_blocks.2._bn1.bias',	(24,),
 'block1.2.bottleneck.0.bn.running_mean',	(32,),	 '_blocks.2._bn1.running_mean',	(24,),
 'block1.2.bottleneck.0.bn.running_var',	(32,),	 '_blocks.2._bn1.running_var',	(24,),
 'block1.2.bottleneck.2.squeeze.weight',	(8, 32, 1, 1),	 '_blocks.2._se_reduce.weight',	(6, 24, 1, 1),
 'block1.2.bottleneck.2.squeeze.bias',	(8,),	 '_blocks.2._se_reduce.bias',	(6,),
 'block1.2.bottleneck.2.excite.weight',	(32, 8, 1, 1),	 '_blocks.2._se_expand.weight',	(24, 6, 1, 1),
 'block1.2.bottleneck.2.excite.bias',	(32,),	 '_blocks.2._se_expand.bias',	(24,),
 'block1.2.bottleneck.3.conv.weight',	(32, 32, 1, 1),	 '_blocks.2._project_conv.weight',	(24, 24, 1, 1),
 'block1.2.bottleneck.3.bn.weight',	(32,),	 '_blocks.2._bn2.weight',	(24,),
 'block1.2.bottleneck.3.bn.bias',	(32,),	 '_blocks.2._bn2.bias',	(24,),
 'block1.2.bottleneck.3.bn.running_mean',	(32,),	 '_blocks.2._bn2.running_mean',	(24,),
 'block1.2.bottleneck.3.bn.running_var',	(32,),	 '_blocks.2._bn2.running_var',	(24,),
 'block2.0.bottleneck.0.conv.weight',	(192, 32, 1, 1),	 '_blocks.3._expand_conv.weight',	(144, 24, 1, 1),
 'block2.0.bottleneck.0.bn.weight',	(192,),	 '_blocks.3._bn0.weight',	(144,),
 'block2.0.bottleneck.0.bn.bias',	(192,),	 '_blocks.3._bn0.bias',	(144,),
 'block2.0.bottleneck.0.bn.running_mean',	(192,),	 '_blocks.3._bn0.running_mean',	(144,),
 'block2.0.bottleneck.0.bn.running_var',	(192,),	 '_blocks.3._bn0.running_var',	(144,),
 'block2.0.bottleneck.2.conv.weight',	(192, 1, 3, 3),	 '_blocks.3._depthwise_conv.weight',	(144, 1, 3, 3),
 'block2.0.bottleneck.2.bn.weight',	(192,),	 '_blocks.3._bn1.weight',	(144,),
 'block2.0.bottleneck.2.bn.bias',	(192,),	 '_blocks.3._bn1.bias',	(144,),
 'block2.0.bottleneck.2.bn.running_mean',	(192,),	 '_blocks.3._bn1.running_mean',	(144,),
 'block2.0.bottleneck.2.bn.running_var',	(192,),	 '_blocks.3._bn1.running_var',	(144,),
 'block2.0.bottleneck.4.squeeze.weight',	(8, 192, 1, 1),	 '_blocks.3._se_reduce.weight',	(6, 144, 1, 1),
 'block2.0.bottleneck.4.squeeze.bias',	(8,),	 '_blocks.3._se_reduce.bias',	(6,),
 'block2.0.bottleneck.4.excite.weight',	(192, 8, 1, 1),	 '_blocks.3._se_expand.weight',	(144, 6, 1, 1),
 'block2.0.bottleneck.4.excite.bias',	(192,),	 '_blocks.3._se_expand.bias',	(144,),
 'block2.0.bottleneck.5.conv.weight',	(48, 192, 1, 1),	 '_blocks.3._project_conv.weight',	(40, 144, 1, 1),
 'block2.0.bottleneck.5.bn.weight',	(48,),	 '_blocks.3._bn2.weight',	(40,),
 'block2.0.bottleneck.5.bn.bias',	(48,),	 '_blocks.3._bn2.bias',	(40,),
 'block2.0.bottleneck.5.bn.running_mean',	(48,),	 '_blocks.3._bn2.running_mean',	(40,),
 'block2.0.bottleneck.5.bn.running_var',	(48,),	 '_blocks.3._bn2.running_var',	(40,),
 'block2.1.bottleneck.0.conv.weight',	(288, 48, 1, 1),	 '_blocks.4._expand_conv.weight',	(240, 40, 1, 1),
 'block2.1.bottleneck.0.bn.weight',	(288,),	 '_blocks.4._bn0.weight',	(240,),
 'block2.1.bottleneck.0.bn.bias',	(288,),	 '_blocks.4._bn0.bias',	(240,),
 'block2.1.bottleneck.0.bn.running_mean',	(288,),	 '_blocks.4._bn0.running_mean',	(240,),
 'block2.1.bottleneck.0.bn.running_var',	(288,),	 '_blocks.4._bn0.running_var',	(240,),
 'block2.1.bottleneck.2.conv.weight',	(288, 1, 3, 3),	 '_blocks.4._depthwise_conv.weight',	(240, 1, 3, 3),
 'block2.1.bottleneck.2.bn.weight',	(288,),	 '_blocks.4._bn1.weight',	(240,),
 'block2.1.bottleneck.2.bn.bias',	(288,),	 '_blocks.4._bn1.bias',	(240,),
 'block2.1.bottleneck.2.bn.running_mean',	(288,),	 '_blocks.4._bn1.running_mean',	(240,),
 'block2.1.bottleneck.2.bn.running_var',	(288,),	 '_blocks.4._bn1.running_var',	(240,),
 'block2.1.bottleneck.4.squeeze.weight',	(12, 288, 1, 1),	 '_blocks.4._se_reduce.weight',	(10, 240, 1, 1),
 'block2.1.bottleneck.4.squeeze.bias',	(12,),	 '_blocks.4._se_reduce.bias',	(10,),
 'block2.1.bottleneck.4.excite.weight',	(288, 12, 1, 1),	 '_blocks.4._se_expand.weight',	(240, 10, 1, 1),
 'block2.1.bottleneck.4.excite.bias',	(288,),	 '_blocks.4._se_expand.bias',	(240,),
 'block2.1.bottleneck.5.conv.weight',	(48, 288, 1, 1),	 '_blocks.4._project_conv.weight',	(40, 240, 1, 1),
 'block2.1.bottleneck.5.bn.weight',	(48,),	 '_blocks.4._bn2.weight',	(40,),
 'block2.1.bottleneck.5.bn.bias',	(48,),	 '_blocks.4._bn2.bias',	(40,),
 'block2.1.bottleneck.5.bn.running_mean',	(48,),	 '_blocks.4._bn2.running_mean',	(40,),
 'block2.1.bottleneck.5.bn.running_var',	(48,),	 '_blocks.4._bn2.running_var',	(40,),
 'block2.2.bottleneck.0.conv.weight',	(288, 48, 1, 1),	 '_blocks.5._expand_conv.weight',	(240, 40, 1, 1),
 'block2.2.bottleneck.0.bn.weight',	(288,),	 '_blocks.5._bn0.weight',	(240,),
 'block2.2.bottleneck.0.bn.bias',	(288,),	 '_blocks.5._bn0.bias',	(240,),
 'block2.2.bottleneck.0.bn.running_mean',	(288,),	 '_blocks.5._bn0.running_mean',	(240,),
 'block2.2.bottleneck.0.bn.running_var',	(288,),	 '_blocks.5._bn0.running_var',	(240,),
 'block2.2.bottleneck.2.conv.weight',	(288, 1, 3, 3),	 '_blocks.5._depthwise_conv.weight',	(240, 1, 3, 3),
 'block2.2.bottleneck.2.bn.weight',	(288,),	 '_blocks.5._bn1.weight',	(240,),
 'block2.2.bottleneck.2.bn.bias',	(288,),	 '_blocks.5._bn1.bias',	(240,),
 'block2.2.bottleneck.2.bn.running_mean',	(288,),	 '_blocks.5._bn1.running_mean',	(240,),
 'block2.2.bottleneck.2.bn.running_var',	(288,),	 '_blocks.5._bn1.running_var',	(240,),
 'block2.2.bottleneck.4.squeeze.weight',	(12, 288, 1, 1),	 '_blocks.5._se_reduce.weight',	(10, 240, 1, 1),
 'block2.2.bottleneck.4.squeeze.bias',	(12,),	 '_blocks.5._se_reduce.bias',	(10,),
 'block2.2.bottleneck.4.excite.weight',	(288, 12, 1, 1),	 '_blocks.5._se_expand.weight',	(240, 10, 1, 1),
 'block2.2.bottleneck.4.excite.bias',	(288,),	 '_blocks.5._se_expand.bias',	(240,),
 'block2.2.bottleneck.5.conv.weight',	(48, 288, 1, 1),	 '_blocks.5._project_conv.weight',	(40, 240, 1, 1),
 'block2.2.bottleneck.5.bn.weight',	(48,),	 '_blocks.5._bn2.weight',	(40,),
 'block2.2.bottleneck.5.bn.bias',	(48,),	 '_blocks.5._bn2.bias',	(40,),
 'block2.2.bottleneck.5.bn.running_mean',	(48,),	 '_blocks.5._bn2.running_mean',	(40,),
 'block2.2.bottleneck.5.bn.running_var',	(48,),	 '_blocks.5._bn2.running_var',	(40,),
 'block2.3.bottleneck.0.conv.weight',	(288, 48, 1, 1),	 '_blocks.6._expand_conv.weight',	(240, 40, 1, 1),
 'block2.3.bottleneck.0.bn.weight',	(288,),	 '_blocks.6._bn0.weight',	(240,),
 'block2.3.bottleneck.0.bn.bias',	(288,),	 '_blocks.6._bn0.bias',	(240,),
 'block2.3.bottleneck.0.bn.running_mean',	(288,),	 '_blocks.6._bn0.running_mean',	(240,),
 'block2.3.bottleneck.0.bn.running_var',	(288,),	 '_blocks.6._bn0.running_var',	(240,),
 'block2.3.bottleneck.2.conv.weight',	(288, 1, 3, 3),	 '_blocks.6._depthwise_conv.weight',	(240, 1, 3, 3),
 'block2.3.bottleneck.2.bn.weight',	(288,),	 '_blocks.6._bn1.weight',	(240,),
 'block2.3.bottleneck.2.bn.bias',	(288,),	 '_blocks.6._bn1.bias',	(240,),
 'block2.3.bottleneck.2.bn.running_mean',	(288,),	 '_blocks.6._bn1.running_mean',	(240,),
 'block2.3.bottleneck.2.bn.running_var',	(288,),	 '_blocks.6._bn1.running_var',	(240,),
 'block2.3.bottleneck.4.squeeze.weight',	(12, 288, 1, 1),	 '_blocks.6._se_reduce.weight',	(10, 240, 1, 1),
 'block2.3.bottleneck.4.squeeze.bias',	(12,),	 '_blocks.6._se_reduce.bias',	(10,),
 'block2.3.bottleneck.4.excite.weight',	(288, 12, 1, 1),	 '_blocks.6._se_expand.weight',	(240, 10, 1, 1),
 'block2.3.bottleneck.4.excite.bias',	(288,),	 '_blocks.6._se_expand.bias',	(240,),
 'block2.3.bottleneck.5.conv.weight',	(48, 288, 1, 1),	 '_blocks.6._project_conv.weight',	(40, 240, 1, 1),
 'block2.3.bottleneck.5.bn.weight',	(48,),	 '_blocks.6._bn2.weight',	(40,),
 'block2.3.bottleneck.5.bn.bias',	(48,),	 '_blocks.6._bn2.bias',	(40,),
 'block2.3.bottleneck.5.bn.running_mean',	(48,),	 '_blocks.6._bn2.running_mean',	(40,),
 'block2.3.bottleneck.5.bn.running_var',	(48,),	 '_blocks.6._bn2.running_var',	(40,),
 'block2.4.bottleneck.0.conv.weight',	(288, 48, 1, 1),	 '_blocks.7._expand_conv.weight',	(240, 40, 1, 1),
 'block2.4.bottleneck.0.bn.weight',	(288,),	 '_blocks.7._bn0.weight',	(240,),
 'block2.4.bottleneck.0.bn.bias',	(288,),	 '_blocks.7._bn0.bias',	(240,),
 'block2.4.bottleneck.0.bn.running_mean',	(288,),	 '_blocks.7._bn0.running_mean',	(240,),
 'block2.4.bottleneck.0.bn.running_var',	(288,),	 '_blocks.7._bn0.running_var',	(240,),
 'block2.4.bottleneck.2.conv.weight',	(288, 1, 3, 3),	 '_blocks.7._depthwise_conv.weight',	(240, 1, 3, 3),
 'block2.4.bottleneck.2.bn.weight',	(288,),	 '_blocks.7._bn1.weight',	(240,),
 'block2.4.bottleneck.2.bn.bias',	(288,),	 '_blocks.7._bn1.bias',	(240,),
 'block2.4.bottleneck.2.bn.running_mean',	(288,),	 '_blocks.7._bn1.running_mean',	(240,),
 'block2.4.bottleneck.2.bn.running_var',	(288,),	 '_blocks.7._bn1.running_var',	(240,),
 'block2.4.bottleneck.4.squeeze.weight',	(12, 288, 1, 1),	 '_blocks.7._se_reduce.weight',	(10, 240, 1, 1),
 'block2.4.bottleneck.4.squeeze.bias',	(12,),	 '_blocks.7._se_reduce.bias',	(10,),
 'block2.4.bottleneck.4.excite.weight',	(288, 12, 1, 1),	 '_blocks.7._se_expand.weight',	(240, 10, 1, 1),
 'block2.4.bottleneck.4.excite.bias',	(288,),	 '_blocks.7._se_expand.bias',	(240,),
 'block2.4.bottleneck.5.conv.weight',	(48, 288, 1, 1),	 '_blocks.7._project_conv.weight',	(40, 240, 1, 1),
 'block2.4.bottleneck.5.bn.weight',	(48,),	 '_blocks.7._bn2.weight',	(40,),
 'block2.4.bottleneck.5.bn.bias',	(48,),	 '_blocks.7._bn2.bias',	(40,),
 'block2.4.bottleneck.5.bn.running_mean',	(48,),	 '_blocks.7._bn2.running_mean',	(40,),
 'block2.4.bottleneck.5.bn.running_var',	(48,),	 '_blocks.7._bn2.running_var',	(40,),
 'block3.0.bottleneck.0.conv.weight',	(288, 48, 1, 1),	 '_blocks.8._expand_conv.weight',	(240, 40, 1, 1),
 'block3.0.bottleneck.0.bn.weight',	(288,),	 '_blocks.8._bn0.weight',	(240,),
 'block3.0.bottleneck.0.bn.bias',	(288,),	 '_blocks.8._bn0.bias',	(240,),
 'block3.0.bottleneck.0.bn.running_mean',	(288,),	 '_blocks.8._bn0.running_mean',	(240,),
 'block3.0.bottleneck.0.bn.running_var',	(288,),	 '_blocks.8._bn0.running_var',	(240,),
 'block3.0.bottleneck.2.conv.weight',	(288, 1, 5, 5),	 '_blocks.8._depthwise_conv.weight',	(240, 1, 5, 5),
 'block3.0.bottleneck.2.bn.weight',	(288,),	 '_blocks.8._bn1.weight',	(240,),
 'block3.0.bottleneck.2.bn.bias',	(288,),	 '_blocks.8._bn1.bias',	(240,),
 'block3.0.bottleneck.2.bn.running_mean',	(288,),	 '_blocks.8._bn1.running_mean',	(240,),
 'block3.0.bottleneck.2.bn.running_var',	(288,),	 '_blocks.8._bn1.running_var',	(240,),
 'block3.0.bottleneck.4.squeeze.weight',	(12, 288, 1, 1),	 '_blocks.8._se_reduce.weight',	(10, 240, 1, 1),
 'block3.0.bottleneck.4.squeeze.bias',	(12,),	 '_blocks.8._se_reduce.bias',	(10,),
 'block3.0.bottleneck.4.excite.weight',	(288, 12, 1, 1),	 '_blocks.8._se_expand.weight',	(240, 10, 1, 1),
 'block3.0.bottleneck.4.excite.bias',	(288,),	 '_blocks.8._se_expand.bias',	(240,),
 'block3.0.bottleneck.5.conv.weight',	(80, 288, 1, 1),	 '_blocks.8._project_conv.weight',	(64, 240, 1, 1),
 'block3.0.bottleneck.5.bn.weight',	(80,),	 '_blocks.8._bn2.weight',	(64,),
 'block3.0.bottleneck.5.bn.bias',	(80,),	 '_blocks.8._bn2.bias',	(64,),
 'block3.0.bottleneck.5.bn.running_mean',	(80,),	 '_blocks.8._bn2.running_mean',	(64,),
 'block3.0.bottleneck.5.bn.running_var',	(80,),	 '_blocks.8._bn2.running_var',	(64,),
 'block3.1.bottleneck.0.conv.weight',	(480, 80, 1, 1),	 '_blocks.9._expand_conv.weight',	(384, 64, 1, 1),
 'block3.1.bottleneck.0.bn.weight',	(480,),	 '_blocks.9._bn0.weight',	(384,),
 'block3.1.bottleneck.0.bn.bias',	(480,),	 '_blocks.9._bn0.bias',	(384,),
 'block3.1.bottleneck.0.bn.running_mean',	(480,),	 '_blocks.9._bn0.running_mean',	(384,),
 'block3.1.bottleneck.0.bn.running_var',	(480,),	 '_blocks.9._bn0.running_var',	(384,),
 'block3.1.bottleneck.2.conv.weight',	(480, 1, 5, 5),	 '_blocks.9._depthwise_conv.weight',	(384, 1, 5, 5),
 'block3.1.bottleneck.2.bn.weight',	(480,),	 '_blocks.9._bn1.weight',	(384,),
 'block3.1.bottleneck.2.bn.bias',	(480,),	 '_blocks.9._bn1.bias',	(384,),
 'block3.1.bottleneck.2.bn.running_mean',	(480,),	 '_blocks.9._bn1.running_mean',	(384,),
 'block3.1.bottleneck.2.bn.running_var',	(480,),	 '_blocks.9._bn1.running_var',	(384,),
 'block3.1.bottleneck.4.squeeze.weight',	(20, 480, 1, 1),	 '_blocks.9._se_reduce.weight',	(16, 384, 1, 1),
 'block3.1.bottleneck.4.squeeze.bias',	(20,),	 '_blocks.9._se_reduce.bias',	(16,),
 'block3.1.bottleneck.4.excite.weight',	(480, 20, 1, 1),	 '_blocks.9._se_expand.weight',	(384, 16, 1, 1),
 'block3.1.bottleneck.4.excite.bias',	(480,),	 '_blocks.9._se_expand.bias',	(384,),
 'block3.1.bottleneck.5.conv.weight',	(80, 480, 1, 1),	 '_blocks.9._project_conv.weight',	(64, 384, 1, 1),
 'block3.1.bottleneck.5.bn.weight',	(80,),	 '_blocks.9._bn2.weight',	(64,),
 'block3.1.bottleneck.5.bn.bias',	(80,),	 '_blocks.9._bn2.bias',	(64,),
 'block3.1.bottleneck.5.bn.running_mean',	(80,),	 '_blocks.9._bn2.running_mean',	(64,),
 'block3.1.bottleneck.5.bn.running_var',	(80,),	 '_blocks.9._bn2.running_var',	(64,),
 'block3.2.bottleneck.0.conv.weight',	(480, 80, 1, 1),	 '_blocks.10._expand_conv.weight',	(384, 64, 1, 1),
 'block3.2.bottleneck.0.bn.weight',	(480,),	 '_blocks.10._bn0.weight',	(384,),
 'block3.2.bottleneck.0.bn.bias',	(480,),	 '_blocks.10._bn0.bias',	(384,),
 'block3.2.bottleneck.0.bn.running_mean',	(480,),	 '_blocks.10._bn0.running_mean',	(384,),
 'block3.2.bottleneck.0.bn.running_var',	(480,),	 '_blocks.10._bn0.running_var',	(384,),
 'block3.2.bottleneck.2.conv.weight',	(480, 1, 5, 5),	 '_blocks.10._depthwise_conv.weight',	(384, 1, 5, 5),
 'block3.2.bottleneck.2.bn.weight',	(480,),	 '_blocks.10._bn1.weight',	(384,),
 'block3.2.bottleneck.2.bn.bias',	(480,),	 '_blocks.10._bn1.bias',	(384,),
 'block3.2.bottleneck.2.bn.running_mean',	(480,),	 '_blocks.10._bn1.running_mean',	(384,),
 'block3.2.bottleneck.2.bn.running_var',	(480,),	 '_blocks.10._bn1.running_var',	(384,),
 'block3.2.bottleneck.4.squeeze.weight',	(20, 480, 1, 1),	 '_blocks.10._se_reduce.weight',	(16, 384, 1, 1),
 'block3.2.bottleneck.4.squeeze.bias',	(20,),	 '_blocks.10._se_reduce.bias',	(16,),
 'block3.2.bottleneck.4.excite.weight',	(480, 20, 1, 1),	 '_blocks.10._se_expand.weight',	(384, 16, 1, 1),
 'block3.2.bottleneck.4.excite.bias',	(480,),	 '_blocks.10._se_expand.bias',	(384,),
 'block3.2.bottleneck.5.conv.weight',	(80, 480, 1, 1),	 '_blocks.10._project_conv.weight',	(64, 384, 1, 1),
 'block3.2.bottleneck.5.bn.weight',	(80,),	 '_blocks.10._bn2.weight',	(64,),
 'block3.2.bottleneck.5.bn.bias',	(80,),	 '_blocks.10._bn2.bias',	(64,),
 'block3.2.bottleneck.5.bn.running_mean',	(80,),	 '_blocks.10._bn2.running_mean',	(64,),
 'block3.2.bottleneck.5.bn.running_var',	(80,),	 '_blocks.10._bn2.running_var',	(64,),
 'block3.3.bottleneck.0.conv.weight',	(480, 80, 1, 1),	 '_blocks.11._expand_conv.weight',	(384, 64, 1, 1),
 'block3.3.bottleneck.0.bn.weight',	(480,),	 '_blocks.11._bn0.weight',	(384,),
 'block3.3.bottleneck.0.bn.bias',	(480,),	 '_blocks.11._bn0.bias',	(384,),
 'block3.3.bottleneck.0.bn.running_mean',	(480,),	 '_blocks.11._bn0.running_mean',	(384,),
 'block3.3.bottleneck.0.bn.running_var',	(480,),	 '_blocks.11._bn0.running_var',	(384,),
 'block3.3.bottleneck.2.conv.weight',	(480, 1, 5, 5),	 '_blocks.11._depthwise_conv.weight',	(384, 1, 5, 5),
 'block3.3.bottleneck.2.bn.weight',	(480,),	 '_blocks.11._bn1.weight',	(384,),
 'block3.3.bottleneck.2.bn.bias',	(480,),	 '_blocks.11._bn1.bias',	(384,),
 'block3.3.bottleneck.2.bn.running_mean',	(480,),	 '_blocks.11._bn1.running_mean',	(384,),
 'block3.3.bottleneck.2.bn.running_var',	(480,),	 '_blocks.11._bn1.running_var',	(384,),
 'block3.3.bottleneck.4.squeeze.weight',	(20, 480, 1, 1),	 '_blocks.11._se_reduce.weight',	(16, 384, 1, 1),
 'block3.3.bottleneck.4.squeeze.bias',	(20,),	 '_blocks.11._se_reduce.bias',	(16,),
 'block3.3.bottleneck.4.excite.weight',	(480, 20, 1, 1),	 '_blocks.11._se_expand.weight',	(384, 16, 1, 1),
 'block3.3.bottleneck.4.excite.bias',	(480,),	 '_blocks.11._se_expand.bias',	(384,),
 'block3.3.bottleneck.5.conv.weight',	(80, 480, 1, 1),	 '_blocks.11._project_conv.weight',	(64, 384, 1, 1),
 'block3.3.bottleneck.5.bn.weight',	(80,),	 '_blocks.11._bn2.weight',	(64,),
 'block3.3.bottleneck.5.bn.bias',	(80,),	 '_blocks.11._bn2.bias',	(64,),
 'block3.3.bottleneck.5.bn.running_mean',	(80,),	 '_blocks.11._bn2.running_mean',	(64,),
 'block3.3.bottleneck.5.bn.running_var',	(80,),	 '_blocks.11._bn2.running_var',	(64,),
 'block3.4.bottleneck.0.conv.weight',	(480, 80, 1, 1),	 '_blocks.12._expand_conv.weight',	(384, 64, 1, 1),
 'block3.4.bottleneck.0.bn.weight',	(480,),	 '_blocks.12._bn0.weight',	(384,),
 'block3.4.bottleneck.0.bn.bias',	(480,),	 '_blocks.12._bn0.bias',	(384,),
 'block3.4.bottleneck.0.bn.running_mean',	(480,),	 '_blocks.12._bn0.running_mean',	(384,),
 'block3.4.bottleneck.0.bn.running_var',	(480,),	 '_blocks.12._bn0.running_var',	(384,),
 'block3.4.bottleneck.2.conv.weight',	(480, 1, 5, 5),	 '_blocks.12._depthwise_conv.weight',	(384, 1, 5, 5),
 'block3.4.bottleneck.2.bn.weight',	(480,),	 '_blocks.12._bn1.weight',	(384,),
 'block3.4.bottleneck.2.bn.bias',	(480,),	 '_blocks.12._bn1.bias',	(384,),
 'block3.4.bottleneck.2.bn.running_mean',	(480,),	 '_blocks.12._bn1.running_mean',	(384,),
 'block3.4.bottleneck.2.bn.running_var',	(480,),	 '_blocks.12._bn1.running_var',	(384,),
 'block3.4.bottleneck.4.squeeze.weight',	(20, 480, 1, 1),	 '_blocks.12._se_reduce.weight',	(16, 384, 1, 1),
 'block3.4.bottleneck.4.squeeze.bias',	(20,),	 '_blocks.12._se_reduce.bias',	(16,),
 'block3.4.bottleneck.4.excite.weight',	(480, 20, 1, 1),	 '_blocks.12._se_expand.weight',	(384, 16, 1, 1),
 'block3.4.bottleneck.4.excite.bias',	(480,),	 '_blocks.12._se_expand.bias',	(384,),
 'block3.4.bottleneck.5.conv.weight',	(80, 480, 1, 1),	 '_blocks.12._project_conv.weight',	(64, 384, 1, 1),
 'block3.4.bottleneck.5.bn.weight',	(80,),	 '_blocks.12._bn2.weight',	(64,),
 'block3.4.bottleneck.5.bn.bias',	(80,),	 '_blocks.12._bn2.bias',	(64,),
 'block3.4.bottleneck.5.bn.running_mean',	(80,),	 '_blocks.12._bn2.running_mean',	(64,),
 'block3.4.bottleneck.5.bn.running_var',	(80,),	 '_blocks.12._bn2.running_var',	(64,),
 'block4.0.bottleneck.0.conv.weight',	(480, 80, 1, 1),	 '_blocks.13._expand_conv.weight',	(384, 64, 1, 1),
 'block4.0.bottleneck.0.bn.weight',	(480,),	 '_blocks.13._bn0.weight',	(384,),
 'block4.0.bottleneck.0.bn.bias',	(480,),	 '_blocks.13._bn0.bias',	(384,),
 'block4.0.bottleneck.0.bn.running_mean',	(480,),	 '_blocks.13._bn0.running_mean',	(384,),
 'block4.0.bottleneck.0.bn.running_var',	(480,),	 '_blocks.13._bn0.running_var',	(384,),
 'block4.0.bottleneck.2.conv.weight',	(480, 1, 3, 3),	 '_blocks.13._depthwise_conv.weight',	(384, 1, 3, 3),
 'block4.0.bottleneck.2.bn.weight',	(480,),	 '_blocks.13._bn1.weight',	(384,),
 'block4.0.bottleneck.2.bn.bias',	(480,),	 '_blocks.13._bn1.bias',	(384,),
 'block4.0.bottleneck.2.bn.running_mean',	(480,),	 '_blocks.13._bn1.running_mean',	(384,),
 'block4.0.bottleneck.2.bn.running_var',	(480,),	 '_blocks.13._bn1.running_var',	(384,),
 'block4.0.bottleneck.4.squeeze.weight',	(20, 480, 1, 1),	 '_blocks.13._se_reduce.weight',	(16, 384, 1, 1),
 'block4.0.bottleneck.4.squeeze.bias',	(20,),	 '_blocks.13._se_reduce.bias',	(16,),
 'block4.0.bottleneck.4.excite.weight',	(480, 20, 1, 1),	 '_blocks.13._se_expand.weight',	(384, 16, 1, 1),
 'block4.0.bottleneck.4.excite.bias',	(480,),	 '_blocks.13._se_expand.bias',	(384,),
 'block4.0.bottleneck.5.conv.weight',	(160, 480, 1, 1),	 '_blocks.13._project_conv.weight',	(128, 384, 1, 1),
 'block4.0.bottleneck.5.bn.weight',	(160,),	 '_blocks.13._bn2.weight',	(128,),
 'block4.0.bottleneck.5.bn.bias',	(160,),	 '_blocks.13._bn2.bias',	(128,),
 'block4.0.bottleneck.5.bn.running_mean',	(160,),	 '_blocks.13._bn2.running_mean',	(128,),
 'block4.0.bottleneck.5.bn.running_var',	(160,),	 '_blocks.13._bn2.running_var',	(128,),
 'block4.1.bottleneck.0.conv.weight',	(960, 160, 1, 1),	 '_blocks.14._expand_conv.weight',	(768, 128, 1, 1),
 'block4.1.bottleneck.0.bn.weight',	(960,),	 '_blocks.14._bn0.weight',	(768,),
 'block4.1.bottleneck.0.bn.bias',	(960,),	 '_blocks.14._bn0.bias',	(768,),
 'block4.1.bottleneck.0.bn.running_mean',	(960,),	 '_blocks.14._bn0.running_mean',	(768,),
 'block4.1.bottleneck.0.bn.running_var',	(960,),	 '_blocks.14._bn0.running_var',	(768,),
 'block4.1.bottleneck.2.conv.weight',	(960, 1, 3, 3),	 '_blocks.14._depthwise_conv.weight',	(768, 1, 3, 3),
 'block4.1.bottleneck.2.bn.weight',	(960,),	 '_blocks.14._bn1.weight',	(768,),
 'block4.1.bottleneck.2.bn.bias',	(960,),	 '_blocks.14._bn1.bias',	(768,),
 'block4.1.bottleneck.2.bn.running_mean',	(960,),	 '_blocks.14._bn1.running_mean',	(768,),
 'block4.1.bottleneck.2.bn.running_var',	(960,),	 '_blocks.14._bn1.running_var',	(768,),
 'block4.1.bottleneck.4.squeeze.weight',	(40, 960, 1, 1),	 '_blocks.14._se_reduce.weight',	(32, 768, 1, 1),
 'block4.1.bottleneck.4.squeeze.bias',	(40,),	 '_blocks.14._se_reduce.bias',	(32,),
 'block4.1.bottleneck.4.excite.weight',	(960, 40, 1, 1),	 '_blocks.14._se_expand.weight',	(768, 32, 1, 1),
 'block4.1.bottleneck.4.excite.bias',	(960,),	 '_blocks.14._se_expand.bias',	(768,),
 'block4.1.bottleneck.5.conv.weight',	(160, 960, 1, 1),	 '_blocks.14._project_conv.weight',	(128, 768, 1, 1),
 'block4.1.bottleneck.5.bn.weight',	(160,),	 '_blocks.14._bn2.weight',	(128,),
 'block4.1.bottleneck.5.bn.bias',	(160,),	 '_blocks.14._bn2.bias',	(128,),
 'block4.1.bottleneck.5.bn.running_mean',	(160,),	 '_blocks.14._bn2.running_mean',	(128,),
 'block4.1.bottleneck.5.bn.running_var',	(160,),	 '_blocks.14._bn2.running_var',	(128,),
 'block4.2.bottleneck.0.conv.weight',	(960, 160, 1, 1),	 '_blocks.15._expand_conv.weight',	(768, 128, 1, 1),
 'block4.2.bottleneck.0.bn.weight',	(960,),	 '_blocks.15._bn0.weight',	(768,),
 'block4.2.bottleneck.0.bn.bias',	(960,),	 '_blocks.15._bn0.bias',	(768,),
 'block4.2.bottleneck.0.bn.running_mean',	(960,),	 '_blocks.15._bn0.running_mean',	(768,),
 'block4.2.bottleneck.0.bn.running_var',	(960,),	 '_blocks.15._bn0.running_var',	(768,),
 'block4.2.bottleneck.2.conv.weight',	(960, 1, 3, 3),	 '_blocks.15._depthwise_conv.weight',	(768, 1, 3, 3),
 'block4.2.bottleneck.2.bn.weight',	(960,),	 '_blocks.15._bn1.weight',	(768,),
 'block4.2.bottleneck.2.bn.bias',	(960,),	 '_blocks.15._bn1.bias',	(768,),
 'block4.2.bottleneck.2.bn.running_mean',	(960,),	 '_blocks.15._bn1.running_mean',	(768,),
 'block4.2.bottleneck.2.bn.running_var',	(960,),	 '_blocks.15._bn1.running_var',	(768,),
 'block4.2.bottleneck.4.squeeze.weight',	(40, 960, 1, 1),	 '_blocks.15._se_reduce.weight',	(32, 768, 1, 1),
 'block4.2.bottleneck.4.squeeze.bias',	(40,),	 '_blocks.15._se_reduce.bias',	(32,),
 'block4.2.bottleneck.4.excite.weight',	(960, 40, 1, 1),	 '_blocks.15._se_expand.weight',	(768, 32, 1, 1),
 'block4.2.bottleneck.4.excite.bias',	(960,),	 '_blocks.15._se_expand.bias',	(768,),
 'block4.2.bottleneck.5.conv.weight',	(160, 960, 1, 1),	 '_blocks.15._project_conv.weight',	(128, 768, 1, 1),
 'block4.2.bottleneck.5.bn.weight',	(160,),	 '_blocks.15._bn2.weight',	(128,),
 'block4.2.bottleneck.5.bn.bias',	(160,),	 '_blocks.15._bn2.bias',	(128,),
 'block4.2.bottleneck.5.bn.running_mean',	(160,),	 '_blocks.15._bn2.running_mean',	(128,),
 'block4.2.bottleneck.5.bn.running_var',	(160,),	 '_blocks.15._bn2.running_var',	(128,),
 'block4.3.bottleneck.0.conv.weight',	(960, 160, 1, 1),	 '_blocks.16._expand_conv.weight',	(768, 128, 1, 1),
 'block4.3.bottleneck.0.bn.weight',	(960,),	 '_blocks.16._bn0.weight',	(768,),
 'block4.3.bottleneck.0.bn.bias',	(960,),	 '_blocks.16._bn0.bias',	(768,),
 'block4.3.bottleneck.0.bn.running_mean',	(960,),	 '_blocks.16._bn0.running_mean',	(768,),
 'block4.3.bottleneck.0.bn.running_var',	(960,),	 '_blocks.16._bn0.running_var',	(768,),
 'block4.3.bottleneck.2.conv.weight',	(960, 1, 3, 3),	 '_blocks.16._depthwise_conv.weight',	(768, 1, 3, 3),
 'block4.3.bottleneck.2.bn.weight',	(960,),	 '_blocks.16._bn1.weight',	(768,),
 'block4.3.bottleneck.2.bn.bias',	(960,),	 '_blocks.16._bn1.bias',	(768,),
 'block4.3.bottleneck.2.bn.running_mean',	(960,),	 '_blocks.16._bn1.running_mean',	(768,),
 'block4.3.bottleneck.2.bn.running_var',	(960,),	 '_blocks.16._bn1.running_var',	(768,),
 'block4.3.bottleneck.4.squeeze.weight',	(40, 960, 1, 1),	 '_blocks.16._se_reduce.weight',	(32, 768, 1, 1),
 'block4.3.bottleneck.4.squeeze.bias',	(40,),	 '_blocks.16._se_reduce.bias',	(32,),
 'block4.3.bottleneck.4.excite.weight',	(960, 40, 1, 1),	 '_blocks.16._se_expand.weight',	(768, 32, 1, 1),
 'block4.3.bottleneck.4.excite.bias',	(960,),	 '_blocks.16._se_expand.bias',	(768,),
 'block4.3.bottleneck.5.conv.weight',	(160, 960, 1, 1),	 '_blocks.16._project_conv.weight',	(128, 768, 1, 1),
 'block4.3.bottleneck.5.bn.weight',	(160,),	 '_blocks.16._bn2.weight',	(128,),
 'block4.3.bottleneck.5.bn.bias',	(160,),	 '_blocks.16._bn2.bias',	(128,),
 'block4.3.bottleneck.5.bn.running_mean',	(160,),	 '_blocks.16._bn2.running_mean',	(128,),
 'block4.3.bottleneck.5.bn.running_var',	(160,),	 '_blocks.16._bn2.running_var',	(128,),
 'block4.4.bottleneck.0.conv.weight',	(960, 160, 1, 1),	 '_blocks.17._expand_conv.weight',	(768, 128, 1, 1),
 'block4.4.bottleneck.0.bn.weight',	(960,),	 '_blocks.17._bn0.weight',	(768,),
 'block4.4.bottleneck.0.bn.bias',	(960,),	 '_blocks.17._bn0.bias',	(768,),
 'block4.4.bottleneck.0.bn.running_mean',	(960,),	 '_blocks.17._bn0.running_mean',	(768,),
 'block4.4.bottleneck.0.bn.running_var',	(960,),	 '_blocks.17._bn0.running_var',	(768,),
 'block4.4.bottleneck.2.conv.weight',	(960, 1, 3, 3),	 '_blocks.17._depthwise_conv.weight',	(768, 1, 3, 3),
 'block4.4.bottleneck.2.bn.weight',	(960,),	 '_blocks.17._bn1.weight',	(768,),
 'block4.4.bottleneck.2.bn.bias',	(960,),	 '_blocks.17._bn1.bias',	(768,),
 'block4.4.bottleneck.2.bn.running_mean',	(960,),	 '_blocks.17._bn1.running_mean',	(768,),
 'block4.4.bottleneck.2.bn.running_var',	(960,),	 '_blocks.17._bn1.running_var',	(768,),
 'block4.4.bottleneck.4.squeeze.weight',	(40, 960, 1, 1),	 '_blocks.17._se_reduce.weight',	(32, 768, 1, 1),
 'block4.4.bottleneck.4.squeeze.bias',	(40,),	 '_blocks.17._se_reduce.bias',	(32,),
 'block4.4.bottleneck.4.excite.weight',	(960, 40, 1, 1),	 '_blocks.17._se_expand.weight',	(768, 32, 1, 1),
 'block4.4.bottleneck.4.excite.bias',	(960,),	 '_blocks.17._se_expand.bias',	(768,),
 'block4.4.bottleneck.5.conv.weight',	(160, 960, 1, 1),	 '_blocks.17._project_conv.weight',	(128, 768, 1, 1),
 'block4.4.bottleneck.5.bn.weight',	(160,),	 '_blocks.17._bn2.weight',	(128,),
 'block4.4.bottleneck.5.bn.bias',	(160,),	 '_blocks.17._bn2.bias',	(128,),
 'block4.4.bottleneck.5.bn.running_mean',	(160,),	 '_blocks.17._bn2.running_mean',	(128,),
 'block4.4.bottleneck.5.bn.running_var',	(160,),	 '_blocks.17._bn2.running_var',	(128,),
 'block4.5.bottleneck.0.conv.weight',	(960, 160, 1, 1),	 '_blocks.18._expand_conv.weight',	(768, 128, 1, 1),
 'block4.5.bottleneck.0.bn.weight',	(960,),	 '_blocks.18._bn0.weight',	(768,),
 'block4.5.bottleneck.0.bn.bias',	(960,),	 '_blocks.18._bn0.bias',	(768,),
 'block4.5.bottleneck.0.bn.running_mean',	(960,),	 '_blocks.18._bn0.running_mean',	(768,),
 'block4.5.bottleneck.0.bn.running_var',	(960,),	 '_blocks.18._bn0.running_var',	(768,),
 'block4.5.bottleneck.2.conv.weight',	(960, 1, 3, 3),	 '_blocks.18._depthwise_conv.weight',	(768, 1, 3, 3),
 'block4.5.bottleneck.2.bn.weight',	(960,),	 '_blocks.18._bn1.weight',	(768,),
 'block4.5.bottleneck.2.bn.bias',	(960,),	 '_blocks.18._bn1.bias',	(768,),
 'block4.5.bottleneck.2.bn.running_mean',	(960,),	 '_blocks.18._bn1.running_mean',	(768,),
 'block4.5.bottleneck.2.bn.running_var',	(960,),	 '_blocks.18._bn1.running_var',	(768,),
 'block4.5.bottleneck.4.squeeze.weight',	(40, 960, 1, 1),	 '_blocks.18._se_reduce.weight',	(32, 768, 1, 1),
 'block4.5.bottleneck.4.squeeze.bias',	(40,),	 '_blocks.18._se_reduce.bias',	(32,),
 'block4.5.bottleneck.4.excite.weight',	(960, 40, 1, 1),	 '_blocks.18._se_expand.weight',	(768, 32, 1, 1),
 'block4.5.bottleneck.4.excite.bias',	(960,),	 '_blocks.18._se_expand.bias',	(768,),
 'block4.5.bottleneck.5.conv.weight',	(160, 960, 1, 1),	 '_blocks.18._project_conv.weight',	(128, 768, 1, 1),
 'block4.5.bottleneck.5.bn.weight',	(160,),	 '_blocks.18._bn2.weight',	(128,),
 'block4.5.bottleneck.5.bn.bias',	(160,),	 '_blocks.18._bn2.bias',	(128,),
 'block4.5.bottleneck.5.bn.running_mean',	(160,),	 '_blocks.18._bn2.running_mean',	(128,),
 'block4.5.bottleneck.5.bn.running_var',	(160,),	 '_blocks.18._bn2.running_var',	(128,),
 'block4.6.bottleneck.0.conv.weight',	(960, 160, 1, 1),	 '_blocks.19._expand_conv.weight',	(768, 128, 1, 1),
 'block4.6.bottleneck.0.bn.weight',	(960,),	 '_blocks.19._bn0.weight',	(768,),
 'block4.6.bottleneck.0.bn.bias',	(960,),	 '_blocks.19._bn0.bias',	(768,),
 'block4.6.bottleneck.0.bn.running_mean',	(960,),	 '_blocks.19._bn0.running_mean',	(768,),
 'block4.6.bottleneck.0.bn.running_var',	(960,),	 '_blocks.19._bn0.running_var',	(768,),
 'block4.6.bottleneck.2.conv.weight',	(960, 1, 3, 3),	 '_blocks.19._depthwise_conv.weight',	(768, 1, 3, 3),
 'block4.6.bottleneck.2.bn.weight',	(960,),	 '_blocks.19._bn1.weight',	(768,),
 'block4.6.bottleneck.2.bn.bias',	(960,),	 '_blocks.19._bn1.bias',	(768,),
 'block4.6.bottleneck.2.bn.running_mean',	(960,),	 '_blocks.19._bn1.running_mean',	(768,),
 'block4.6.bottleneck.2.bn.running_var',	(960,),	 '_blocks.19._bn1.running_var',	(768,),
 'block4.6.bottleneck.4.squeeze.weight',	(40, 960, 1, 1),	 '_blocks.19._se_reduce.weight',	(32, 768, 1, 1),
 'block4.6.bottleneck.4.squeeze.bias',	(40,),	 '_blocks.19._se_reduce.bias',	(32,),
 'block4.6.bottleneck.4.excite.weight',	(960, 40, 1, 1),	 '_blocks.19._se_expand.weight',	(768, 32, 1, 1),
 'block4.6.bottleneck.4.excite.bias',	(960,),	 '_blocks.19._se_expand.bias',	(768,),
 'block4.6.bottleneck.5.conv.weight',	(160, 960, 1, 1),	 '_blocks.19._project_conv.weight',	(128, 768, 1, 1),
 'block4.6.bottleneck.5.bn.weight',	(160,),	 '_blocks.19._bn2.weight',	(128,),
 'block4.6.bottleneck.5.bn.bias',	(160,),	 '_blocks.19._bn2.bias',	(128,),
 'block4.6.bottleneck.5.bn.running_mean',	(160,),	 '_blocks.19._bn2.running_mean',	(128,),
 'block4.6.bottleneck.5.bn.running_var',	(160,),	 '_blocks.19._bn2.running_var',	(128,),
 'block5.0.bottleneck.0.conv.weight',	(960, 160, 1, 1),	 '_blocks.20._expand_conv.weight',	(768, 128, 1, 1),
 'block5.0.bottleneck.0.bn.weight',	(960,),	 '_blocks.20._bn0.weight',	(768,),
 'block5.0.bottleneck.0.bn.bias',	(960,),	 '_blocks.20._bn0.bias',	(768,),
 'block5.0.bottleneck.0.bn.running_mean',	(960,),	 '_blocks.20._bn0.running_mean',	(768,),
 'block5.0.bottleneck.0.bn.running_var',	(960,),	 '_blocks.20._bn0.running_var',	(768,),
 'block5.0.bottleneck.2.conv.weight',	(960, 1, 5, 5),	 '_blocks.20._depthwise_conv.weight',	(768, 1, 5, 5),
 'block5.0.bottleneck.2.bn.weight',	(960,),	 '_blocks.20._bn1.weight',	(768,),
 'block5.0.bottleneck.2.bn.bias',	(960,),	 '_blocks.20._bn1.bias',	(768,),
 'block5.0.bottleneck.2.bn.running_mean',	(960,),	 '_blocks.20._bn1.running_mean',	(768,),
 'block5.0.bottleneck.2.bn.running_var',	(960,),	 '_blocks.20._bn1.running_var',	(768,),
 'block5.0.bottleneck.4.squeeze.weight',	(40, 960, 1, 1),	 '_blocks.20._se_reduce.weight',	(32, 768, 1, 1),
 'block5.0.bottleneck.4.squeeze.bias',	(40,),	 '_blocks.20._se_reduce.bias',	(32,),
 'block5.0.bottleneck.4.excite.weight',	(960, 40, 1, 1),	 '_blocks.20._se_expand.weight',	(768, 32, 1, 1),
 'block5.0.bottleneck.4.excite.bias',	(960,),	 '_blocks.20._se_expand.bias',	(768,),
 'block5.0.bottleneck.5.conv.weight',	(224, 960, 1, 1),	 '_blocks.20._project_conv.weight',	(176, 768, 1, 1),
 'block5.0.bottleneck.5.bn.weight',	(224,),	 '_blocks.20._bn2.weight',	(176,),
 'block5.0.bottleneck.5.bn.bias',	(224,),	 '_blocks.20._bn2.bias',	(176,),
 'block5.0.bottleneck.5.bn.running_mean',	(224,),	 '_blocks.20._bn2.running_mean',	(176,),
 'block5.0.bottleneck.5.bn.running_var',	(224,),	 '_blocks.20._bn2.running_var',	(176,),
 'block5.1.bottleneck.0.conv.weight',	(1344, 224, 1, 1),	 '_blocks.21._expand_conv.weight',	(1056, 176, 1, 1),
 'block5.1.bottleneck.0.bn.weight',	(1344,),	 '_blocks.21._bn0.weight',	(1056,),
 'block5.1.bottleneck.0.bn.bias',	(1344,),	 '_blocks.21._bn0.bias',	(1056,),
 'block5.1.bottleneck.0.bn.running_mean',	(1344,),	 '_blocks.21._bn0.running_mean',	(1056,),
 'block5.1.bottleneck.0.bn.running_var',	(1344,),	 '_blocks.21._bn0.running_var',	(1056,),
 'block5.1.bottleneck.2.conv.weight',	(1344, 1, 5, 5),	 '_blocks.21._depthwise_conv.weight',	(1056, 1, 5, 5),
 'block5.1.bottleneck.2.bn.weight',	(1344,),	 '_blocks.21._bn1.weight',	(1056,),
 'block5.1.bottleneck.2.bn.bias',	(1344,),	 '_blocks.21._bn1.bias',	(1056,),
 'block5.1.bottleneck.2.bn.running_mean',	(1344,),	 '_blocks.21._bn1.running_mean',	(1056,),
 'block5.1.bottleneck.2.bn.running_var',	(1344,),	 '_blocks.21._bn1.running_var',	(1056,),
 'block5.1.bottleneck.4.squeeze.weight',	(56, 1344, 1, 1),	 '_blocks.21._se_reduce.weight',	(44, 1056, 1, 1),
 'block5.1.bottleneck.4.squeeze.bias',	(56,),	 '_blocks.21._se_reduce.bias',	(44,),
 'block5.1.bottleneck.4.excite.weight',	(1344, 56, 1, 1),	 '_blocks.21._se_expand.weight',	(1056, 44, 1, 1),
 'block5.1.bottleneck.4.excite.bias',	(1344,),	 '_blocks.21._se_expand.bias',	(1056,),
 'block5.1.bottleneck.5.conv.weight',	(224, 1344, 1, 1),	 '_blocks.21._project_conv.weight',	(176, 1056, 1, 1),
 'block5.1.bottleneck.5.bn.weight',	(224,),	 '_blocks.21._bn2.weight',	(176,),
 'block5.1.bottleneck.5.bn.bias',	(224,),	 '_blocks.21._bn2.bias',	(176,),
 'block5.1.bottleneck.5.bn.running_mean',	(224,),	 '_blocks.21._bn2.running_mean',	(176,),
 'block5.1.bottleneck.5.bn.running_var',	(224,),	 '_blocks.21._bn2.running_var',	(176,),
 'block5.2.bottleneck.0.conv.weight',	(1344, 224, 1, 1),	 '_blocks.22._expand_conv.weight',	(1056, 176, 1, 1),
 'block5.2.bottleneck.0.bn.weight',	(1344,),	 '_blocks.22._bn0.weight',	(1056,),
 'block5.2.bottleneck.0.bn.bias',	(1344,),	 '_blocks.22._bn0.bias',	(1056,),
 'block5.2.bottleneck.0.bn.running_mean',	(1344,),	 '_blocks.22._bn0.running_mean',	(1056,),
 'block5.2.bottleneck.0.bn.running_var',	(1344,),	 '_blocks.22._bn0.running_var',	(1056,),
 'block5.2.bottleneck.2.conv.weight',	(1344, 1, 5, 5),	 '_blocks.22._depthwise_conv.weight',	(1056, 1, 5, 5),
 'block5.2.bottleneck.2.bn.weight',	(1344,),	 '_blocks.22._bn1.weight',	(1056,),
 'block5.2.bottleneck.2.bn.bias',	(1344,),	 '_blocks.22._bn1.bias',	(1056,),
 'block5.2.bottleneck.2.bn.running_mean',	(1344,),	 '_blocks.22._bn1.running_mean',	(1056,),
 'block5.2.bottleneck.2.bn.running_var',	(1344,),	 '_blocks.22._bn1.running_var',	(1056,),
 'block5.2.bottleneck.4.squeeze.weight',	(56, 1344, 1, 1),	 '_blocks.22._se_reduce.weight',	(44, 1056, 1, 1),
 'block5.2.bottleneck.4.squeeze.bias',	(56,),	 '_blocks.22._se_reduce.bias',	(44,),
 'block5.2.bottleneck.4.excite.weight',	(1344, 56, 1, 1),	 '_blocks.22._se_expand.weight',	(1056, 44, 1, 1),
 'block5.2.bottleneck.4.excite.bias',	(1344,),	 '_blocks.22._se_expand.bias',	(1056,),
 'block5.2.bottleneck.5.conv.weight',	(224, 1344, 1, 1),	 '_blocks.22._project_conv.weight',	(176, 1056, 1, 1),
 'block5.2.bottleneck.5.bn.weight',	(224,),	 '_blocks.22._bn2.weight',	(176,),
 'block5.2.bottleneck.5.bn.bias',	(224,),	 '_blocks.22._bn2.bias',	(176,),
 'block5.2.bottleneck.5.bn.running_mean',	(224,),	 '_blocks.22._bn2.running_mean',	(176,),
 'block5.2.bottleneck.5.bn.running_var',	(224,),	 '_blocks.22._bn2.running_var',	(176,),
 'block5.3.bottleneck.0.conv.weight',	(1344, 224, 1, 1),	 '_blocks.23._expand_conv.weight',	(1056, 176, 1, 1),
 'block5.3.bottleneck.0.bn.weight',	(1344,),	 '_blocks.23._bn0.weight',	(1056,),
 'block5.3.bottleneck.0.bn.bias',	(1344,),	 '_blocks.23._bn0.bias',	(1056,),
 'block5.3.bottleneck.0.bn.running_mean',	(1344,),	 '_blocks.23._bn0.running_mean',	(1056,),
 'block5.3.bottleneck.0.bn.running_var',	(1344,),	 '_blocks.23._bn0.running_var',	(1056,),
 'block5.3.bottleneck.2.conv.weight',	(1344, 1, 5, 5),	 '_blocks.23._depthwise_conv.weight',	(1056, 1, 5, 5),
 'block5.3.bottleneck.2.bn.weight',	(1344,),	 '_blocks.23._bn1.weight',	(1056,),
 'block5.3.bottleneck.2.bn.bias',	(1344,),	 '_blocks.23._bn1.bias',	(1056,),
 'block5.3.bottleneck.2.bn.running_mean',	(1344,),	 '_blocks.23._bn1.running_mean',	(1056,),
 'block5.3.bottleneck.2.bn.running_var',	(1344,),	 '_blocks.23._bn1.running_var',	(1056,),
 'block5.3.bottleneck.4.squeeze.weight',	(56, 1344, 1, 1),	 '_blocks.23._se_reduce.weight',	(44, 1056, 1, 1),
 'block5.3.bottleneck.4.squeeze.bias',	(56,),	 '_blocks.23._se_reduce.bias',	(44,),
 'block5.3.bottleneck.4.excite.weight',	(1344, 56, 1, 1),	 '_blocks.23._se_expand.weight',	(1056, 44, 1, 1),
 'block5.3.bottleneck.4.excite.bias',	(1344,),	 '_blocks.23._se_expand.bias',	(1056,),
 'block5.3.bottleneck.5.conv.weight',	(224, 1344, 1, 1),	 '_blocks.23._project_conv.weight',	(176, 1056, 1, 1),
 'block5.3.bottleneck.5.bn.weight',	(224,),	 '_blocks.23._bn2.weight',	(176,),
 'block5.3.bottleneck.5.bn.bias',	(224,),	 '_blocks.23._bn2.bias',	(176,),
 'block5.3.bottleneck.5.bn.running_mean',	(224,),	 '_blocks.23._bn2.running_mean',	(176,),
 'block5.3.bottleneck.5.bn.running_var',	(224,),	 '_blocks.23._bn2.running_var',	(176,),
 'block5.4.bottleneck.0.conv.weight',	(1344, 224, 1, 1),	 '_blocks.24._expand_conv.weight',	(1056, 176, 1, 1),
 'block5.4.bottleneck.0.bn.weight',	(1344,),	 '_blocks.24._bn0.weight',	(1056,),
 'block5.4.bottleneck.0.bn.bias',	(1344,),	 '_blocks.24._bn0.bias',	(1056,),
 'block5.4.bottleneck.0.bn.running_mean',	(1344,),	 '_blocks.24._bn0.running_mean',	(1056,),
 'block5.4.bottleneck.0.bn.running_var',	(1344,),	 '_blocks.24._bn0.running_var',	(1056,),
 'block5.4.bottleneck.2.conv.weight',	(1344, 1, 5, 5),	 '_blocks.24._depthwise_conv.weight',	(1056, 1, 5, 5),
 'block5.4.bottleneck.2.bn.weight',	(1344,),	 '_blocks.24._bn1.weight',	(1056,),
 'block5.4.bottleneck.2.bn.bias',	(1344,),	 '_blocks.24._bn1.bias',	(1056,),
 'block5.4.bottleneck.2.bn.running_mean',	(1344,),	 '_blocks.24._bn1.running_mean',	(1056,),
 'block5.4.bottleneck.2.bn.running_var',	(1344,),	 '_blocks.24._bn1.running_var',	(1056,),
 'block5.4.bottleneck.4.squeeze.weight',	(56, 1344, 1, 1),	 '_blocks.24._se_reduce.weight',	(44, 1056, 1, 1),
 'block5.4.bottleneck.4.squeeze.bias',	(56,),	 '_blocks.24._se_reduce.bias',	(44,),
 'block5.4.bottleneck.4.excite.weight',	(1344, 56, 1, 1),	 '_blocks.24._se_expand.weight',	(1056, 44, 1, 1),
 'block5.4.bottleneck.4.excite.bias',	(1344,),	 '_blocks.24._se_expand.bias',	(1056,),
 'block5.4.bottleneck.5.conv.weight',	(224, 1344, 1, 1),	 '_blocks.24._project_conv.weight',	(176, 1056, 1, 1),
 'block5.4.bottleneck.5.bn.weight',	(224,),	 '_blocks.24._bn2.weight',	(176,),
 'block5.4.bottleneck.5.bn.bias',	(224,),	 '_blocks.24._bn2.bias',	(176,),
 'block5.4.bottleneck.5.bn.running_mean',	(224,),	 '_blocks.24._bn2.running_mean',	(176,),
 'block5.4.bottleneck.5.bn.running_var',	(224,),	 '_blocks.24._bn2.running_var',	(176,),
 'block5.5.bottleneck.0.conv.weight',	(1344, 224, 1, 1),	 '_blocks.25._expand_conv.weight',	(1056, 176, 1, 1),
 'block5.5.bottleneck.0.bn.weight',	(1344,),	 '_blocks.25._bn0.weight',	(1056,),
 'block5.5.bottleneck.0.bn.bias',	(1344,),	 '_blocks.25._bn0.bias',	(1056,),
 'block5.5.bottleneck.0.bn.running_mean',	(1344,),	 '_blocks.25._bn0.running_mean',	(1056,),
 'block5.5.bottleneck.0.bn.running_var',	(1344,),	 '_blocks.25._bn0.running_var',	(1056,),
 'block5.5.bottleneck.2.conv.weight',	(1344, 1, 5, 5),	 '_blocks.25._depthwise_conv.weight',	(1056, 1, 5, 5),
 'block5.5.bottleneck.2.bn.weight',	(1344,),	 '_blocks.25._bn1.weight',	(1056,),
 'block5.5.bottleneck.2.bn.bias',	(1344,),	 '_blocks.25._bn1.bias',	(1056,),
 'block5.5.bottleneck.2.bn.running_mean',	(1344,),	 '_blocks.25._bn1.running_mean',	(1056,),
 'block5.5.bottleneck.2.bn.running_var',	(1344,),	 '_blocks.25._bn1.running_var',	(1056,),
 'block5.5.bottleneck.4.squeeze.weight',	(56, 1344, 1, 1),	 '_blocks.25._se_reduce.weight',	(44, 1056, 1, 1),
 'block5.5.bottleneck.4.squeeze.bias',	(56,),	 '_blocks.25._se_reduce.bias',	(44,),
 'block5.5.bottleneck.4.excite.weight',	(1344, 56, 1, 1),	 '_blocks.25._se_expand.weight',	(1056, 44, 1, 1),
 'block5.5.bottleneck.4.excite.bias',	(1344,),	 '_blocks.25._se_expand.bias',	(1056,),
 'block5.5.bottleneck.5.conv.weight',	(224, 1344, 1, 1),	 '_blocks.25._project_conv.weight',	(176, 1056, 1, 1),
 'block5.5.bottleneck.5.bn.weight',	(224,),	 '_blocks.25._bn2.weight',	(176,),
 'block5.5.bottleneck.5.bn.bias',	(224,),	 '_blocks.25._bn2.bias',	(176,),
 'block5.5.bottleneck.5.bn.running_mean',	(224,),	 '_blocks.25._bn2.running_mean',	(176,),
 'block5.5.bottleneck.5.bn.running_var',	(224,),	 '_blocks.25._bn2.running_var',	(176,),
 'block5.6.bottleneck.0.conv.weight',	(1344, 224, 1, 1),	 '_blocks.26._expand_conv.weight',	(1056, 176, 1, 1),
 'block5.6.bottleneck.0.bn.weight',	(1344,),	 '_blocks.26._bn0.weight',	(1056,),
 'block5.6.bottleneck.0.bn.bias',	(1344,),	 '_blocks.26._bn0.bias',	(1056,),
 'block5.6.bottleneck.0.bn.running_mean',	(1344,),	 '_blocks.26._bn0.running_mean',	(1056,),
 'block5.6.bottleneck.0.bn.running_var',	(1344,),	 '_blocks.26._bn0.running_var',	(1056,),
 'block5.6.bottleneck.2.conv.weight',	(1344, 1, 5, 5),	 '_blocks.26._depthwise_conv.weight',	(1056, 1, 5, 5),
 'block5.6.bottleneck.2.bn.weight',	(1344,),	 '_blocks.26._bn1.weight',	(1056,),
 'block5.6.bottleneck.2.bn.bias',	(1344,),	 '_blocks.26._bn1.bias',	(1056,),
 'block5.6.bottleneck.2.bn.running_mean',	(1344,),	 '_blocks.26._bn1.running_mean',	(1056,),
 'block5.6.bottleneck.2.bn.running_var',	(1344,),	 '_blocks.26._bn1.running_var',	(1056,),
 'block5.6.bottleneck.4.squeeze.weight',	(56, 1344, 1, 1),	 '_blocks.26._se_reduce.weight',	(44, 1056, 1, 1),
 'block5.6.bottleneck.4.squeeze.bias',	(56,),	 '_blocks.26._se_reduce.bias',	(44,),
 'block5.6.bottleneck.4.excite.weight',	(1344, 56, 1, 1),	 '_blocks.26._se_expand.weight',	(1056, 44, 1, 1),
 'block5.6.bottleneck.4.excite.bias',	(1344,),	 '_blocks.26._se_expand.bias',	(1056,),
 'block5.6.bottleneck.5.conv.weight',	(224, 1344, 1, 1),	 '_blocks.26._project_conv.weight',	(176, 1056, 1, 1),
 'block5.6.bottleneck.5.bn.weight',	(224,),	 '_blocks.26._bn2.weight',	(176,),
 'block5.6.bottleneck.5.bn.bias',	(224,),	 '_blocks.26._bn2.bias',	(176,),
 'block5.6.bottleneck.5.bn.running_mean',	(224,),	 '_blocks.26._bn2.running_mean',	(176,),
 'block5.6.bottleneck.5.bn.running_var',	(224,),	 '_blocks.26._bn2.running_var',	(176,),
 'block6.0.bottleneck.0.conv.weight',	(1344, 224, 1, 1),	 '_blocks.27._expand_conv.weight',	(1056, 176, 1, 1),
 'block6.0.bottleneck.0.bn.weight',	(1344,),	 '_blocks.27._bn0.weight',	(1056,),
 'block6.0.bottleneck.0.bn.bias',	(1344,),	 '_blocks.27._bn0.bias',	(1056,),
 'block6.0.bottleneck.0.bn.running_mean',	(1344,),	 '_blocks.27._bn0.running_mean',	(1056,),
 'block6.0.bottleneck.0.bn.running_var',	(1344,),	 '_blocks.27._bn0.running_var',	(1056,),
 'block6.0.bottleneck.2.conv.weight',	(1344, 1, 5, 5),	 '_blocks.27._depthwise_conv.weight',	(1056, 1, 5, 5),
 'block6.0.bottleneck.2.bn.weight',	(1344,),	 '_blocks.27._bn1.weight',	(1056,),
 'block6.0.bottleneck.2.bn.bias',	(1344,),	 '_blocks.27._bn1.bias',	(1056,),
 'block6.0.bottleneck.2.bn.running_mean',	(1344,),	 '_blocks.27._bn1.running_mean',	(1056,),
 'block6.0.bottleneck.2.bn.running_var',	(1344,),	 '_blocks.27._bn1.running_var',	(1056,),
 'block6.0.bottleneck.4.squeeze.weight',	(56, 1344, 1, 1),	 '_blocks.27._se_reduce.weight',	(44, 1056, 1, 1),
 'block6.0.bottleneck.4.squeeze.bias',	(56,),	 '_blocks.27._se_reduce.bias',	(44,),
 'block6.0.bottleneck.4.excite.weight',	(1344, 56, 1, 1),	 '_blocks.27._se_expand.weight',	(1056, 44, 1, 1),
 'block6.0.bottleneck.4.excite.bias',	(1344,),	 '_blocks.27._se_expand.bias',	(1056,),
 'block6.0.bottleneck.5.conv.weight',	(384, 1344, 1, 1),	 '_blocks.27._project_conv.weight',	(304, 1056, 1, 1),
 'block6.0.bottleneck.5.bn.weight',	(384,),	 '_blocks.27._bn2.weight',	(304,),
 'block6.0.bottleneck.5.bn.bias',	(384,),	 '_blocks.27._bn2.bias',	(304,),
 'block6.0.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.27._bn2.running_mean',	(304,),
 'block6.0.bottleneck.5.bn.running_var',	(384,),	 '_blocks.27._bn2.running_var',	(304,),
 'block6.1.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.28._expand_conv.weight',	(1824, 304, 1, 1),
 'block6.1.bottleneck.0.bn.weight',	(2304,),	 '_blocks.28._bn0.weight',	(1824,),
 'block6.1.bottleneck.0.bn.bias',	(2304,),	 '_blocks.28._bn0.bias',	(1824,),
 'block6.1.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.28._bn0.running_mean',	(1824,),
 'block6.1.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.28._bn0.running_var',	(1824,),
 'block6.1.bottleneck.2.conv.weight',	(2304, 1, 5, 5),	 '_blocks.28._depthwise_conv.weight',	(1824, 1, 5, 5),
 'block6.1.bottleneck.2.bn.weight',	(2304,),	 '_blocks.28._bn1.weight',	(1824,),
 'block6.1.bottleneck.2.bn.bias',	(2304,),	 '_blocks.28._bn1.bias',	(1824,),
 'block6.1.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.28._bn1.running_mean',	(1824,),
 'block6.1.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.28._bn1.running_var',	(1824,),
 'block6.1.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.28._se_reduce.weight',	(76, 1824, 1, 1),
 'block6.1.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.28._se_reduce.bias',	(76,),
 'block6.1.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.28._se_expand.weight',	(1824, 76, 1, 1),
 'block6.1.bottleneck.4.excite.bias',	(2304,),	 '_blocks.28._se_expand.bias',	(1824,),
 'block6.1.bottleneck.5.conv.weight',	(384, 2304, 1, 1),	 '_blocks.28._project_conv.weight',	(304, 1824, 1, 1),
 'block6.1.bottleneck.5.bn.weight',	(384,),	 '_blocks.28._bn2.weight',	(304,),
 'block6.1.bottleneck.5.bn.bias',	(384,),	 '_blocks.28._bn2.bias',	(304,),
 'block6.1.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.28._bn2.running_mean',	(304,),
 'block6.1.bottleneck.5.bn.running_var',	(384,),	 '_blocks.28._bn2.running_var',	(304,),
 'block6.2.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.29._expand_conv.weight',	(1824, 304, 1, 1),
 'block6.2.bottleneck.0.bn.weight',	(2304,),	 '_blocks.29._bn0.weight',	(1824,),
 'block6.2.bottleneck.0.bn.bias',	(2304,),	 '_blocks.29._bn0.bias',	(1824,),
 'block6.2.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.29._bn0.running_mean',	(1824,),
 'block6.2.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.29._bn0.running_var',	(1824,),
 'block6.2.bottleneck.2.conv.weight',	(2304, 1, 5, 5),	 '_blocks.29._depthwise_conv.weight',	(1824, 1, 5, 5),
 'block6.2.bottleneck.2.bn.weight',	(2304,),	 '_blocks.29._bn1.weight',	(1824,),
 'block6.2.bottleneck.2.bn.bias',	(2304,),	 '_blocks.29._bn1.bias',	(1824,),
 'block6.2.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.29._bn1.running_mean',	(1824,),
 'block6.2.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.29._bn1.running_var',	(1824,),
 'block6.2.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.29._se_reduce.weight',	(76, 1824, 1, 1),
 'block6.2.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.29._se_reduce.bias',	(76,),
 'block6.2.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.29._se_expand.weight',	(1824, 76, 1, 1),
 'block6.2.bottleneck.4.excite.bias',	(2304,),	 '_blocks.29._se_expand.bias',	(1824,),
 'block6.2.bottleneck.5.conv.weight',	(384, 2304, 1, 1),	 '_blocks.29._project_conv.weight',	(304, 1824, 1, 1),
 'block6.2.bottleneck.5.bn.weight',	(384,),	 '_blocks.29._bn2.weight',	(304,),
 'block6.2.bottleneck.5.bn.bias',	(384,),	 '_blocks.29._bn2.bias',	(304,),
 'block6.2.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.29._bn2.running_mean',	(304,),
 'block6.2.bottleneck.5.bn.running_var',	(384,),	 '_blocks.29._bn2.running_var',	(304,),
 'block6.3.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.30._expand_conv.weight',	(1824, 304, 1, 1),
 'block6.3.bottleneck.0.bn.weight',	(2304,),	 '_blocks.30._bn0.weight',	(1824,),
 'block6.3.bottleneck.0.bn.bias',	(2304,),	 '_blocks.30._bn0.bias',	(1824,),
 'block6.3.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.30._bn0.running_mean',	(1824,),
 'block6.3.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.30._bn0.running_var',	(1824,),
 'block6.3.bottleneck.2.conv.weight',	(2304, 1, 5, 5),	 '_blocks.30._depthwise_conv.weight',	(1824, 1, 5, 5),
 'block6.3.bottleneck.2.bn.weight',	(2304,),	 '_blocks.30._bn1.weight',	(1824,),
 'block6.3.bottleneck.2.bn.bias',	(2304,),	 '_blocks.30._bn1.bias',	(1824,),
 'block6.3.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.30._bn1.running_mean',	(1824,),
 'block6.3.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.30._bn1.running_var',	(1824,),
 'block6.3.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.30._se_reduce.weight',	(76, 1824, 1, 1),
 'block6.3.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.30._se_reduce.bias',	(76,),
 'block6.3.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.30._se_expand.weight',	(1824, 76, 1, 1),
 'block6.3.bottleneck.4.excite.bias',	(2304,),	 '_blocks.30._se_expand.bias',	(1824,),
 'block6.3.bottleneck.5.conv.weight',	(384, 2304, 1, 1),	 '_blocks.30._project_conv.weight',	(304, 1824, 1, 1),
 'block6.3.bottleneck.5.bn.weight',	(384,),	 '_blocks.30._bn2.weight',	(304,),
 'block6.3.bottleneck.5.bn.bias',	(384,),	 '_blocks.30._bn2.bias',	(304,),
 'block6.3.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.30._bn2.running_mean',	(304,),
 'block6.3.bottleneck.5.bn.running_var',	(384,),	 '_blocks.30._bn2.running_var',	(304,),
 'block6.4.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.31._expand_conv.weight',	(1824, 304, 1, 1),
 'block6.4.bottleneck.0.bn.weight',	(2304,),	 '_blocks.31._bn0.weight',	(1824,),
 'block6.4.bottleneck.0.bn.bias',	(2304,),	 '_blocks.31._bn0.bias',	(1824,),
 'block6.4.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.31._bn0.running_mean',	(1824,),
 'block6.4.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.31._bn0.running_var',	(1824,),
 'block6.4.bottleneck.2.conv.weight',	(2304, 1, 5, 5),	 '_blocks.31._depthwise_conv.weight',	(1824, 1, 5, 5),
 'block6.4.bottleneck.2.bn.weight',	(2304,),	 '_blocks.31._bn1.weight',	(1824,),
 'block6.4.bottleneck.2.bn.bias',	(2304,),	 '_blocks.31._bn1.bias',	(1824,),
 'block6.4.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.31._bn1.running_mean',	(1824,),
 'block6.4.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.31._bn1.running_var',	(1824,),
 'block6.4.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.31._se_reduce.weight',	(76, 1824, 1, 1),
 'block6.4.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.31._se_reduce.bias',	(76,),
 'block6.4.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.31._se_expand.weight',	(1824, 76, 1, 1),
 'block6.4.bottleneck.4.excite.bias',	(2304,),	 '_blocks.31._se_expand.bias',	(1824,),
 'block6.4.bottleneck.5.conv.weight',	(384, 2304, 1, 1),	 '_blocks.31._project_conv.weight',	(304, 1824, 1, 1),
 'block6.4.bottleneck.5.bn.weight',	(384,),	 '_blocks.31._bn2.weight',	(304,),
 'block6.4.bottleneck.5.bn.bias',	(384,),	 '_blocks.31._bn2.bias',	(304,),
 'block6.4.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.31._bn2.running_mean',	(304,),
 'block6.4.bottleneck.5.bn.running_var',	(384,),	 '_blocks.31._bn2.running_var',	(304,),
 'block6.5.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.32._expand_conv.weight',	(1824, 304, 1, 1),
 'block6.5.bottleneck.0.bn.weight',	(2304,),	 '_blocks.32._bn0.weight',	(1824,),
 'block6.5.bottleneck.0.bn.bias',	(2304,),	 '_blocks.32._bn0.bias',	(1824,),
 'block6.5.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.32._bn0.running_mean',	(1824,),
 'block6.5.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.32._bn0.running_var',	(1824,),
 'block6.5.bottleneck.2.conv.weight',	(2304, 1, 5, 5),	 '_blocks.32._depthwise_conv.weight',	(1824, 1, 5, 5),
 'block6.5.bottleneck.2.bn.weight',	(2304,),	 '_blocks.32._bn1.weight',	(1824,),
 'block6.5.bottleneck.2.bn.bias',	(2304,),	 '_blocks.32._bn1.bias',	(1824,),
 'block6.5.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.32._bn1.running_mean',	(1824,),
 'block6.5.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.32._bn1.running_var',	(1824,),
 'block6.5.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.32._se_reduce.weight',	(76, 1824, 1, 1),
 'block6.5.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.32._se_reduce.bias',	(76,),
 'block6.5.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.32._se_expand.weight',	(1824, 76, 1, 1),
 'block6.5.bottleneck.4.excite.bias',	(2304,),	 '_blocks.32._se_expand.bias',	(1824,),
 'block6.5.bottleneck.5.conv.weight',	(384, 2304, 1, 1),	 '_blocks.32._project_conv.weight',	(304, 1824, 1, 1),
 'block6.5.bottleneck.5.bn.weight',	(384,),	 '_blocks.32._bn2.weight',	(304,),
 'block6.5.bottleneck.5.bn.bias',	(384,),	 '_blocks.32._bn2.bias',	(304,),
 'block6.5.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.32._bn2.running_mean',	(304,),
 'block6.5.bottleneck.5.bn.running_var',	(384,),	 '_blocks.32._bn2.running_var',	(304,),
 'block6.6.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.33._expand_conv.weight',	(1824, 304, 1, 1),
 'block6.6.bottleneck.0.bn.weight',	(2304,),	 '_blocks.33._bn0.weight',	(1824,),
 'block6.6.bottleneck.0.bn.bias',	(2304,),	 '_blocks.33._bn0.bias',	(1824,),
 'block6.6.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.33._bn0.running_mean',	(1824,),
 'block6.6.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.33._bn0.running_var',	(1824,),
 'block6.6.bottleneck.2.conv.weight',	(2304, 1, 5, 5),	 '_blocks.33._depthwise_conv.weight',	(1824, 1, 5, 5),
 'block6.6.bottleneck.2.bn.weight',	(2304,),	 '_blocks.33._bn1.weight',	(1824,),
 'block6.6.bottleneck.2.bn.bias',	(2304,),	 '_blocks.33._bn1.bias',	(1824,),
 'block6.6.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.33._bn1.running_mean',	(1824,),
 'block6.6.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.33._bn1.running_var',	(1824,),
 'block6.6.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.33._se_reduce.weight',	(76, 1824, 1, 1),
 'block6.6.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.33._se_reduce.bias',	(76,),
 'block6.6.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.33._se_expand.weight',	(1824, 76, 1, 1),
 'block6.6.bottleneck.4.excite.bias',	(2304,),	 '_blocks.33._se_expand.bias',	(1824,),
 'block6.6.bottleneck.5.conv.weight',	(384, 2304, 1, 1),	 '_blocks.33._project_conv.weight',	(304, 1824, 1, 1),
 'block6.6.bottleneck.5.bn.weight',	(384,),	 '_blocks.33._bn2.weight',	(304,),
 'block6.6.bottleneck.5.bn.bias',	(384,),	 '_blocks.33._bn2.bias',	(304,),
 'block6.6.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.33._bn2.running_mean',	(304,),
 'block6.6.bottleneck.5.bn.running_var',	(384,),	 '_blocks.33._bn2.running_var',	(304,),
 'block6.7.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.34._expand_conv.weight',	(1824, 304, 1, 1),
 'block6.7.bottleneck.0.bn.weight',	(2304,),	 '_blocks.34._bn0.weight',	(1824,),
 'block6.7.bottleneck.0.bn.bias',	(2304,),	 '_blocks.34._bn0.bias',	(1824,),
 'block6.7.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.34._bn0.running_mean',	(1824,),
 'block6.7.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.34._bn0.running_var',	(1824,),
 'block6.7.bottleneck.2.conv.weight',	(2304, 1, 5, 5),	 '_blocks.34._depthwise_conv.weight',	(1824, 1, 5, 5),
 'block6.7.bottleneck.2.bn.weight',	(2304,),	 '_blocks.34._bn1.weight',	(1824,),
 'block6.7.bottleneck.2.bn.bias',	(2304,),	 '_blocks.34._bn1.bias',	(1824,),
 'block6.7.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.34._bn1.running_mean',	(1824,),
 'block6.7.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.34._bn1.running_var',	(1824,),
 'block6.7.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.34._se_reduce.weight',	(76, 1824, 1, 1),
 'block6.7.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.34._se_reduce.bias',	(76,),
 'block6.7.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.34._se_expand.weight',	(1824, 76, 1, 1),
 'block6.7.bottleneck.4.excite.bias',	(2304,),	 '_blocks.34._se_expand.bias',	(1824,),
 'block6.7.bottleneck.5.conv.weight',	(384, 2304, 1, 1),	 '_blocks.34._project_conv.weight',	(304, 1824, 1, 1),
 'block6.7.bottleneck.5.bn.weight',	(384,),	 '_blocks.34._bn2.weight',	(304,),
 'block6.7.bottleneck.5.bn.bias',	(384,),	 '_blocks.34._bn2.bias',	(304,),
 'block6.7.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.34._bn2.running_mean',	(304,),
 'block6.7.bottleneck.5.bn.running_var',	(384,),	 '_blocks.34._bn2.running_var',	(304,),
 'block6.8.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.35._expand_conv.weight',	(1824, 304, 1, 1),
 'block6.8.bottleneck.0.bn.weight',	(2304,),	 '_blocks.35._bn0.weight',	(1824,),
 'block6.8.bottleneck.0.bn.bias',	(2304,),	 '_blocks.35._bn0.bias',	(1824,),
 'block6.8.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.35._bn0.running_mean',	(1824,),
 'block6.8.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.35._bn0.running_var',	(1824,),
 'block6.8.bottleneck.2.conv.weight',	(2304, 1, 5, 5),	 '_blocks.35._depthwise_conv.weight',	(1824, 1, 5, 5),
 'block6.8.bottleneck.2.bn.weight',	(2304,),	 '_blocks.35._bn1.weight',	(1824,),
 'block6.8.bottleneck.2.bn.bias',	(2304,),	 '_blocks.35._bn1.bias',	(1824,),
 'block6.8.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.35._bn1.running_mean',	(1824,),
 'block6.8.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.35._bn1.running_var',	(1824,),
 'block6.8.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.35._se_reduce.weight',	(76, 1824, 1, 1),
 'block6.8.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.35._se_reduce.bias',	(76,),
 'block6.8.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.35._se_expand.weight',	(1824, 76, 1, 1),
 'block6.8.bottleneck.4.excite.bias',	(2304,),	 '_blocks.35._se_expand.bias',	(1824,),
 'block6.8.bottleneck.5.conv.weight',	(384, 2304, 1, 1),	 '_blocks.35._project_conv.weight',	(304, 1824, 1, 1),
 'block6.8.bottleneck.5.bn.weight',	(384,),	 '_blocks.35._bn2.weight',	(304,),
 'block6.8.bottleneck.5.bn.bias',	(384,),	 '_blocks.35._bn2.bias',	(304,),
 'block6.8.bottleneck.5.bn.running_mean',	(384,),	 '_blocks.35._bn2.running_mean',	(304,),
 'block6.8.bottleneck.5.bn.running_var',	(384,),	 '_blocks.35._bn2.running_var',	(304,),
 'block7.0.bottleneck.0.conv.weight',	(2304, 384, 1, 1),	 '_blocks.36._expand_conv.weight',	(1824, 304, 1, 1),
 'block7.0.bottleneck.0.bn.weight',	(2304,),	 '_blocks.36._bn0.weight',	(1824,),
 'block7.0.bottleneck.0.bn.bias',	(2304,),	 '_blocks.36._bn0.bias',	(1824,),
 'block7.0.bottleneck.0.bn.running_mean',	(2304,),	 '_blocks.36._bn0.running_mean',	(1824,),
 'block7.0.bottleneck.0.bn.running_var',	(2304,),	 '_blocks.36._bn0.running_var',	(1824,),
 'block7.0.bottleneck.2.conv.weight',	(2304, 1, 3, 3),	 '_blocks.36._depthwise_conv.weight',	(1824, 1, 3, 3),
 'block7.0.bottleneck.2.bn.weight',	(2304,),	 '_blocks.36._bn1.weight',	(1824,),
 'block7.0.bottleneck.2.bn.bias',	(2304,),	 '_blocks.36._bn1.bias',	(1824,),
 'block7.0.bottleneck.2.bn.running_mean',	(2304,),	 '_blocks.36._bn1.running_mean',	(1824,),
 'block7.0.bottleneck.2.bn.running_var',	(2304,),	 '_blocks.36._bn1.running_var',	(1824,),
 'block7.0.bottleneck.4.squeeze.weight',	(96, 2304, 1, 1),	 '_blocks.36._se_reduce.weight',	(76, 1824, 1, 1),
 'block7.0.bottleneck.4.squeeze.bias',	(96,),	 '_blocks.36._se_reduce.bias',	(76,),
 'block7.0.bottleneck.4.excite.weight',	(2304, 96, 1, 1),	 '_blocks.36._se_expand.weight',	(1824, 76, 1, 1),
 'block7.0.bottleneck.4.excite.bias',	(2304,),	 '_blocks.36._se_expand.bias',	(1824,),
 'block7.0.bottleneck.5.conv.weight',	(640, 2304, 1, 1),	 '_blocks.36._project_conv.weight',	(512, 1824, 1, 1),
 'block7.0.bottleneck.5.bn.weight',	(640,),	 '_blocks.36._bn2.weight',	(512,),
 'block7.0.bottleneck.5.bn.bias',	(640,),	 '_blocks.36._bn2.bias',	(512,),
 'block7.0.bottleneck.5.bn.running_mean',	(640,),	 '_blocks.36._bn2.running_mean',	(512,),
 'block7.0.bottleneck.5.bn.running_var',	(640,),	 '_blocks.36._bn2.running_var',	(512,),
 'block7.1.bottleneck.0.conv.weight',	(3840, 640, 1, 1),	 '_blocks.37._expand_conv.weight',	(3072, 512, 1, 1),
 'block7.1.bottleneck.0.bn.weight',	(3840,),	 '_blocks.37._bn0.weight',	(3072,),
 'block7.1.bottleneck.0.bn.bias',	(3840,),	 '_blocks.37._bn0.bias',	(3072,),
 'block7.1.bottleneck.0.bn.running_mean',	(3840,),	 '_blocks.37._bn0.running_mean',	(3072,),
 'block7.1.bottleneck.0.bn.running_var',	(3840,),	 '_blocks.37._bn0.running_var',	(3072,),
 'block7.1.bottleneck.2.conv.weight',	(3840, 1, 3, 3),	 '_blocks.37._depthwise_conv.weight',	(3072, 1, 3, 3),
 'block7.1.bottleneck.2.bn.weight',	(3840,),	 '_blocks.37._bn1.weight',	(3072,),
 'block7.1.bottleneck.2.bn.bias',	(3840,),	 '_blocks.37._bn1.bias',	(3072,),
 'block7.1.bottleneck.2.bn.running_mean',	(3840,),	 '_blocks.37._bn1.running_mean',	(3072,),
 'block7.1.bottleneck.2.bn.running_var',	(3840,),	 '_blocks.37._bn1.running_var',	(3072,),
 'block7.1.bottleneck.4.squeeze.weight',	(160, 3840, 1, 1),	 '_blocks.37._se_reduce.weight',	(128, 3072, 1, 1),
 'block7.1.bottleneck.4.squeeze.bias',	(160,),	 '_blocks.37._se_reduce.bias',	(128,),
 'block7.1.bottleneck.4.excite.weight',	(3840, 160, 1, 1),	 '_blocks.37._se_expand.weight',	(3072, 128, 1, 1),
 'block7.1.bottleneck.4.excite.bias',	(3840,),	 '_blocks.37._se_expand.bias',	(3072,),
 'block7.1.bottleneck.5.conv.weight',	(640, 3840, 1, 1),	 '_blocks.37._project_conv.weight',	(512, 3072, 1, 1),
 'block7.1.bottleneck.5.bn.weight',	(640,),	 '_blocks.37._bn2.weight',	(512,),
 'block7.1.bottleneck.5.bn.bias',	(640,),	 '_blocks.37._bn2.bias',	(512,),
 'block7.1.bottleneck.5.bn.running_mean',	(640,),	 '_blocks.37._bn2.running_mean',	(512,),
 'block7.1.bottleneck.5.bn.running_var',	(640,),	 '_blocks.37._bn2.running_var',	(512,),
 'block7.2.bottleneck.0.conv.weight',	(3840, 640, 1, 1),	 '_blocks.38._expand_conv.weight',	(3072, 512, 1, 1),
 'block7.2.bottleneck.0.bn.weight',	(3840,),	 '_blocks.38._bn0.weight',	(3072,),
 'block7.2.bottleneck.0.bn.bias',	(3840,),	 '_blocks.38._bn0.bias',	(3072,),
 'block7.2.bottleneck.0.bn.running_mean',	(3840,),	 '_blocks.38._bn0.running_mean',	(3072,),
 'block7.2.bottleneck.0.bn.running_var',	(3840,),	 '_blocks.38._bn0.running_var',	(3072,),
 'block7.2.bottleneck.2.conv.weight',	(3840, 1, 3, 3),	 '_blocks.38._depthwise_conv.weight',	(3072, 1, 3, 3),
 'block7.2.bottleneck.2.bn.weight',	(3840,),	 '_blocks.38._bn1.weight',	(3072,),
 'block7.2.bottleneck.2.bn.bias',	(3840,),	 '_blocks.38._bn1.bias',	(3072,),
 'block7.2.bottleneck.2.bn.running_mean',	(3840,),	 '_blocks.38._bn1.running_mean',	(3072,),
 'block7.2.bottleneck.2.bn.running_var',	(3840,),	 '_blocks.38._bn1.running_var',	(3072,),
 'block7.2.bottleneck.4.squeeze.weight',	(160, 3840, 1, 1),	 '_blocks.38._se_reduce.weight',	(128, 3072, 1, 1),
 'block7.2.bottleneck.4.squeeze.bias',	(160,),	 '_blocks.38._se_reduce.bias',	(128,),
 'block7.2.bottleneck.4.excite.weight',	(3840, 160, 1, 1),	 '_blocks.38._se_expand.weight',	(3072, 128, 1, 1),
 'block7.2.bottleneck.4.excite.bias',	(3840,),	 '_blocks.38._se_expand.bias',	(3072,),
 'block7.2.bottleneck.5.conv.weight',	(640, 3840, 1, 1),	 '_blocks.38._project_conv.weight',	(512, 3072, 1, 1),
 'block7.2.bottleneck.5.bn.weight',	(640,),	 '_blocks.38._bn2.weight',	(512,),
 'block7.2.bottleneck.5.bn.bias',	(640,),	 '_blocks.38._bn2.bias',	(512,),
 'block7.2.bottleneck.5.bn.running_mean',	(640,),	 '_blocks.38._bn2.running_mean',	(512,),
 'block7.2.bottleneck.5.bn.running_var',	(640,),	 '_blocks.38._bn2.running_var',	(512,),
 'last.0.conv.weight',	(2560, 640, 1, 1),	 '_conv_head.weight',	(2048, 512, 1, 1),
 'last.0.bn.weight',	(2560,),	 '_bn1.weight',	(2048,),
 'last.0.bn.bias',	(2560,),	 '_bn1.bias',	(2048,),
 'last.0.bn.running_mean',	(2560,),	 '_bn1.running_mean',	(2048,),
 'last.0.bn.running_var',	(2560,),	 '_bn1.running_var',	(2048,),
 'logit.weight',	(1000, 2560),	 '_fc.weight',	(1000, 2048),
 'logit.bias',	(1000,),	 '_fc.bias',	(1000,),
]

PRETRAIN_FILE = '../efficientnet-b5-586e6cc6.pth'
def load_pretrain(net, skip=[], pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=True):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(CONVERSION).reshape(-1,4)
    for key,_,pretrain_key,_ in conversion:
        if any(s in key for s in
            ['.num_batches_tracked',]+skip):
            continue

        #print('\t\t',key)
        if is_print:
            print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
                key, str(state_dict[key].shape),
                pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
            ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')



### efficientnet #######################################################################
def drop_connect(x, probability, training):
    if not training: return x

    batch_size = len(x)
    keep_probability = 1 - probability
    noise = keep_probability
    noise += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    mask = torch.floor(noise)
    x = x / keep_probability * mask

    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

class Identity(nn.Module):
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x),-1)

class Conv2dBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, zero_pad=[0,0,0,0], group=1):
        super(Conv2dBn, self).__init__()
        if IS_PYTORCH_PAD: zero_pad = [kernel_size//2]*4
        self.pad  = nn.ZeroPad2d(zero_pad)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=stride, groups=group, bias=False)
        self.bn   = BatchNorm2d(out_channel, eps=1e-03, momentum=0.01)
        #print(zero_pad)

    def forward(self, x):
        x = self.pad (x)
        x = self.conv(x)
        x = self.bn  (x)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction_channel, excite_size):
        super(SqueezeExcite, self).__init__()
        self.excite_size=excite_size

        self.squeeze = nn.Conv2d(in_channel, reduction_channel, kernel_size=1, padding=0)
        self.excite  = nn.Conv2d(reduction_channel, in_channel, kernel_size=1, padding=0)
        self.act = Mish()

    # def forward(self, x):
    #     s = F.adaptive_avg_pool2d(x,1)
    #     s = self.act(self.squeeze(s))
    #     s = torch.sigmoid(self.excite(s))
    #
    #     x = s*x
    #     return x


    def forward(self, x):

        if IS_GATHER_EXCITE:
            s = F.avg_pool2d(x, kernel_size=self.excite_size)
        else:
            s = F.adaptive_avg_pool2d(x,1)

        s = self.act(self.squeeze(s))
        s = torch.sigmoid(self.excite(s))

        if IS_GATHER_EXCITE:
            s = F.interpolate(s, size=(x.shape[2],x.shape[3]), mode='nearest')

        x = s*x
        return x


class EfficientBlock(nn.Module):

    def __init__(self, in_channel, channel, out_channel, kernel_size, stride, zero_pad, excite_size, drop_connect_rate):
        super().__init__()
        self.is_shortcut = stride == 1 and in_channel == out_channel
        self.drop_connect_rate = drop_connect_rate

        if in_channel == channel:
            self.bottleneck = nn.Sequential(
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Mish(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.bottleneck = nn.Sequential(
                Conv2dBn(in_channel, channel, kernel_size=1, stride=1),
                Mish(),
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Mish(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1)
            )

    def forward(self, x):
        b = self.bottleneck(x)

        if self.is_shortcut:
            if self.training: b = drop_connect(b, self.drop_connect_rate, True)
            x = b + x
        else:
            x = b
        return x


class EfficientNetB5(nn.Module):

    def __init__(self, drop_connect_rate=0.4):
        super(EfficientNetB5, self).__init__()
        d = drop_connect_rate

        # bottom-top
        self.stem  = nn.Sequential(
            Conv2dBn(3,48, kernel_size=3,stride=2,zero_pad=[0,1,0,1]),
            Mish()
        )

        self.block1 = nn.Sequential(
               EfficientBlock( 48,  48,  24, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=128, drop_connect_rate=d*1/7),
            * [EfficientBlock( 24,  24,  24, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=128, drop_connect_rate=d*1/7) for i in range(1,3)],
        )
        self.block2 = nn.Sequential(
               EfficientBlock( 24, 144,  40, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 64, drop_connect_rate=d*2/7),
            * [EfficientBlock( 40, 240,  40, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 64, drop_connect_rate=d*2/7) for i in range(1,5)],
        )
        self.block3 = nn.Sequential(
               EfficientBlock( 40, 240,  64, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size= 32, drop_connect_rate=d*3/7),
            * [EfficientBlock( 64, 384,  64, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 32, drop_connect_rate=d*3/7) for i in range(1,5)],
        )
        self.block4 = nn.Sequential(
               EfficientBlock( 64, 384, 128, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 16, drop_connect_rate=d*4/7),
            * [EfficientBlock(128, 768, 128, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 16, drop_connect_rate=d*4/7) for i in range(1,7)],
        )
        self.block5 = nn.Sequential(
               EfficientBlock(128, 768, 176, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7),
            * [EfficientBlock(176,1056, 176, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7) for i in range(1,7)],
        )
        self.block6 = nn.Sequential(
               EfficientBlock(176,1056, 304, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size=  8, drop_connect_rate=d*6/7),
            * [EfficientBlock(304,1824, 304, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size=  8, drop_connect_rate=d*6/7) for i in range(1,9)],
        )
        self.block7 = nn.Sequential(
               EfficientBlock(304,1824, 512, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=  8, drop_connect_rate=d*7/7),
            * [EfficientBlock(512,3072, 512, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=  8, drop_connect_rate=d*7/7) for i in range(1,3)],
        )

        self.last = nn.Sequential(
            Conv2dBn(512, 2048,kernel_size=1,stride=1),
            Mish()
        )

        self.logit = nn.Linear(2048,1000)

    def forward(self, x):
        batch_size = len(x)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.last(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)

        return logit

