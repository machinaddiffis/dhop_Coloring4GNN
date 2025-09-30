from utils import *
from feature_aug import *
import os

confInfo = {
'BPP': {
    'name':'BPP',
    'trainDir':r'data/BPP/train',
    'testDir':r'',
    'nGroup':32,
    'reorder':reorderBP,
    'featureAugFuncs':{
        'empty':addEmpty,
        'uniform':addNoiseUniform,
        'pos':addNoisePos,
        'orbit':addBPNoiseOrbit,
        'group':addBPNoiseGroup, #orbit+
        "colorUID":addColorBPPPos,
        'colorGNN':addColorBPPNET,#ColorGNN(emb)
        "colorOrbit":addColorBPPOrbit, #ColorGNN(emb)+orbit
        "colorGroup":addColorBPPGroup, #ColorGNN(emb)+obirt+
        "empty32":addEmpty, #Non-Aug+

    }
},

'SMSP':{
    'name':'SMSP',
    'trainDir':r'data/SMSP/train',
    'testDir':r'',
    'nGroup':32,
    'reorder':reorderSMSP,
    'featureAugFuncs':{
        'empty':addEmpty,
        'uniform':addNoiseUniform,
        'pos':addNoisePos,
        'orbit':addSMSPNoiseOrbit,
        'group':addSMSPNoiseGroup,
        'colorUID':addColorSMSPPos
    }
},


'BIP':{
    'name':'BIP',
    'trainDir':r'data/BIP/train',
    'testDir':r'',
    'nGroup':32,
    'reorder':reorderIP,
    'featureAugFuncs':{
        'empty':addEmpty,
        'uniform':addNoiseUniform,
        'pos':addNoisePos,
        'orbit':addIPNoiseOrbit,
        'group':addIPNoiseGroup,
        'colorUID':addColorIPPos
    }
},


}


