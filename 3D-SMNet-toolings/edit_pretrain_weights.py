import math
import torch

weight_path = 'demo_files/pretrained_votenet_on_scannet.tar'
output_path = 'scannet_pretrained_weights_for_mp3d.pkl'

N_classes = 15
N_heading_bins = 1
N_sizes_bins = 15

M = 2+3+N_heading_bins*2+N_sizes_bins*4+N_classes

checkpoint = torch.load(weight_path, map_location='cpu')
model_state = checkpoint['model_state_dict']

print('Original matrix sizes: ', model_state['pnet.conv3.weight'].shape)
print('Original bias sizes: ', model_state['pnet.conv3.bias'].shape)

model_state['pnet.conv3.weight'] = torch.randn((M, 128, 1))
model_state['pnet.conv3.bias'] = torch.randn((M))

print(model_state['pnet.conv3.weight'].max())

torch.nn.init.kaiming_uniform_(model_state['pnet.conv3.weight'], a=math.sqrt(5))

print(model_state['pnet.conv3.weight'].max())

print(model_state['pnet.conv3.bias'].max())
fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(model_state['pnet.conv3.weight'])
bound = 1 / math.sqrt(fan_in)
torch.nn.init.uniform_(model_state['pnet.conv3.bias'], -bound, bound)

print(model_state['pnet.conv3.bias'].max())

print('NEW matrix sizes: ', checkpoint['model_state_dict']['pnet.conv3.weight'].shape)
print('NEW bias sizes: ', checkpoint['model_state_dict']['pnet.conv3.bias'].shape)

torch.save(checkpoint, output_path)
