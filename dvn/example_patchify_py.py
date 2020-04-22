patch_size = (1, 84, 84, 84)
stride = (60, 60, 60)

# condition: some_integer*stride+patch_size has to be same as input size

patches = patchify(input, patch_size, stride)
patch_shape = patches.shape
patches = Variable(patches.view((-1,) + patch_size).cuda().type(torch.cuda.FloatTensor), requires_grad=False)

output = torch.zeros((0,) + patch_size[1:])

batch_size = 8 #user input
for i in range(np.ceil(1.0 * patches.shape[0] / batch_size)):
	output = torch.cat((output, model(patches[batch_size * i:i * batch_size+batch_size]).data.cpu()), 0) # replace model with your network

new_shape = patch_shape

output = unpatchify(output.view(new_shape), stride, input.shape, 1)