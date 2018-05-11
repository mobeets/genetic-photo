import numpy as np
from skimage.util.shape import view_as_blocks
from scipy.misc import imread
from PIL import Image, ImageOps

# from keras import backend as K
# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16, preprocess_input

def load_model():
	model = VGG16(weights='imagenet', include_top=False)
	return model

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.
    src: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)
    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

def load_emojis(infile='images/emojis.png', width=8):
	img = imread(infile, mode='RGBA')
	# img = np.array(pure_pil_alpha_to_color_v2(Image.fromarray(img)))
	# img = np.array(Image.open(infile).convert('RGB'))
	# img = np.array(Image.fromarray(img).convert('RGB'))
	B = view_as_blocks(img, block_shape=(32, 32, 4))
	B = B[:,:,0,:,:,:]
	C = np.reshape(B, (B.shape[0]*B.shape[1], 32, 32, 4))
	# for i in range(C.shape[0]):
	# 	img = Image.fromarray(C[i,:,:,:])
	# 	img.save('images/emojis/{}.png'.format(i))
	return [Image.fromarray(c).resize((width,width)) for c in C]

def load_target(infile='images/trump.png'):
	# img = imread(infile, mode='RGBA')
	# img = np.array(pure_pil_alpha_to_color_v2(Image.fromarray(img)))
	img = np.array(Image.open(infile).convert('RGB'))
	# img = np.array(Image.fromarray(img).convert('RGB'))
	return img
	
def fitness0(img_predicted, img_target):
    res = 1.0*np.square(img_predicted - img_target).sum()
    # tot = np.prod(img_target.shape).sum() # maximum possible
    tot = 1.0*np.square(img_target).sum() # maximum possible
    return 1.0 - res/tot

def make_fitness_fcn(model, img_target, layer_name='block5_conv1'):
	model = load_model()
	prep_img = lambda img: preprocess_input(np.expand_dims(img, axis=0))
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	layer_output = layer_dict[layer_name].output
	input_img = model.input
	response = K.function([input_img], [layer_output])

	imgfcn = lambda img: response([prep_img(img.astype('float64'))])[0].flatten()
	# import time
	# s = time.time()
	# img_t = imgfcn(img_target)
	# e = time.time()
	# print('Processed in {} seconds'.format(e-s))
	fitness = lambda yh, y: -np.sqrt(np.square(1.0*y - 1.0*yh).sum())
	# fitness = lambda imgh, img, cx, cy, px, py: -np.square(imgfcn(imgh[cx:cx+px,cy:cy+py]) - imgfcn(img[cx:cx+px,cy:cy+py])).sum()
	return fitness, imgfcn

def main(niters=2400, infile='images/trump.png', outfile='images/out.png', force_add=True, use_pixel_loss=True, use_grid=False, width=8):
	emojis = load_emojis(width=8)#[:20]
	Y = load_target(infile)
	Image.fromarray(Y).save(outfile.replace('.', '_target.'))

	fitness, imgfcn = make_fitness_fcn(Y, layer_name='block5_conv1')
	print('Loaded {} emojis.'.format(len(emojis)))
	print('Target image is {}.'.format(Y.shape))

	Yh = Image.new('RGBA', Y.shape[:2])
	# Yh = Image.new('RGB', Y.shape[:2])
	last_score = 0.0

	# find layer response to each 32x32 emoji
	sx,sy = emojis[0].size
	resps = []
	for i in range(len(emojis)):
		em = pure_pil_alpha_to_color_v2(emojis[i])
		if use_pixel_loss:
			resps.append(np.array(em).flatten())
		else:
			resps.append(imgfcn(np.array(em)))

	if use_grid:
		xs = np.arange(0, Y.shape[0], emojis[0].size[0])
		ys = np.arange(0, Y.shape[1], emojis[0].size[1])
		px, py = np.meshgrid(xs, ys)
		pos = np.vstack([px.flatten(), py.flatten()]).T
	else:
		pos = (np.random.rand(niters, 2)*Y.shape[:2]).astype(int)

	print('Trying {} positions.'.format(len(pos)))
	for i in range(len(pos)):
		# pick random location
		cx,cy = pos[i,:]
		# try each target
		scores = np.zeros(len(emojis))
		# print('Iteration {}'.format(i))

		# find layer response to each 32x32 block of the target
		Yc = Y[cx:cx+sx,cy:cy+sy]
		nx = sx - Yc.shape[0]
		ny = sy - Yc.shape[1]
		if nx > 0 or ny > 0:
			Yc = np.pad(Y[cx:cx+sx,cy:cy+sy],
				[(0,nx), (0,ny), (0,0)],
				mode='mean')#, constant_values=255)
		if use_pixel_loss:
			cur_img_block = Yc.flatten()
		else:
			cur_img_block = imgfcn(Yc)

		# Yhc = np.pad(np.array(Yh)[cx:cx+sx,cy:cy+sy],
		# 	[(0,nx), (0,ny), (0,0)],
		# 	mode='mean')#, constant_values=255)
		# Yhc = Image.fromarray(Yhc)
		for j in range(len(emojis)):
			# Yhcc = Yhc.copy()
			# Yhcc.paste(emojis[j], (0,0), emojis[j])
			# em = pure_pil_alpha_to_color_v2(Yhcc)
			# if use_pixel_loss:
			# 	resps[j] = np.array(em).flatten()
			# else:
			# 	resps[j] = imgfcn(np.array(em))
			scores[j] = fitness(resps[j], cur_img_block)
		null_score = fitness(0.*resps[j], cur_img_block)

		# for j in range(len(emojis)):
		# 	Yhc = Yh.copy()
		# 	Yhc.paste(emojis[j], (cx,cy))#, emojis[j])
		# 	scores[j] = fitness(np.array(Yhc), Y, cx, cy, emojis[0].size[0], emojis[0].size[1])

		j = np.argmax(scores)
		# need to also consider pasting nothing
		if force_add or scores[j] > null_score:
			# Yh.paste(emojis[j], (cx,cy))#, emojis[i])
			Yh.paste(emojis[j], (cx,cy), emojis[j])
			# Yh.rotate(90).transpose(Image.FLIP_LEFT_RIGHT).save(outfile)
			ImageOps.flip(Yh.rotate(90)).save(outfile)
			# Yh.rotate(90).transpose(Image.FLIP_LEFT_RIGHT).save(outfile)
			if i % 20 == 0:
				print(i, j, null_score)
		else:
			if i % 20 == 0:
				print(i, np.nan, null_score)

if __name__ == '__main__':
	main(outfile='logs/outs/trmp2.png')
	# for j in range(100):
	# 	main(outfile='logs/outs/{}.png'.format(j))
