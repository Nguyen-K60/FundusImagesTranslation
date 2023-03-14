def generate_mask(img_size):
            center = [img_size/2-0.5, img_size/2-0.5]
            center = asarray(center)
            print(center)
            mask = np.zeros((img_size, img_size, 3))
            for h in range(img_size):
                for w in range(img_size):
                    point = [h, w]
                    a = center-asarray(point)
                    dist = np.linalg.norm(a)
                    if dist < img_size/2:

                    # if dist < img_size/2-2:
                        mask[h, w, :] = 1
            
            save_img('mask.png', mask)
def masking(src_dir, save_dir, mask):
    if src_dir[-1] != '/' or save_dir[-1] != '/':
        print('dir should end with /')
    with tf.device('/device:GPU:0'):
        files = sorted(os.listdir(src_dir))
        for filename in files:
            # print(filename)
            name, ext = os.path.splitext(filename)
            # if os.path.exists(save_dir+name+'.png') == True:
            #     continue
            image = load_img(src_dir+filename)
            image = img_to_array(image)
            
            save_img(save_dir+name+'.png', mask*image)
        end = time.time()
        print('average masking time: ', (end-begin)/len(files))

mask = load_img('mask.png')
src_dir = '/home/pham/Documents/Python/UFI2OFI/resized-realCFI/'
save_dir = '/home/pham/Documents/Python/UFI2OFI/mask-realCFI/'
generate_mask(256)

masking(src_dir, save_dir, mask)