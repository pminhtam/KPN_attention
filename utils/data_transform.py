import torchvision.transforms.functional as TF

def random_flip(image,rand_hflip,rand_vflip):
    # Random horizontal flipping
    # print(rand_hflip, rand_vflip)
    if rand_hflip > 0.5:
        # print("rand_hflip")
        image = TF.hflip(image)
    # Random vertical flipping
    if rand_vflip > 0.5:
        # print("rand_vflip")
        image = TF.vflip(image)
    return image