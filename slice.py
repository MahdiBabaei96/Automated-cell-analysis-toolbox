import tifffile

def main(inPath, outPath):
    try:
        tiff_stack = tifffile.imread(inPath)
    except:
        raise FileNotFoundError

    shape = tiff_stack.shape[0]
    for i in range(1,shape,50):
        specific_file = tiff_stack[i]
        # change file name as your choice
        tempPath = outPath + f'/0149_sliced_{i}.tif'
        tifffile.imwrite(tempPath, specific_file)

if __name__ == '__main__':
    main('D:/Tissue Engineering/NewFeb4th/data/3D/Default_0149_Mode3D.tif', \
         'D:/Tissue Engineering/NewFeb4th/data/3D/sliced_0149')