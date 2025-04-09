import tifffile
from cellpose import models, utils
import os

def main(inPath, outPath):
    try:
        tiff_stack = tifffile.imread(inPath)
    except:
        raise FileNotFoundError(f"File not found: {inPath}")

    # Create output directory if it doesn't exist
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # Load pre-trained cellpose model (cyto, nuclei, etc.)
    model = models.Cellpose(gpu=False, model_type='cyto')  # Change model_type as needed

    shape = tiff_stack.shape[0]
    for i in range(1, shape, 50):
        specific_file = tiff_stack[i]

        # Segment the image using cellpose
        masks, flows, styles, diams = model.eval(specific_file, diameter=None, channels=[0, 0])

        # Save the original sliced image and the segmentation result
        slice_out_path = os.path.join(outPath, f'0149_sliced_{i}.tif')
        mask_out_path = os.path.join(outPath, f'0149_mask_{i}.tif')

        # Save both the original slice and the mask
        tifffile.imwrite(slice_out_path, specific_file)
        tifffile.imwrite(mask_out_path, masks.astype('uint16'))  # Use appropriate datatype

        print(f"Saved original slice to {slice_out_path} and mask to {mask_out_path}")

if __name__ == '__main__':
    main('D:/Tissue Engineering/NewFeb4th/data/3D/Default_0149_Mode3D.tif',
         'D:/Tissue Engineering/NewFeb4th/data/3D/sliced_0149_cellpose')
