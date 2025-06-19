import glob
import numpy as np
import cv2 as cv
import pandas as pd


class MLIC():
    def __init__(self, data_path, ld_file, light_dimension=2, src_img_type='jpg', mask=None, test=False):

        self.ld_file = ld_file  # .lp file name containing light directions
        self.data_path = data_path  # path to data (where light directions and images are)
        self.mask = mask
        self.binary_mask = None
        self.test = test  # whether test or train is performing
        # get the global file name for each image in the dataset
        self.filenames = sorted(glob.glob(data_path + '/*.' + src_img_type))
        self.num_samples = len(self.filenames)

        # get height and width
        self.img_resolution = cv.imread(self.filenames[5])
        self.h, self.w, self.channel = self.img_resolution.shape

        if self.mask:
            self.mask_img = cv.imread(data_path + '/mask.png', cv.IMREAD_GRAYSCALE)
            _, self.binary_mask = cv.threshold(self.mask_img, 127, 255, cv.THRESH_BINARY)
            self.samples = np.zeros((len(self.binary_mask[self.binary_mask == 255]), self.num_samples, 3),
                                    dtype=np.float32)
        else:
            self.samples = np.zeros((self.h * self.w, self.num_samples, 3), dtype=np.float32)

        for i, imPath in enumerate(self.filenames):
            img = cv.imread(imPath, -1)

            if self.mask:
                img = img[self.binary_mask == 255]
            else:
                img = np.reshape(img, (self.h * self.w, 3))
            self.samples[:, i, :] = img

        # read light directions from ld_file
        with open(self.ld_file) as f:
            data = f.read()
        data = data.split('\n')
        data = data[1:int(
            data[0]) + 1]  # keep the lines with light directions, remove the first one which is the number of samples

        self.num_lights = len(data)

        # check that number of images and number of light directions match (only for training)
        if not self.test:
            assert self.num_samples == self.num_lights, "Number of train samples and train lights must be equal, check whether you're using the correct type of source images (default is .jpg)"

        # store ligth directions into the ld variable
        self.ld = np.zeros((self.num_lights, light_dimension), np.float32)
        ld_df = pd.DataFrame(columns=['name', 'lx', 'ly', 'lz'])

        for i, dirs in enumerate(data):
            if (len(dirs.split(' ')) == 4):
                sep = ' '
            else:
                sep = '\t'
            # the line could contain the image name in first position, in that case don't take it
            s = dirs.split(sep)
            ld_df.loc[len(ld_df)] = s
            if len(s) == 4:
                self.ld[i] = [float(s[l]) for l in range(1, light_dimension + 1)]

            else:
                self.ld[i] = [float(s[l]) for l in range(light_dimension)]

        # -----------------------------------------------------------------------------------

        # define img_norm to normalize pixel values between 0 and 1
        if cv.imread(self.filenames[0], -1).dtype == 'uint8':
            self.img_norm = 2 ** 8
        else:
            self.img_norm = 2 ** 16

        # -----------------------------------------------------------------------------------

        # normalization

        self.samples /= self.img_norm

    '''
    Function to get samples and light directions

        Parameters
        ----------
        return_lights: bool
            If true, return also light directions array

        Returns
        -------
    '''

    def get_data(self, return_lights=False):

        if return_lights:
            return self.samples / self.img_norm, self.ld

        return self.samples / self.img_norm

    def reshape_samples(self, shape):
        self.samples = np.reshape(self.samples, shape)

    # -----------------------------------------------------------------------------------


if __name__ == '__main__':
    pass
