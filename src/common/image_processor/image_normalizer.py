import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from sklearn.utils.extmath import fast_dot
from sklearn.utils import check_array
from math import sqrt


class ImageNormalizer(object):
    def __init__(self):
        pass

    def zca_whitening(self, image, eps):
        """
        N = 1 
        X = image[:,:].reshape((N, -1)).astype(np.float64)

        X = check_array(X, dtype=[np.float64], ensure_2d=True, copy=True)

        # Center data
        self.mean_ = np.mean(X, axis=0)
        print(X.shape)
        X -= self.mean_

        U, S, V = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        zca_matrix = U.dot(np.diag(1.0/np.sqrt(np.diag(S) + 1))).dot(U.T) #ZCA Whitening matrix

        return fast_dot(zca_matrix, X).reshape(image.shape)   #Data whitening
        """
        image = self.local_contrast_normalization(image)
        N = 1
        X = image.reshape((N, -1))

        pca = PCA(whiten=True, svd_solver='full', n_components=X.shape[-1])
        transformed = pca.fit_transform(X)  # return U
        pca.whiten = False
        zca = fast_dot(transformed, pca.components_+eps) + pca.mean_
        # zca = pca.inverse_transform(transformed)
        return zca.reshape(image.shape)


    def local_contrast_normalization(self, image, args=None):
        def __histogram_equalize(img):
            h, w, ch = image.shape
            if ch==3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            img = img.astype(np.uint8)
            cn_channels = tuple(cv2.equalizeHist(d_ch) for d_ch in cv2.split(img))

            if len(cn_channels)==3:
                img = cv2.merge(cn_channels)
                return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
            elif len(cn_channels)==1:
                return cn_channels[0].reshape((h, w, 1))
        return __histogram_equalize(image)


    def global_contrast_normalization(self, image, args=None):
        mean = np.mean(image)
        var = np.var(image)
        return (image-mean)/float(sqrt(var))
