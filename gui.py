import tkinter as tk
import tkinter.font as tkFont
from PIL import Image
from skimage import data,img_as_float
from skimage.util import invert
import numpy as np
from skimage.measure.pnpoly import grid_points_in_poly
from PIL import Image as im
from PIL import  ImageEnhance
import cv2 as cv2
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt

# Libraries are used only for reading, displaying and for primary purposes of the image.
# The main Processing functions are written without any in-built functions.

class App:
    def __init__(self, root):
        #setting title
        root.title("")
        #setting window size
        width=600
        height=500
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GRadio_989=tk.Radiobutton(root)
        GRadio_989["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_989["font"] = ft
        GRadio_989["fg"] = "#333333"
        GRadio_989["justify"] = "center"
        GRadio_989["text"] = "Region Filling"
        GRadio_989.place(x=340,y=260,width=222,height=62)
        GRadio_989["command"] = self.GRadio_989_command

        GRadio_27=tk.Radiobutton(root)
        GRadio_27["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_27["font"] = ft
        GRadio_27["fg"] = "#333333"
        GRadio_27["justify"] = "center"
        GRadio_27["text"] = "Convex Hull"
        GRadio_27.place(x=340,y=360,width=222,height=62)
        GRadio_27["command"] = self.GRadio_27_command

        GRadio_589=tk.Radiobutton(root)
        GRadio_589["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_589["font"] = ft
        GRadio_589["fg"] = "#333333"
        GRadio_589["justify"] = "center"
        GRadio_589["text"] = "Connected Component Extraction"
        GRadio_589.place(x=70,y=360,width=219,height=63)
        GRadio_589["command"] = self.GRadio_589_command

        GRadio_850=tk.Radiobutton(root)
        GRadio_850["bg"] = "#5fb878"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_850["font"] = ft
        GRadio_850["fg"] = "#333333"
        GRadio_850["justify"] = "center"
        GRadio_850["text"] = "Original Images"
        GRadio_850.place(x=240,y=160,width=144,height=57)
        GRadio_850["command"] = self.GRadio_850_command

        GRadio_602=tk.Radiobutton(root)
        GRadio_602["bg"] = "#ff4500"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_602["font"] = ft
        GRadio_602["fg"] = "#333333"
        GRadio_602["justify"] = "left"
        GRadio_602["text"] = "Boundary Extraction"
        GRadio_602.place(x=70,y=260,width=213,height=63)
        GRadio_602["command"] = self.GRadio_602_command

        GMessage_312=tk.Message(root)
        ft = tkFont.Font(family='Times',size=18)
        GMessage_312["font"] = ft
        GMessage_312["fg"] = "#333333"
        GMessage_312["justify"] = "center"
        GMessage_312["text"] = "Image Processing"
        GMessage_312.place(x=160,y=20,width=300,height=65)

        GMessage_51=tk.Message(root)
        ft = tkFont.Font(family='Times',size=18)
        GMessage_51["font"] = ft
        GMessage_51["fg"] = "#333333"
        GMessage_51["justify"] = "center"
        GMessage_51["text"] = "Quiz 2"
        GMessage_51.place(x=90,y=100,width=431,height=53)

    def GRadio_989_command(self):
        #Reading the image
        img = cv2.imread('Region Filling/regionfilling.png',0)
        kernel=np.ones((3,3))

        #Dilation Function
        def dilation(image,kernel):
            image=image//255
            #Padding with zeros at the boundary
            o_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
            image=np.zeros(image.shape)
            rows=len(o_image)
            cols=len(o_image[0])
            krows=len(kernel)
            kcols=len(kernel[0])
            for i in range(rows-2):
                for j in range(cols-2):
                    counter=0
                    for r in range(krows):
                        for c in range(kcols):
                            if(o_image[i+r][j+c]==1):
                                counter=counter+1
                    if(counter>=1):
                        image[i][j]=1
            return image*255
        def boolean2float(img):
            for _ in range(7):
                img=dilation(img,kernel)   
            return img
        ####Erosion function####
        def erosion(image,kernel):
            image=image//255
            #Padding with zeros at the boundary
            o_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
            image=np.zeros(image.shape)
            rows=len(o_image)
            cols=len(o_image[0])
            krows=len(kernel)
            kcols=len(kernel[0])
            #Convolving the kernel over the image
            for i in range(rows-2):
                for j in range(cols-2):
                    counter=0
                    for r in range(krows):
                        for c in range(kcols):
                            if(o_image[i+r][j+c]==1):
                                counter=counter+1
                    if(counter==krows*kcols):
                        image[i][j]=1
            return image*255

        img = boolean2float(img)
        ###closing function####
        img_noise1=dilation(img,kernel)
        img_noise1=erosion(img_noise1,kernel)


        cv2.imwrite('Region Filling/filling1.jpg',img)
        fill=Image.open("Region Filling/filling1.jpg")
        fill.show()
        collage=Image.open("Region Filling/regionfilling_collage.png")
        collage.show()


    def GRadio_27_command(self):
        from scipy.spatial import ConvexHull
        image=invert(data.horse())
        n =np.ascontiguousarray(image, dtype=np.uint8)
        rows,cols=n.shape
        ndim = image.ndim
        coords = np.ones((2 * (rows + cols), 2), dtype=np.intp)
        coords *= -1
        nonzero = coords
        rows_cols = rows + cols
        rows_2_cols = 2 * rows + cols

        for r in range(rows):
            rows_cols_r = rows_cols + r
            for c in range(cols):
                if n[r, c] != 0:
                    rows_c = rows + c
                    rows_2_cols_c = rows_2_cols + c
                    if nonzero[r, 1] == -1:
                        nonzero[r, 0] = r
                        nonzero[r, 1] = c
                    elif nonzero[rows_cols_r, 1] < c:
                        nonzero[rows_cols_r, 0] = r
                        nonzero[rows_cols_r, 1] = c
                    if nonzero[rows_c, 1] == -1:
                        nonzero[rows_c, 0] = r
                        nonzero[rows_c, 1] = c
                    elif nonzero[rows_2_cols_c, 0] < r:
                        nonzero[rows_2_cols_c, 0] = r
                        nonzero[rows_2_cols_c, 1] = c

        coords =coords[coords[:, 0] != -1]


        from itertools import product
        offsets = np.zeros((2 * image.ndim, image.ndim))
        for vertex, (axis, offset) in enumerate(product(range(image.ndim), (-0.5, 0.5))):
            offsets[vertex, axis] = offset
        coords = (coords[:, np.newaxis, :] + offsets).reshape(-1,ndim)


        def unique_rows(ar):
            ar = np.ascontiguousarray(ar)
            ar_row_view = ar.view('|S%d' % (ar.itemsize * ar.shape[1]))
            _, unique_row_indices = np.unique(ar_row_view, return_index=True)
            ar_out = ar[unique_row_indices]
            return ar_out
        coords = unique_rows(coords)
        hull = ConvexHull(coords)
        vertices = hull.points[hull.vertices]
        mask = grid_points_in_poly(image.shape, vertices)
        mask_data = im.fromarray(mask)
        mask_data.save('Convex Hull/mask1.png')
        mask = im.open("Convex Hull/mask1.png")
        mask.show()
        chull_diff = img_as_float(mask.copy())
        chull_diff[image] = 2
        chull_data = im.fromarray(chull_diff)
        chull_data = chull_data.convert('L')
        chull_im = ImageEnhance.Brightness(chull_data)
        chull_im.enhance(150).save('Convex Hull/final1.png')
        final = im.open("Convex Hull/final1.png")
        final.show()
        collage=Image.open("Convex Hull/convexhull_collage.png")
        collage.show()


    def GRadio_589_command(self):
        n = 12
        l = 256
        np.random.seed(1)
        im = np.zeros((l, l))
        points = l * np.random.random((2, n ** 2))
        im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
        im = filters.gaussian(im, sigma= l / (4. * n))
        blobs = im > 0.7 * im.mean()

        all_labels = measure.label(blobs)
        blobs_labels = measure.label(blobs, background=0)

        plt.figure(figsize=(9, 3.5))
        plt.subplot(131)
        plt.imshow(blobs, cmap='gray')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(all_labels, cmap='nipy_spectral')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(blobs_labels, cmap='nipy_spectral')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    def GRadio_850_command(self):
        org_img = Image.open("image.png")
        org_img.show()


    def GRadio_602_command(self):
        img = cv2.imread('Boundary Extraction/noise.jpg',0)

        kernel=np.ones((3,3))
        ####Erosion function####
        def erosion(image,kernel):
            image=image//255
            #Padding with zeros at the boundary
            o_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
            image=np.zeros(image.shape)
            rows=len(o_image)
            cols=len(o_image[0])
            krows=len(kernel)
            kcols=len(kernel[0])
            #Convolving the kernel over the image
            for i in range(rows-2):
                for j in range(cols-2):
                    counter=0
                    for r in range(krows):
                        for c in range(kcols):
                            if(o_image[i+r][j+c]==1):
                                counter=counter+1
                    if(counter==krows*kcols):
                        image[i][j]=1
            return image*255
        #Dilation Function
        def dilation(image,kernel):
            image=image//255
            #Padding with zeros at the boundary
            o_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
            image=np.zeros(image.shape)
            rows=len(o_image)
            cols=len(o_image[0])
            krows=len(kernel)
            kcols=len(kernel[0])
            for i in range(rows-2):
                for j in range(cols-2):
                    counter=0
                    for r in range(krows):
                        for c in range(kcols):
                            if(o_image[i+r][j+c]==1):
                                counter=counter+1
                    if(counter>=1):
                        image[i][j]=1
            return image*255
        ####Opening Function#####
        img_noise1=erosion(img,kernel)
        img_noise1=dilation(img_noise1,kernel)
        ###closing function####
        img_noise1=dilation(img_noise1,kernel)
        img_noise1=erosion(img_noise1,kernel)


        ####Closing Function#####
        img_noise2=dilation(img,kernel)
        img_noise2=erosion(img_noise2,kernel)
        ###Opening function####
        img_noise2=erosion(img_noise2,kernel)
        img_noise2=dilation(img_noise2,kernel)



        img_bound2=erosion(img_noise2,kernel)
        img_bound2=img_noise2-img_bound2
        cv2.imwrite('Boundary Extraction/boundary1.png',img_bound2)

        boundary = Image.open('Boundary Extraction/boundary1.png')
        boundary.show()
        collage = Image.open('Boundary Extraction/boundary_collage.png')
        collage.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
