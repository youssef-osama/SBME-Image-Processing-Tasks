from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import uic, QtGui, QtCore
import sys
import pydicom as dicom
import cv2
from PIL import Image
import atexit
import os
import numpy as np
import pyqtgraph as pg
import math
import random
from skimage.draw import disk, rectangle_perimeter
import phantominator
from skimage.transform import radon, iradon
import numpy.matlib

def NNRound(x): #ROUNDS NUMBERS AT 0.6 (INSTEAD OF 0.5)
    if math.modf(x)[0]<0.6:
        x = math.floor(x)
    else:
        x = math.ceil(x)
    return x

def deletor(): #DELETES TEMP IMAGE UPON GRACEFUL EXIT
    os.remove(r'temp.jpg') #only necessary for dcm images, really..
    os.remove('tempgray.jpg')
    os.remove('tempgray2.jpg')
    os.remove('drawtemp.jpg')
atexit.register(deletor)

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        ############################################### LOAD UI ####################################################
        
        uic.loadUi(r"UI.ui", self)
        self.HideDICOMData()
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Warning)
        self.msg.setText("Error")
        self.msg.setInformativeText('Invalid Image File.')
        self.msg.setWindowTitle("Error")
        self.FileOpened = False
        self.isEnlarged = False
        self.msg2 = QMessageBox()
        self.msg2.setIcon(QMessageBox.Warning)
        self.msg2.setText("Error")
        self.msg2.setInformativeText('No Image Loaded.')
        self.msg2.setWindowTitle("Error") 
        self.msg3 = QMessageBox()
        self.msg3.setIcon(QMessageBox.Warning)
        self.msg3.setText("Error")
        self.msg3.setInformativeText('Cannot multiply by a factor of zero.')
        self.msg3.setWindowTitle("Error")       
        self.ImageLabel.setScaledContents(True)
        self.enlarged.setScaledContents(True)
        self.ROIselected = False
        self.OpenBinary()
      

        ############################################## CONNECTIONS #################################################
        
        self.actionOpen.triggered.connect(self.Browse)
        self.enlarger.clicked.connect(self.Enlarge)
        self.saveEnlarged.clicked.connect(self.Saver)
        self.rotateButton.clicked.connect(self.Rotate)
        self.actionT_Image.triggered.connect(self.tImage)
        self.actionBlack_T.triggered.connect(self.tImage2)
        self.sButton.clicked.connect(self.Shear)
        self.gHist.clicked.connect(self.Histogram)
        self.fixHistoButton.clicked.connect(self.Evenly)
        self.UnsharpButton.clicked.connect(self.Unsharp)
        self.AddNoise.clicked.connect(self.Noise)
        self.CleanNoise.clicked.connect(self.DeNoise)
        self.FFTGO.clicked.connect(self.Fourier)
        self.Calculate.clicked.connect(self.SpatialFourier)
        self.actionPhantom.triggered.connect(self.createPhantom)
        self.GaussianNoiseButton.clicked.connect(self.GaussianNoise)
        self.UniformNoiseButton.clicked.connect(self.UniformNoise)
        self.verticalSlider.sliderReleased.connect(self.ROIfunc)
        self.horizontalSlider.sliderReleased.connect(self.ROIfunc)
        self.verticalSlider_2.sliderReleased.connect(self.ROIfunc)
        self.horizontalSlider_2.sliderReleased.connect(self.ROIfunc)
        self.phantomcombo.activated.connect(self.ROIfunc)
        self.GphantomHist.clicked.connect(self.PhantomHistogram)
        self.GphantomHist2.clicked.connect(self.PhantomHistogram2)
        self.ResetROIButton.clicked.connect(self.ResetROI)
        self.actionSLPhantom.triggered.connect(self.createSLPhantom)
        self.SinoButton.clicked.connect(self.Radon)
        self.morphGO.clicked.connect(self.morphOPs)
        self.actionOpenB.triggered.connect(self.BrowseB)
        
        ############################################## SHOW UI #####################################################
        self.show()

    ############################################################################################################
    ############################################## FUNCTIONS ###################################################
    ############################################################################################################

    def HideDICOMData(self): # Hide the DICOM Data labels if the loaded image is not a DICOM image
        self.Pat1.setHidden(True)
        self.Pat2.setHidden(True)
        self.Scan1.setHidden(True)
        self.Mod1.setHidden(True)
        self.AgeLabel.setHidden(True)
        self.NameLabel.setHidden(True)
        self.ModLabel.setHidden(True)
        self.PartLabel.setHidden(True)

    def ShowDICOMData(self): # Show DICOM Data labels if the loaded Image is a DICOM Image
        self.Pat1.show()
        self.Pat2.show()
        self.Scan1.show()
        self.Mod1.show()
        self.NameLabel.show()
        self.AgeLabel.show()
        self.ModLabel.show()
        self.PartLabel.show()

    def Saver(self): # Save the enlarged image from the zoom operation
        if self.isEnlarged == True:
            ffname = QFileDialog.getSaveFileName(self, "Save file", "", self.extCombo.currentText())
            print(ffname[0]+ffname[1])
            cv2.imwrite(ffname[0]+ffname[1], self.zoomed)
        else:
            self.msg2.exec_()

    def tImage(self): # Draw white T on black background
        # Creating a 2D array of zeros, and then setting the values of the array to 255 in the
        # specified locations.
        self.TArray = np.zeros([128, 128], dtype=np.float32)
        self.TArray[29:49, 29:99] = 255
        self.TArray[49:99, 54:74] = 255
        cv2.imwrite('tempT.jpg', self.TArray)
        self.ImageInfo = Image.open('tempT.jpg')
        self.LabelImageT = QtGui.QPixmap('tempT.jpg')
        self.preRotated.setPixmap(self.LabelImageT)

    def tImage2(self): # Draw Black T on white background
        self.TArray = np.zeros([128, 128], dtype=np.float32)
        self.TArray[0:128,0:128] = 255
        self.TArray[29:49, 29:99] = 0
        self.TArray[49:99, 54:74] = 0
        cv2.imwrite('tempT.jpg', self.TArray)
        self.ImageInfo = Image.open('tempT.jpg')
        self.LabelImageT = QtGui.QPixmap('tempT.jpg')
        self.preRotated.setPixmap(self.LabelImageT)
    
    def Threshold(self, image): #Convert an image to binary by setting all values to either 255 or 0 (white or black)
        for i in range (0,image.shape[0]):
            for j in range (0,image.shape[1]):
                if image[i,j]>=128:
                    image[i,j]=255
                elif image[i,j]<128:
                    image[i,j]=0
        return image
    
    def BrowseB(self): # Browse and open an image as binary
        global fname
        fname=QFileDialog.getOpenFileName(None, str("Browse Files"), None, str("Image Files (*.bmp *.jpg *.DCM *.png)"))
        self.BinaryInfo = Image.open(fname[0])
        self.BinaryInfo = self.BinaryInfo.convert('L')
        self.Binarynp = np.array(self.BinaryInfo)
        self.Binarynp = self.Threshold(self.Binarynp) # Convert to binary by applying threshold operation
        self.Draw(self.morphOG, self.Binarynp)
    
    def Erode(self, Array, SElement):
        # Make a new array and pad it
        newarray = np.zeros_like(Array)
        Padded = np.pad(Array, 2)
        rows = Array.shape[0]
        cols = Array.shape[1]
        for x in range(rows):
                for y in range(cols):
                    Paddedtemp = Padded[x:x + 5, y:y + 5] # Give a proper name for the current area
                    logic = np.nonzero(SElement) # Get the white parts from the Structuring Element
                    if np.array_equal(SElement[logic], Paddedtemp[logic]): # Check for fit (white parts in SE and current area are equal)
                        newarray[x,y] = 255 # Set pixel as white if there's a perfect fit 
        return newarray
    
    def Dilate(self, Array, SElement):
        newarray = np.zeros_like(Array)
        Padded = np.pad(Array, 2)
        rows = Array.shape[0]
        cols = Array.shape[1]
        for x in range(rows):
                for y in range(cols):
                    Paddedtemp = Padded[x:x + 5, y:y + 5]
                    logic = np.nonzero(Paddedtemp) # Get white parts from the current area
                    for i in SElement[logic]: 
                        if i == 255: # Check if there's any white in the SE in the parts where it's white in the current area (check for hit)
                            newarray[x,y] = 255 # Set the pixel as white if there's a hit (at least one white has a corresponding white)
        
        return newarray
    
    def morphOPs(self):
        # Perform morophological operation based on the currently selected operation from the drop down list
        if self.morphCombo.currentIndex() == 0:
            self.Draw(self.morphdone, self.Erode(self.Binarynp, self.SElement))
        elif self.morphCombo.currentIndex() == 1:
            self.Draw(self.morphdone, self.Dilate(self.Binarynp, self.SElement))
        elif self.morphCombo.currentIndex() == 2:
            Eroded = self.Erode(self.Binarynp, self.SElement)
            Opened = self.Dilate(Eroded, self.SElement)
            self.Draw(self.morphdone, Opened)
        elif self.morphCombo.currentIndex() == 3:
            Dilated = self.Dilate(self.Binarynp, self.SElement)
            Closed = self.Erode(Dilated, self.SElement)
            self.Draw(self.morphdone, Closed)
    
    def OpenBinary(self):
        # Open the binary image (this function runs on its own)
        self.BinaryInfo = Image.open('binary_image.png')
        self.BinaryInfo = self.BinaryInfo.convert('L')
        self.Binarynp = np.array(self.BinaryInfo)
        self.Draw(self.morphOG, self.Binarynp)
        self.SElement = np.ones([5,5], dtype=np.float32)
        self.SElement[0,0] = 0
        self.SElement[0,4] = 0
        self.SElement[4,0] = 0
        self.SElement[4,4] = 0
        self.SElement*=255  
    
    def createSLPhantom(self):
        global ph
        ph = phantominator.ct_shepp_logan(256, modified=True)
        ph *= 255
        self.Draw(self.SLOGLabel, ph)
    
    def Radon(self):
        # Sinogram 0, 20, 40, 60,...
        sino1 = radon(ph,theta=[0,20,40,60,160])
        lamino1 = iradon(sino1, theta=[0,20,40,60,160])
        sino1 = self.ScalePixels(sino1)
        lamino1 = self.ScalePixels(lamino1)
        self.Draw(self.lamino1, lamino1)
        self.Draw(self.Sino2, sino1)
        # 0-180 theta
        theta1 = np.linspace(0., 180., 256, endpoint=False)
        # Sinogram 0-180:
        sinogram = radon(ph, theta = theta1)
        sinogram = self.ScalePixels(sinogram)  
        lamino = iradon(sinogram, theta=theta1)
        lamino = self.ScalePixels(lamino)
        self.Draw(self.lamino2, lamino)
        self.Draw(self.Sino, sinogram)
        # Laminogram with Ram-Lak filter
        ramlak = iradon(sinogram, theta=theta1, filter_name="ramp")
        ramlak = self.ScalePixels(ramlak)
        self.Draw(self.ramlaklamino, ramlak)
        self.Draw(self.ramlaklabel, sinogram)
        # Hamming
        lamino_4 = iradon(sinogram, theta=theta1, filter_name="hamming")
        lamino_4 = self.ScalePixels(lamino_4)
        self.Draw(self.hamming, sinogram)
        self.Draw(self.hamminglamino, lamino_4)

    
    def createPhantom(self):
        # Draw a simple phantom for noise operations
        self.phantom = np.ones([256,256], dtype=np.float32) # Make a 256x256 of ones
        self.phantom *= 50 # Set the entire array to 50 (50x1)
        self.phantom[33:223,33:223] = 150 # Make a rectangle and set the pixel value to 150
        ri, ci = disk((int(math.ceil(256/2)),int(math.ceil(256/2))),65) # Returns indices for a circle at the centre of the image with r=65
        self.phantom[ri,ci] = 250 # Set the area of the circle to 250
        self.Draw(self.PhantomLabel, self.phantom)
        self.Draw(self.ROILabel, self.phantom)

    def GaussianNoise(self):
        try:
            # This part is self explanatory
            gaussarray = self.phantom
            mean = 0
            sigma = 5
            gauss = np.random.normal(mean,sigma,(256,256))
            gauss = gauss.reshape(256,256)
            self.gnoisy = gaussarray + gauss # Add the array to the noise
            self.gnoisy = self.clipping(self.gnoisy) # Clip the values above 255
            self.Draw(self.GaussianNoiseLabel, self.gnoisy)
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Phantom Not Created!')
            msg.setWindowTitle('Error!')
            msg.exec_()

    def UniformNoise(self):
        try:
            uniarray = self.phantom
            uni = np.zeros_like(self.phantom) # Make a blank array that's the same shape as the phantom
            for i in range(256):
                for j in range(256):
                    uni[i,j]=np.random.uniform(-10,10)
            uni = uni.reshape(256,256)
            self.unoisy = uniarray + uni # Add the two arrays
            self.unoisy = self.clipping(self.unoisy) # Clip the values above 255
            self.Draw(self.UniNoiseLabel, self.unoisy)
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Phantom Not Created!')
            msg.setWindowTitle('Error!')
            msg.exec_()
    
    #Draw ROI Box.
    def ROIfunc(self):
        try:
            # Retrieve values from UI.
            self.vpos1 = 255-self.verticalSlider.value()
            self.hpos1 = self.horizontalSlider.value()
            self.vpos2 = 255-self.verticalSlider_2.value()
            self.hpos2 = self.horizontalSlider_2.value()
            # Draw Rectangle with values from UI
            row, col = rectangle_perimeter(start=(self.vpos1,self.hpos1), end=(self.vpos2,self.hpos2))
            # Make new arrays and draw rectangles on them
            phantom2 = self.phantom.copy() #for ROI tab
            phantom2[row,col] = 0
            phantom3 = self.gnoisy.copy() #for original tab
            phantom3[row,col] = 0
            phantom4 = self.unoisy.copy()
            phantom4[row,col] = 0
            if self.phantomcombo.currentIndex() == 0:
                self.Draw(self.ROILabel, phantom2)
            elif self.phantomcombo.currentIndex() == 1:
                self.Draw(self.ROILabel, phantom3)
            else:
                self.Draw(self.ROILabel, phantom4)
            self.Draw(self.GaussianNoiseLabel, phantom3)
            self.Draw(self.UniNoiseLabel, phantom4)
            self.ROIselected = True
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('No Image Loaded')
            msg.setWindowTitle('Error!')
            msg.exec_()

    #Remove ROI Box.
    def ResetROI(self):
        try:
            if self.phantomcombo.currentIndex() == 0:
                phantom2 = self.phantom.copy()
            elif self.phantomcombo.currentIndex() == 1:
                phantom2 = self.gnoisy.copy()
            else:
                phantom2 = self.unoisy.copy()
            self.Draw(self.ROILabel, phantom2)
            self.Draw(self.GaussianNoiseLabel, self.gnoisy)
            self.Draw(self.UniNoiseLabel, self.unoisy)
            self.ROIselected = False
            self.Resetted = True
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('No Image Loaded')
            msg.setWindowTitle('Error!')
            msg.exec_()

    # Generate Histogram for Gaussian Noised Image.
    def PhantomHistogram(self):
        try:
            self.GaussianNoiseHist.clear()
            self.twodeearray = np.array(self.gnoisy)
            self.histo = np.zeros([256], dtype=np.float32)
            if self.ROIselected == True:
                # Swapping the values of vpos1 and vpos2 if vpos2 is less than vpos1.
                if self.vpos2 < self.vpos1:
                    temp = self.vpos1
                    self.vpos1 = self.vpos2
                    self.vpos2 = temp
                # Swapping the values of hpos1 and hpos2.
                if self.hpos2 < self.hpos1:
                    temp = self.hpos1
                    self.hpos1 = self.hpos2
                    self.hpos2 = temp
                for i in range(self.hpos1,self.hpos2):
                    for j in range(self.vpos1,self.vpos2):
                        val = math.floor(self.twodeearray[i,j])
                        self.histo[val] += 1
            else:
                for i in range(0,255):
                    for j in range(0,255):
                        val = math.floor(self.twodeearray[i,j])
                        self.histo[val] += 1
            count = 0
            histo2 = np.zeros([256], dtype=np.float32)
            for i in self.histo:
                histo2[count] = i/(256*256)
                count+=1
            self.histoItemPhantom = pg.BarGraphItem(x = np.arange(0,256), height = histo2, width = 0.6, brush ='w')
            self.GaussianNoiseHist.addItem(self.histoItemPhantom)
            self.GaussianNoiseHist.setYRange(0, np.amax(histo2))
            self.GMean.setText(f"Mean: {np.mean(self.histo)}, Sigma: {np.std(self.histo)}")
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('No Image Loaded')
            msg.setWindowTitle('Error!')
            msg.exec_()

    # Generate Histogram for Uniformly Noised Image.
    def PhantomHistogram2(self):
        try:
            self.GaussianNoiseHist2.clear()
            self.twodeearray = np.array(self.unoisy)
            self.histo = np.zeros([256], dtype=np.float32)
            if self.ROIselected == True:
                # Swapping the values of vpos1 and vpos2 if vpos2 is less than vpos1.
                if self.vpos2 < self.vpos1:
                    temp = self.vpos1
                    self.vpos1 = self.vpos2
                    self.vpos2 = temp
                # Swapping the values of hpos1 and hpos2.
                if self.hpos2 < self.hpos1:
                    temp = self.hpos1
                    self.hpos1 = self.hpos2
                    self.hpos2 = temp
                for i in range(self.hpos1,self.hpos2):
                    for j in range(self.vpos1,self.vpos2):
                        val = math.floor(self.twodeearray[i,j])
                        self.histo[val] += 1
            else:
                for i in range(0,255):
                    for j in range(0,255):
                        val = math.floor(self.twodeearray[i,j])
                        self.histo[val] += 1
            count = 0
            histo2 = np.zeros([256], dtype=np.float32)
            for i in self.histo:
                histo2[count] = i/(256*256)
                count+=1
            self.histoItemPhantom = pg.BarGraphItem(x = np.arange(0,256), height = histo2, width = 0.6, brush ='w')
            self.GaussianNoiseHist2.addItem(self.histoItemPhantom)
            self.GaussianNoiseHist2.setYRange(0, np.amax(histo2))
            self.UMean.setText(f"Mean: {np.mean(self.histo)}, Sigma: {np.std(self.histo)}")
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('No Image Loaded')
            msg.setWindowTitle('Error!')
            msg.exec_()
        
    def SpatialFourier(self):
        # try : 
            # kernal gets bigger bzawad el filtering 
            self.kernel_width = int(self.kernelfft.text())
            self.kernel_height = int(self.kernelfft.text())
            self.greyed = np.array(self.greyed)
            img_height = self.greyed.shape[0]
            img_width = self.greyed.shape[1]
            # Pad if even to avoid shifting
            if(img_height % 2 == 0):
                self.greyed = np.insert(self.greyed, img_height, self.greyed[img_height-1,:], axis=0)
            if(img_width % 2 == 0):
                self.greyed = np.insert(self.greyed, img_width, self.greyed[:,img_width-1], axis=1)
            # Pad around kernel
            new_kernel = np.zeros(self.greyed.shape)
            KernelHeight = (img_height - self.KernelHeighteight+1)//2
            KernelWidth = (img_width - self.KernelWidthidth+1)//2
            # Multiply Arrays
            for i in range(KernelHeight, KernelHeight + self.KernelHeighteight):
                for j in range(KernelWidth, KernelWidth + self.KernelWidthidth):
                    new_kernel[i, j] = 1/(self.KernelHeighteight * self.KernelWidthidth)
            # Inverse operations to be able to see the image
            Fouirer_image = np.fft.fftshift(np.fft.fft2(self.greyed))
            Fourier_kernel = np.fft.fftshift(np.fft.fft2(new_kernel))
            Multiply_FK = np.zeros(self.greyed.shape)
            Multiply_FK = Fourier_kernel * Fouirer_image
            FourierFilterImage = np.fft.ifft2(np.fft.ifftshift(Multiply_FK))
            FourierFilterImage = np.abs(np.fft.fftshift(FourierFilterImage))
            # Fourier Filtered Image
            FourierFilterImage = self.ScalePixels(FourierFilterImage)
            self.Draw(self.FilteredImage, FourierFilterImage)
            # Smoothed Image
            self.Unsharp() # Perform the Unsharp operation the self.temp output is a global variable that we will use here
            self.temp = self.ScalePixels(self.temp)
            self.Draw(self.Smoothed, self.temp)
            # Subtract the images (the ranges are there to eliminate size errors that happen after padding)
            subtracted_image =self.temp[0:FourierFilterImage.shape[0]-1,0:FourierFilterImage.shape[1]-1] - FourierFilterImage[0:FourierFilterImage.shape[0]-1,0:FourierFilterImage.shape[1]-1]
            subtracted_image = self.clipping(subtracted_image) # Clip values below 0 and above 255
            self.Draw(self.SubtractedImage, subtracted_image)

    def Fourier(self):
        try :
            self.greyed2=np.array(self.greyed)
            dark_image_grey_fourier = np.fft.fft2(self.greyed2)
            dark_image_grey_fourier2 =np.fft.fftshift(dark_image_grey_fourier)
            real = dark_image_grey_fourier2.real
            imaginary = dark_image_grey_fourier2.imag
            ############ magnitude before log ############ 
            magnitude = np.sqrt((real**2) + (imaginary**2))
            magnitude = self.ScalePixels(magnitude)
            self.Draw(self.mag, magnitude)
            ########## phase before log #############
            phase = np.arctan2(imaginary,real)
            phase = self.ScalePixels(phase)
            self.Draw(self.phase, phase)
            ###########  magnitude after log ###########
            shifted_magnitude = np.float32(magnitude)
            c = 255 / np.log(1 + np.max(shifted_magnitude))
            shifted_magnitude = c * np.log(1 + shifted_magnitude)
            shifted_magnitude = np.asarray(shifted_magnitude)
            scaled_shifted_magnitude = (np.double(shifted_magnitude) - np.double(np.min(shifted_magnitude)))/ (np.double(np.max(shifted_magnitude)) - np.double(np.min(shifted_magnitude))) * np.double(np.max(shifted_magnitude))
            scaled_shifted_magnitude = self.ScalePixels(scaled_shifted_magnitude)
            self.Draw(self.logmag,scaled_shifted_magnitude)
            ########### phase after log ###########
            shifted_phase = np.log(phase+2*math.pi)
            scaled_shifted_phase = (np.double(shifted_phase) - np.double(np.min(shifted_phase)))/ (np.double(np.max(shifted_phase)) - np.double(np.min(shifted_phase))) * np.double(np.max(shifted_phase))
            scaled_shifted_phase = self.ScalePixels(scaled_shifted_phase)
            self.Draw(self.logphase, scaled_shifted_phase)
        except:    
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('CHECK YOUR ERROR')
            msg.setWindowTitle('ERROR')
            msg.exec_()
    
    def Draw(self, target, array): #'target' is a pyqt5 label object, 'array' is the array that needs to be drawn in that label
        cv2.imwrite('drawtemp.jpg', array)
        drawtemp = QtGui.QPixmap('drawtemp.jpg')
        target.setPixmap(drawtemp)
    
    def Noise(self):
        # Add Salt & Pepper noise to an image
        try:
            twodeearray = np.array(self.greyed)
            rows = twodeearray.shape[0]
            cols = twodeearray.shape[1]
            self.noisedarray = twodeearray
            NoiseLottery = random.randint(300, 10000)
            print(NoiseLottery)
            # Creating a random number of white pixels in the image.
            for i in range(NoiseLottery):
                Y = random.randint(1, cols -1)
                X = random.randint(1, rows -1)
                self.noisedarray[X,Y] = 255
            NoiseLottery = random.randint(300, 10000)
            print(NoiseLottery)
            # Creating a random number of black pixels in the image.
            for i in range(NoiseLottery):
                Y = random.randint(1, cols -1)
                X = random.randint(1, rows -1)
                self.noisedarray[X,Y] = 0
            self.Draw(self.Noised, self.noisedarray)
        except:
            self.msg2.exec_()

    def DeNoise(self):
        try:
            twodeearray = np.array(self.greyed)
            rows = twodeearray.shape[0]
            cols = twodeearray.shape[1]
            denoisedarray = self.noisedarray
            temp = []
            indexer = 3 // 2
            data_final = []
            data_final = np.zeros_like(self.noisedarray)
            # Creating a 3x3 matrix of the pixel values around the pixel being processed.
            for i in range(rows):
                for j in range(cols):
                    for z in range(3):
                        if i + z - indexer < 0 or i + z - indexer > len(self.noisedarray) - 1:
                            for c in range(3):
                                temp.append(0)
                        else:
                            if j + z - indexer < 0 or j + indexer > len(self.noisedarray[0]) - 1:
                                temp.append(0)
                            else:
                                for k in range(3):
                                    temp.append(self.noisedarray[i + z - indexer][j + k - indexer])

                    temp.sort() # Sort the data to be able to get the median
                    data_final[i][j] = temp[len(temp) // 2] # Get the median value
                    temp = [] # Clear the temp array to be able to append new values to it
            denoisedarray = data_final
            self.Draw(self.Cleaned, denoisedarray)
        except:
            self.msg2.exec_()
    
    def Unsharp(self):
        try:
            # Creating a 2D array from the greyed image.
            twodeearray = np.array(self.greyed)
            rows = twodeearray.shape[0]
            cols = twodeearray.shape[1]
            self.image = np.array(self.greyed)
            # Retrieve data from UI
            self.factor =int (self.Factorx.text())
            self.kernsize =int (self.KernelSpin.text())
            self.kernel = np.ones((self.kernsize, self.kernsize), dtype=np.float32) / (self.kernsize * self.kernsize)
            self.img_data = np.asarray(self.image)
            self.newcols = cols + (self.kernsize - 1)
            self.newrows = rows + (self.kernsize - 1)

            # Creating a new image with the same dimensions as the original image.
            paddedImageArray = np.zeros((self.newrows,self.newcols),dtype = np.float32)
            self.temp = np.zeros((self.newrows,self.newcols),dtype = np.float32)

            # Padding the image with zeros.
            for i in range (rows):
                for j in range (cols):
                    if i+((self.kernsize)//2) <= rows and j+((self.kernsize)//2) <= cols:
                        paddedImageArray[i+((self.kernsize)//2),j+((self.kernsize)//2)] = self.image[i,j]
                        self.temp[i+((self.kernsize)//2),j+((self.kernsize)//2)] = self.image[i,j]

            # Applying the kernel to the image.
            for x in range(rows):
                for y in range(cols):
                    self.temp[x,y] = np.sum((self.kernel * paddedImageArray[x:x + self.kernsize, y:y + self.kernsize]))

            mask = paddedImageArray - self.temp
            mask = mask * self.factor
            self.final = paddedImageArray + mask
            self.final = self.ScalePixels(self.final)
            self.Draw(self.AfterMask, self.final)
        except:
            self.msg2.exec_()
    
    def ScalePixels(self,img): # Scale the pixel values of an array to be 0-255 and return that array
        imgmin = img - img.min()
        imgscale = 255*(imgmin / imgmin.max())
        return imgscale
    
    def clipping(self,image): # Set values > 255 to = 255, set values < 0 to = 0
        for i in range (0,image.shape[0]):
            for j in range (0,image.shape[1]):
                if image[i,j]>255:
                    image[i,j]=255
                elif image[i,j]<0:
                    image[i,j]=0
        return image

    def Evenly(self):
        self.newHist.clear()
        ############################# MAKE CDF ###########################
        cdf=[]
        total=0
        for i in histo2:
            total=total+i
            cdf.append(total)
        ##################### EQUALIZE ###################################
        sk = np.arange(256)
        for i in range(256):
            sk[i] = round((255*cdf[i]))
        New_img = np.zeros_like(self.twodeearray)
        for i in range(0, self.twodeearray.shape[0]):
            for j in range(0, self.twodeearray.shape[1]):
                New_img[i, j] = sk[self.twodeearray[i, j]]
        ##################### MAKE NEW HISTOGRAM ################
        self.histo3 = np.zeros([256], dtype=np.float32)
        for i in range(0, self.twodeearray.shape[0]):
            for j in range(0, self.twodeearray.shape[1]):
                self.histo3[New_img[i,j]] += 1
        count = 0
        histo4 = np.zeros([256], dtype=np.float32)
        # Normalize
        for i in self.histo3:
            histo4[count] = i/(self.twodeearray.shape[0]*self.twodeearray.shape[1])
            count+=1
        ######################################## UI STUFF ###################################################
        hsnp = np.array(New_img)
        self.Draw(self.newHistImage, hsnp)
        self.histoItem2 = pg.BarGraphItem(x = np.arange(0,256), height = histo4, width = 0.6, brush ='w')
        self.newHist.addItem(self.histoItem2)
        self.newHist.setYRange(0, np.amax(histo4))

    def Histogram(self):
        self.oldHist.clear()
        self.twodeearray = np.array(self.greyed)
        rows = self.twodeearray.shape[0]
        cols = self.twodeearray.shape[1]
        global histo
        global histo2
        self.histo = np.zeros([256], dtype=np.float32)
        # Counting the number of times each value appears in the array.
        for i in range(0, rows):
            for j in range(0, cols):
                self.histo[self.twodeearray[i,j]] += 1
        count = 0
        histo2 = np.zeros([256], dtype=np.float32)
        # Normalize
        for i in self.histo:
            histo2[count] = i/(rows*cols)
            count+=1
        self.histoItem = pg.BarGraphItem(x = np.arange(0,256), height = histo2, width = 0.6, brush ='w')
        self.oldHist.addItem(self.histoItem)
        self.oldHist.setYRange(0, np.amax(histo2))

    def Rotate(self):
        try:
            global rotationdone
            rotationdone = False
            spinned = int(self.angleSpin.text())
            theta = np.radians(int(spinned))
            greyed = self.ImageInfo.convert('L')
            rotationMatrix = np.array(
            [[np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
            twodeearray = np.array(greyed)
            rows = twodeearray.shape[0]
            cols = twodeearray.shape[1]
            self.rotatedarray = np.zeros([math.ceil(rows), math.ceil(cols)], dtype=np.float32)
            #pixelwise rotation by substitution
            for i in range(rows):
                rotationdone = False
                for j in range(cols):
                    newij = np.matmul(rotationMatrix, [i-int(rows/2),j-int(cols/2)])
                    if round(newij[0])+int(rows/2) < rows and round(newij[1])+int(cols/2) < cols and round(newij[1])+int(rows/2) >0 and round(newij[0])+int(rows/2)>0:
                        self.rotatedarray[i,j] = twodeearray[round(newij[0])+int(rows/2), round(newij[1])+int(cols/2)]
                    else:
                        self.rotatedarray[i,j]=0
                    rotationdone = True
            if self.rMethodCombo.currentIndex() == 0:
                if rotationdone:
                    self.Draw(self.postRotated, self.rotatedarray)
                else:
                    print("error in rotation")
            else:
                self.rotatedbi = np.zeros([int(rows), int(cols)],dtype=np.float32)

                Xscale = 1
                Yscale = 1

                for i in range(int(rows)):
                    for j in range(int(cols)):
                        x1 = math.floor(Xscale * j) #GET VALUES OF CLOSEST FOUR POINTS
                        y1 = math.floor(Yscale * i)
                        x2 = math.ceil(Xscale * j)
                        y2 = math.ceil(Yscale * i)
                        x_weight = (Xscale * j) - x1 #GET VALUES OF WEIGHTS BY SUBTRACTING LENGTH FROM CURRENT POSITION
                        y_weight = (Yscale * i) - y1
                        a = self.rotatedarray[y1, x1] #ASSIGN AFOREMENTIONED VALUES TO PLACEHOLDER VARIABLES
                        b = self.rotatedarray[y1, x2]
                        c = self.rotatedarray[y2, x1]
                        d = self.rotatedarray[y2, x2]
                        ######################################### BILINEAR FORMULA #####################################################
                        pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight
                        self.rotatedbi[i][j] = pixel
                self.Draw(self.postRotated, self.rotatedarray)
        except:
            self.msg2.exec_()

    def Shear(self):
        try:
            global rotationdone
            rotationdone = False
            spinned = int(self.angleSpin.text())
            theta = np.radians(int(spinned))
            greyed = self.ImageInfo.convert('L')
            shearingMatrix = np.array(
            [[1, 0],
            [theta, 1]])
            twodeearray = np.array(greyed)
            rows = twodeearray.shape[0]
            cols = twodeearray.shape[1]
            self.rotatedarray = np.zeros([math.ceil(rows), math.ceil(cols)], dtype=np.float32)
            #pixelwise rotation by substitution
            for i in range(rows):
                rotationdone = False
                for j in range(cols):
                    newij = np.matmul(shearingMatrix, [i-int(rows/2),j-int(cols/2)])
                    if round(newij[0])+int(rows/2) < rows and round(newij[1])+int(cols/2) < cols and round(newij[1])+int(rows/2) >0 and round(newij[0])+int(rows/2)>0:
                        self.rotatedarray[i,j] = twodeearray[round(newij[0])+int(rows/2), round(newij[1])+int(cols/2)]
                    else:
                        self.rotatedarray[i,j]=0
                    rotationdone = True
            if self.rMethodCombo.currentIndex() == 0:
                if rotationdone:
                    self.Draw(self.postRotated, self.rotatedarray)
                else:
                    print("error in rotation")
            else:
                self.rotatedbi = np.zeros([int(rows), int(cols)],dtype=np.float32)

                Xscale = 1
                Yscale = 1

                for i in range(int(rows)):
                    for j in range(int(cols)):
                        x1 = math.floor(Xscale * j) #GET VALUES OF CLOSEST FOUR POINTS
                        y1 = math.floor(Yscale * i)
                        x2 = math.ceil(Xscale * j)
                        y2 = math.ceil(Yscale * i)
                        x_weight = (Xscale * j) - x1 #GET VALUES OF WEIGHTS BY SUBTRACTING LENGTH FROM CURRENT POSITION
                        y_weight = (Yscale * i) - y1
                        a = self.rotatedarray[y1, x1] #ASSIGN AFOREMENTIONED VALUES TO PLACEHOLDER VARIABLES
                        b = self.rotatedarray[y1, x2]
                        c = self.rotatedarray[y2, x1]
                        d = self.rotatedarray[y2, x2]
                        ######################################### BILINEAR FORMULA #####################################################
                        pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight
                        self.rotatedbi[i][j] = pixel
                self.Draw(self.postRotated, self.rotatedarray)
        except:
            self.msg2.exec_()
    
    def Enlarge(self):
        isOkay = True
        if self.FileOpened == True:
            self.ImageInfoGray1 = self.ImageInfo.convert('L')
            self.ImageInfoGray = np.array(self.ImageInfoGray1)
            self.Factor = float(self.FactorCombo.text())
            if self.Factor < 0 : # Make negative values positive
                absolute = abs(self.Factor)
                print(absolute)
                self.Factor = (1/absolute)
            elif self.Factor == 0: #Refuse a factor of zero
                isOkay = False # Set initiation variable to false
                self.msg3.exec_()
            if isOkay == True: # If inititation variable is True (no zero factor)
                if self.type.currentText() == 'Nearest Neighbour':
                    self.NearestNeighbour()
                else:
                    self.Bilinear()
                self.isEnlarged = True
        else:
            self.msg2.exec_()

    def NearestNeighbour(self):
        rows=self.Factor*self.ImageInfoGray.shape[0]
        columns=self.Factor*self.ImageInfoGray.shape[1]
        xscale = rows / (self.ImageInfoGray.shape[0]-1) #SAME AS FACTOR BUT ELIMINATES ERROR
        yscale = columns / (self.ImageInfoGray.shape[1]-1)
        self.zoomed=np.zeros((int(rows),int(columns)),dtype=np.float32) #CREATE NEW ARRAY OF ZEROES
        ############################## NEAREST NEIGHBOUR LOGIC ###################################
        for i in range(0,int(rows)):
            for j in range(0,int(columns)):
                #USE NEW ROUNDING FUNCTION
                self.zoomed[i,j]=self.ImageInfoGray[NNRound(i/xscale),NNRound(j/yscale)]
        ######################### UI ##############################
        cv2.imwrite('tempgray2.jpg', self.zoomed)
        self.ImageInfoGray2 = Image.open('tempgray2.jpg')
        self.LabelImageGray = QtGui.QPixmap('tempgray2.jpg')
        self.newsizelabel.setText(f'Size = {self.ImageInfoGray2.width}x{self.ImageInfoGray2.height}')
        self.enlarged.setPixmap(self.LabelImageGray)
        self.enlarged.resize(self.ImageInfoGray2.width, self.ImageInfoGray2.height)

    def Bilinear(self):
        height = self.Factor*self.ImageInfoGray.shape[0]
        width = self.Factor*self.ImageInfoGray.shape[1]
        img_height, img_width = self.ImageInfoGray.shape[:2]
        self.zoomed = np.zeros([int(height), int(width)],dtype=np.float32)

        Xscale = int(img_width - 1) / (width - 1) 
        Yscale = int(img_height - 1) / (height - 1) 

        for i in range(int(height)):
            for j in range(int(width)):
                x1 = math.floor(Xscale * j) #GET VALUES OF CLOSEST FOUR POINTS
                y1 = math.floor(Yscale * i)
                x2 = math.ceil(Xscale * j)
                y2 = math.ceil(Yscale * i)
                x_weight = (Xscale * j) - x1 #GET VALUES OF WEIGHTS BY SUBTRACTING LENGTH FROM CURRENT POSITION
                y_weight = (Yscale * i) - y1
                a = self.ImageInfoGray[y1, x1] #ASSIGN AFOREMENTIONED VALUES TO PLACEHOLDER VARIABLES
                b = self.ImageInfoGray[y1, x2]
                c = self.ImageInfoGray[y2, x1]
                d = self.ImageInfoGray[y2, x2]
                ######################################### BILINEAR FORMULA #####################################################
                pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight
                self.zoomed[i][j] = pixel
        cv2.imwrite('tempgray2.jpg', self.zoomed)
        self.ImageInfoGray2 = Image.open('tempgray2.jpg')
        self.LabelImageGray = QtGui.QPixmap('tempgray2.jpg')
        self.newsizelabel.setText(f'Size = {self.ImageInfoGray2.width}x{self.ImageInfoGray2.height}')
        self.enlarged.setPixmap(self.LabelImageGray)

    def Browse(self):
        try:
            global fname
            fname=QFileDialog.getOpenFileName(None, str("Browse Files"), None, str("Image Files (*.bmp *.jpg *.DCM *.png)"))
            self.FileOpened = True

            ######################################## DICOM LOGIC ##########################################

            if fname[0].endswith(".DCM") or fname[0].endswith(".dcm") : 
                Ftype = 'DICOM' #SET FILE TYPE STRING
                ds = dicom.dcmread(fname[0]) #PARSE DICOM FILE
                pixelArr = ds.pixel_array #READABLE NAME FOR PIXEL ARRAY
                self.ArrayNumpy = np.array(pixelArr) #CONVERT DCM INTO NUMPY ARRAY TO GET BITDEPTH
                cv2.imwrite('temp.jpg', pixelArr) #CREATE TEMP IMAGE TO SHOW ON UI
                LabelImage = QtGui.QPixmap('temp.jpg') #CONVERT IMAGE TO PIXMAP
                self.ImageInfo = Image.open('temp.jpg') #PARSE THE IMAGE IN PILLOW TO RETRIEVE INFO

                #SET DICOM-SPECIFIC LABELS AND CHECK FOR MISSING DATA
                if hasattr(ds, 'PatientName'):
                    self.NameLabel.setText(str(ds.PatientName))
                else:
                    self.NameLabel.setText("Name Not Defined")
                if hasattr(ds, 'PatientAge'):
                    self.AgeLabel.setText(str(ds.PatientAge))
                else:
                    self.AgeLabel.setText("Age Not Defined")
                if hasattr(ds, 'PatientAge'):
                    self.ModLabel.setText(ds.Modality)
                else:
                    self.ModLabel.setText("Modality Not Defined")
                if hasattr(ds, "get('Slice Location')"):
                    self.PartLabel.setText(str(ds.get('Slice Location')))
                else:
                    self.PartLabel.setText("Not Defined")
                #SHOW DICOM-SPECIFIC LABELS
                self.ShowDICOMData()

            ###################################### PNG/JPEG LOGIC #########################################

            else: 
                self.HideDICOMData() #CALL FUNCTION TO HIDE DICOM-SPECIFIC LABELS
                LabelImage = QtGui.QPixmap(fname[0]) #CONVERT IMAGE TO PIXMAP
                self.ImageInfo = Image.open(fname[0]) #PARSE THE IMAGE IN PILLOW TO RETRIEVE INFO
                self.ArrayNumpy = np.array(self.ImageInfo.getdata()) #CONVERT IMAGE INTO NUMPY ARRAY TO GET BITDEPTH
                Ftype = self.ImageInfo.format #RETRIEVE FILE FORMAT USING PILLOW (STRING)

            ################################# SETTING UNIVERSAL LABEL VARIABLES (K) #######################

            BitDepthUniversal = (self.ArrayNumpy.dtype)
            BDString = str(BitDepthUniversal)
            res = [int(i) for i in BDString if i.isdigit()]
            conc = ""
            for i in res:
                conc = conc + str(i)
            
            #################################### SETTING LABELS RIGHT AFTER BROWSING FOR AN IMAGE #######################################
            # This is needed for the before images of the following operations:
            # 1. Browsing for/Opening and image and interpolation
            # 2. Rotation
            # 3. Unsharp Mask
            # 4. Historgram Equalization
            # 5. Denoising (Median filter)
            # 6. Fourier

            # Make a global variable of the opened image but in black and white 
            self.greyed = self.ImageInfo.convert('L')
            global LI2
            LI2 = self.greyed.toqpixmap() # Convert the image to a pixmap to be able to display on a pyqt label

            # Set Before Image in the following tabs:
            self.ImageLabel.setPixmap(LabelImage) # Browse Tab
            self.oldHistImage.setPixmap(LI2) # Histogram Tab
            self.Noised.setPixmap(LI2) # Median Filter Tab
            self.BeforeMask.setPixmap(LI2) #Unsharp Mask Tab
            self.preRotated.setPixmap(LabelImage) #Rotation Tab
            self.fftOG.setPixmap(LI2) #Fourier Tab

            # Set info label on the browsing tab
            self.Hlabel.setText(str(self.ImageInfo.height)) #SET HEIGHT LABEL
            self.Wlabel.setText(str(self.ImageInfo.width)) #SET WIDTH LABEL
            self.BDlabel.setText(str(self.ImageInfo.bits)) #SET BIT DEPTH LABEL
            self.FTlabel.setText(Ftype + " Image") #SET FILE TYPE LABEL
            self.Mlabel.setText(self.ImageInfo.mode) #SET MODE LABEL
            self.Slabel.setText(str(self.ImageInfo.height*self.ImageInfo.width*self.ImageInfo.bits) + " bytes") #USE OS TO FIND FILE SIZE IN BYTES

            ############################## SET DPI LABEL ######################################

            dpiexists = False #INITIALIZE BOOLEAN VARIABALE TO SKIP DPI VALUE BY DEFAULT

            for i in self.ImageInfo.info: #CHECK FOR DPI VALUE IN self.ImageInfo.info DICT (can't use "hasattr" with dicts)
                if i == 'dpi':
                    dpiexists = True #SET VALUE TO TRUE IF A DPI VALUE IS FOUND
            if dpiexists:
                self.Rlabel.setText((str(self.ImageInfo.info['dpi'][1])+" pixels per inch"))
            else:
                self.Rlabel.setText('Unavailable')
        except:
            self.msg.exec_()

############################################### INITIALIZE APP #############################################

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()