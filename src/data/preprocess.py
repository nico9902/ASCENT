# impor libraries
import os
import cv2
import numpy as np
import torch
import SimpleITK as sitk
import torchio.transforms as tt
from torchvision.ops import masks_to_boxes
from lungmask import LMInferer

# define the Preprocess class
class Preprocess:
    """
    Prepare data for experiments.
    """
    def __init__(self, datapath, dataoutput, dataset):
        # initialize the class variables
        self.datapath = datapath
        self.dataoutput = dataoutput
        self.dataset = dataset
        self.device = torch.device("mps:0")

    def claro(self):
        """
        Preprocess Claro dataset.
        """
        print('Preprocessing Claro dataset...')
        # iterate through patient directories in the given datapath
        for root, directories, files in os.walk(self.datapath, topdown=True):
            for patient in sorted(directories):
                print("Processing patient:", patient)
                # retrieve a list of studies for the current patient
                studies = [f for f in os.listdir(os.path.join(root, patient)) if not f.startswith(".DS_Store")]
                # check if any studies exist for the patient
                if not studies:
                    print(f"No studies found for patient {patient}. Continuing with the next patient.")
                    continue
                # select the first study for processing
                study = studies[0]
                print("Processing study:", study)
                # retrieve a list of elements for the current patient and study
                elements = [f for f in os.listdir(os.path.join(root, patient, study)) if not f.startswith(".DS_Store")]
                elements = sorted(elements)
                print("Processing elements:", elements)
                # iterate through elements to process non-segmentation data
                for element in elements:
                    print("Processing element:", element)
                    if not "segmentation" in element:
                        dcms_path = os.path.join(root, patient, study, element)
                        # read DICOM image
                        patient_ct = self.read_ct(dcms_path)
                        patient_ct = self.normalize(patient_ct)
                        input_image = sitk.GetArrayFromImage(patient_ct)
                        # extract lungs binary mask
                        patient_seg = self.extract_lungs(patient_ct)
                            
                        # Area filtering
                        top_index, bottom_index = self.filtering(patient_seg, min_area_perc=2.0)
                        # extract lungs ROI 
                        patient_image = self.extract_roi(input_image[top_index:bottom_index, :, :], patient_seg[top_index:bottom_index,:,:])
                        print("Patient image shape:", patient_image.shape)
                        # resize to (256,256) and save slices as numpy array
                        for slice_index in range(len(patient_image)):
                            slice = patient_image[slice_index]
                            slice = slice.astype('float32')
                            slice = cv2.resize(slice, (256, 256))
                            self.save_img(patient, slice, self.dataoutput, slice_index, mode='lungs_roi')
            break
    
    def radiomics(self):
        """
        Preprocess Radiomics dataset.
        """
        print('Preprocessing Radiomics dataset...')

        miss_slice_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-246", "LUNG1-128"]
        # iterate through patient directories in the given datapath
        for root, directories, files in os.walk(self.datapath, topdown=True):
            for patient in sorted(directories):
                if patient not in miss_slice_patients:
                    print("Processing patient:", patient)
                    # retrieve a list of studies for the current patient
                    studies = [f for f in os.listdir(os.path.join(root, patient)) if not f.startswith(".DS_Store")]
                    # check if any studies exist for the patient
                    if not studies:
                        print(f"No studies found for patient {patient}. Continuing with the next patient.")
                        continue
                    # select the first study for processing
                    study = studies[0]
                    print("Processing study:", study)
                    # retrieve a list of elements for the current patient and study
                    for study in studies:
                        elements = [f for f in os.listdir(os.path.join(root, patient, study)) if not f.startswith(".DS_Store")]
                        elements = sorted(elements)
                        print("Processing elements:", elements)           
                        # iterate through elements to process non-segmentation data
                        for element in elements:
                            if not "Segmentation" in element:
                                dcms_path = os.path.join(root, patient, study, element)
                                
                                # skip processing if there is only one DICOM file
                                if len(os.listdir(dcms_path)) == 1:
                                    continue

                                # read DICOM image
                                patient_ct = self.read_ct(dcms_path)
                                patient_ct = self.normalize(patient_ct)
                                input_image = sitk.GetArrayFromImage(patient_ct)
                                
                                # extract lungs binary mask
                                patient_seg = self.extract_lungs(patient_ct)

                                # area filtering
                                top_index, bottom_index = self.filtering(patient_seg, min_area_perc=2.0)

                                # extract ROI 
                                patient_image = self.extract_roi(input_image[top_index:bottom_index, :, :], patient_seg[top_index:bottom_index,:,:])

                                # resize to (256,256) and save slices as numpy array
                                for slice_index in range(len(patient_image)):
                                    slice = patient_image[slice_index]
                                    slice = slice.astype('float32')
                                    slice = cv2.resize(slice, (256, 256))
                                    self.save_img(patient, slice, self.dataoutput, slice_index, mode='lungs_roi')
            break
    

    # save image as numpy array
    def save_img(self, patient, image, dataoutput, index, mode='full_ct_gtv', is_label=False):
        # check if the directory for the patient and mode exists, create it if not
        if not os.path.exists(os.path.join(dataoutput, mode, patient)):
            os.makedirs(os.path.join(dataoutput, mode, patient))
        if is_label:
            # save as numpy array
            np.save(os.path.join(dataoutput, mode, patient, '{:03d}'.format(index) + '_seg.npy'), image)
        else:
            # save as numpy array
            np.save(os.path.join(dataoutput, mode, patient, '{:03d}'.format(index) + '.npy'), image)
    
    # resampling voxel to the desired space
    def normalize(self, image, space=(1,1,3)):
        image = tt.Resample(space, image_interpolation='bspline')(image)
        return image

    # compute coordinates of the roi bounding box 
    def roi_coord(self, mask, roi='lungs'):
        # initialize an empty list to store slice indices containing ROI
        frame_list = []
        # initialize variables to hold the minimum and maximum coordinates of the bounding box
        x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

        # if the ROI is 'lungs', convert any value of 2 in the mask to 1
        if roi == 'lungs':
            mask[mask == 2] = 1

        # loop through each slice of the mask
        for i in range(len(mask)):
            # check if the maximum value in the slice is greater than 0, indicating presence of ROI
            if mask[i].max() > 0:
                # if ROI is present, append the slice index to the frame_list
                frame_list.append(i)

                # extract the current slice
                ct_slice = mask[i]
                ct_slice = ct_slice[None, :, :]  # Add batch dimension

                # convert mask to bounding box coordinates
                bbx = masks_to_boxes(ct_slice)
                bbx = bbx[0].detach().tolist()

                # update the minimum and maximum coordinates of the bounding box
                if bbx[0] < x_min: x_min = int(bbx[0])
                if bbx[1] < y_min: y_min = int(bbx[1])
                if bbx[2] > x_max: x_max = int(bbx[2])
                if bbx[3] > y_max: y_max = int(bbx[3])

        # extract the minimum and maximum slice indices containing ROI
        z_min = frame_list[0]
        z_max = frame_list[-1]
        
        # return the computed bounding box coordinates
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    # crop roi bounding box 
    def crop_coord(self, coord, image, is_label=False):
        # if the image is a label (segmentation mask), crop along all dimensions
        if is_label:
            # crop the image using the bounding box coordinates along all dimensions
            image = image[:, coord[4]:coord[5]+1, coord[2]:coord[3]+1, coord[0]:coord[1]+1]
        else:
            # if the image is not a label, crop only spatial dimensions (x, y)
            # crop the image using the bounding box coordinates for spatial dimensions (x, y)
            image = image[coord[4]:coord[5]+1, coord[2]:coord[3]+1, coord[0]:coord[1]+1]
            # add batch dimension
            # image = image[None,:,:,:]

        # return the cropped image
        return image


    # extract lungs binary mask
    def extract_lungs(self, ct):
        # create an instance of LMInferer (assuming it's a class)
        inferer = LMInferer()
        # apply the inference model to the CT scan to extract lungs mask
        extracted = inferer.apply(ct)  # default model is U-net(R231)
        # clip the values of the extracted mask between 0 and 1
        extracted = np.clip(extracted, 0, 1) 
        # threshold the extracted mask to obtain a binary lungs mask
        lungs_masks = (extracted > 0.5).astype(np.uint8)

        # return the binary lungs mask
        return lungs_masks     

    # extract ROI
    def extract_roi(self, ct, lungs_mask):
        # convert lungs_mask to a PyTorch tensor
        lungs_mask = torch.tensor(lungs_mask)
        # obtain the coordinates of the bounding box for the lungs ROI
        lcr = self.roi_coord(lungs_mask, roi='lungs')  # lcr = lung_roi_coord
        # create a copy of the original CT scan
        lungs_ct = ct.copy()
        # crop the CT scan to extract the lungs ROI using the computed coordinates
        lungs_ct = self.crop_coord(lcr, lungs_ct)

        # return the cropped lungs ROI
        return lungs_ct

    # filtering slices based on area criterion
    def filtering(self, patient_seg, min_area_perc):
        # initialize variables to store the indices of the top and bottom slices
        top_index = None
        bottom_index = None
        
        # check if any non-zero values are present in the patient_seg array
        if np.all(patient_seg == 0) == False:
            # filter slices based on minimum area percentage
            for slice_index in range(len(patient_seg)):
                # compute the area of the slice
                area = cv2.countNonZero(patient_seg[slice_index, :, :])
                # check if the area percentage exceeds the minimum area percentage
                if ((area/(patient_seg[slice_index, :, :].size))*100) > min_area_perc:
                    # if the top_index is not set yet, assign the current slice index to it
                    if top_index == None:
                        top_index = slice_index
                        break
            
            # filter slices based on minimum area percentage
            if top_index is not None:
                bottom_index = None
                # loop through slices in reverse order
                for slice_index in range(len(patient_seg) - 1, top_index - 1, -1):  
                    # compute the area of the slice
                    area = cv2.countNonZero(patient_seg[slice_index, :, :])
                    # check if the area percentage falls below the minimum area percentage
                    if ((area / (patient_seg[slice_index, :, :].size)) * 100) < min_area_perc:
                        # assign the current slice index to bottom_index
                        bottom_index = slice_index
                    if ((area / (patient_seg[slice_index, :, :].size)) * 100) > min_area_perc:
                        if bottom_index is not None:
                            break

        # return the indices of the top and bottom slices
        return top_index, bottom_index
    
    # read tc image
    def read_ct(self, path):
        # create an instance of ImageSeriesReader
        reader = sitk.ImageSeriesReader()
        # get the file names of the DICOM series
        dcm_names = reader.GetGDCMSeriesFileNames(path)
        # set the file names in the reader
        reader.SetFileNames(dcm_names)
        # execute the reader to read the DICOM series and obtain the image
        image = reader.Execute()

        # return the image
        return image

if __name__ == "__main__":
    # define input and output directory
    datapath = '/Users/domenicopaolo/Documents/PhD AI/Datasets/Maastro/manifest-1603198545583/NSCLC-Radiomics'
    dataoutput = 'Lung1'
    dataset = 'NSCLC-Radiomics'

    preprocess = Preprocess(datapath=datapath, dataoutput=dataoutput, dataset=dataset)
    preprocess.radiomics()