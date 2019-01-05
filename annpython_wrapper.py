import cv2
import os, io, time, ctypes, array
import numpy as np
from skimage.transform import resize
from ctypes import cdll, c_char_p
from numpy.ctypeslib import ndpointer


class AnnAPI:
    def __init__(self,library):
        self.lib = ctypes.cdll.LoadLibrary(library)
        self.annQueryInference = self.lib.annQueryInference
        self.annQueryInference.restype = ctypes.c_char_p
        self.annQueryInference.argtypes = []
        self.annCreateInference = self.lib.annCreateInference
        self.annCreateInference.restype = ctypes.c_void_p
        self.annCreateInference.argtypes = [ctypes.c_char_p]
        self.annReleaseInference = self.lib.annReleaseInference
        self.annReleaseInference.restype = ctypes.c_int
        self.annReleaseInference.argtypes = [ctypes.c_void_p]
        self.annCopyToInferenceInput = self.lib.annCopyToInferenceInput
        self.annCopyToInferenceInput.restype = ctypes.c_int
        self.annCopyToInferenceInput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_bool]
        self.annCopyFromInferenceOutput = self.lib.annCopyFromInferenceOutput
        self.annCopyFromInferenceOutput.restype = ctypes.c_int
        self.annCopyFromInferenceOutput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
        self.annCopyFromInferenceOutput_1 = self.lib.annCopyFromInferenceOutput_1
        self.annCopyFromInferenceOutput_1.restype = ctypes.c_int
        self.annCopyFromInferenceOutput_1.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
        self.annCopyFromInferenceOutput_2 = self.lib.annCopyFromInferenceOutput_2
        self.annCopyFromInferenceOutput_2.restype = ctypes.c_int
        self.annCopyFromInferenceOutput_2.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
        self.annRunInference = self.lib.annRunInference
        self.annRunInference.restype = ctypes.c_int
        self.annRunInference.argtypes = [ctypes.c_void_p, ctypes.c_int]
        print('OK: AnnAPI found "' + self.annQueryInference().decode("utf-8") + '" as configuration in ' + library)


class AnnieDetector():
    def __init__(self, annpythonlib, weightsfile):
        self.api = AnnAPI(annpythonlib)
        inp_out_list = self.api.annQueryInference().decode("utf-8").split(';')
        str_count = len(inp_out_list)
        out_list = []
        for i in range(str_count-1):
            if (inp_out_list[i].split(',')[0] == 'input'):
                in_name,ni,ci,hi,wi = inp_out_list[i].split(',')[1:]
            else:
                out_list.append([int(j) for j in inp_out_list[i].split(',')[2:]])

        self.hdl = self.api.annCreateInference(weightsfile.encode('utf-8'))
        self.num_outputs = len(out_list)
        self.outputs = []
        for i in range(self.num_outputs):
            out_buf_shape = out_list[i]
            out_buf_size = out_buf_shape[0]*out_buf_shape[1]*out_buf_shape[2]*out_buf_shape[3]*4
            out_buf = bytearray(out_buf_size)
            self.outputs.append(np.frombuffer(out_buf, dtype=np.float32))
            self.outputs[i] = np.reshape(self.outputs[i], out_buf_shape)
        self.inp_dim = (int(hi),int(wi))
        self.nms_threshold = 0.4
        self.conf_thres = 0.5
        self.num_classes = 80
        self.threshold = 0.18

    def __del__(self):
        self.api.annReleaseInference(self.hdl)


    ### Resize image with unchanged aspect ratio using padding
    def PrepareImage(self, img):
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = self.inp_dim
        new_w = int(min(w, img_w*h/img_h))
        new_h = int(min(img_h*w/img_w, h))
        resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)        
        canvas = np.full((self.inp_dim[1], self.inp_dim[0], 3), 128)
        canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

        return canvas[:,:,::-1].transpose([2,0,1]) / 255.0

    def runInference(self,img):
        #convert image to tensor format (RGB in seperate planes)
        status = self.api.annCopyToInferenceInput(self.hdl, np.ascontiguousarray(img, dtype=np.float32), (img.shape[0]*img.shape[1]*img.shape[2]*4), 0)
        print('INFO: annCopyToInferenceInput status %d'  %(status))
        status = self.api.annRunInference(self.hdl, 1)
        print('INFO: annRunInference status %d ' %(status))
        status = self.api.annCopyFromInferenceOutput(self.hdl, np.ascontiguousarray(self.outputs[0], dtype=np.float32), self.outputs[0].nbytes)
        print('INFO: annCopyFromInferenceOutput status %d for output0' %(status))
        if self.num_outputs > 1:
            status = self.api.annCopyFromInferenceOutput_1(self.hdl, np.ascontiguousarray(self.outputs[1], dtype=np.float32), self.outputs[1].nbytes)
            print('INFO: annCopyFromInferenceOutput_1 status %d for output1' %(status))
        if self.num_outputs > 2:
            self.api.annCopyFromInferenceOutput_2(self.hdl, np.ascontiguousarray(self.outputs[2], dtype=np.float32), self.outputs[2].nbytes)
            print('INFO: annCopyFromInferenceOutput_2 status %d for output2' %(status))

        #print('INFO: annCopyFromInferenceOutput status %d' %(status))
        return self.outputs
    
    ### Transform the logspace offset to linear space coordinates
    ### and rearrange the row-wise output
    def predict_transform(self, prediction, anchors):
        batch_size = prediction.shape[0]
        stride =  self.inp_dim[0] // prediction.shape[2]
        grid_size = self.inp_dim[0] // stride
        bbox_attrs = 5 + self.num_classes
        num_anchors = len(anchors)
        
        prediction = np.reshape(prediction, (batch_size, bbox_attrs*num_anchors, grid_size*grid_size))
        prediction = np.swapaxes(prediction, 1, 2)
        prediction = np.reshape(prediction, (batch_size, grid_size*grid_size*num_anchors, bbox_attrs))
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

        #Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:,:,0] = 1 / (1 + np.exp(-prediction[:,:,0]))
        prediction[:,:,1] = 1 / (1 + np.exp(-prediction[:,:,1]))
        prediction[:,:,4] = 1 / (1 + np.exp(-prediction[:,:,4]))
        
        #Add the center offsets
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = a.reshape(-1,1)
        y_offset = b.reshape(-1,1)

        x_y_offset = np.concatenate((x_offset, y_offset), 1)
        x_y_offset = np.tile(x_y_offset, (1, num_anchors))
        x_y_offset = np.expand_dims(x_y_offset.reshape(-1,2), axis=0)

        prediction[:,:,:2] += x_y_offset

        #log space transform height, width and box corner point x-y
        anchors = np.tile(anchors, (grid_size*grid_size, 1))
        anchors = np.expand_dims(anchors, axis=0)

        prediction[:,:,2:4] = np.exp(prediction[:,:,2:4])*anchors
        prediction[:,:,5: 5 + self.num_classes] = 1 / (1 + np.exp(-prediction[:,:, 5 : 5 + self.num_classes]))
        prediction[:,:,:4] *= stride

        box_corner = np.zeros(prediction.shape)
        box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
        box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
        box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
        box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
        prediction[:,:,:4] = box_corner[:,:,:4]

        return prediction
    
    ### Compute intersection of union score between bounding boxes
    def bbox_iou(self, bbox1, bbox2):
        #Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:,0], bbox1[:,1], bbox1[:,2], bbox1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:,0], bbox2[:,1], bbox2[:,2], bbox2[:,3]
        
        #get the corrdinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        
        #Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=None) \
                     * np.clip(inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=None)

        #Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        
        iou = inter_area / (b1_area + b2_area - inter_area)       
        return iou
    
    def rects_prepare(self, output):
        prediction = None
        # transform prediction coordinates to correspond to pixel location
        for i in range(len(output)):
            # anchor sizes are borrowed from YOLOv3 config file
            if i == 0: 
                anchors = [(116, 90), (156, 198), (373, 326)] 
            elif i == 1:
                anchors = [(30, 61), (62, 45), (59, 119)]
            elif i == 2: 
                anchors = [(10, 13), (16, 30), (33, 23)]
            if prediction is None:
                prediction = self.predict_transform(self.outputs[i], anchors=anchors)
            else:
                prediction = np.concatenate([prediction, self.predict_transform(self.outputs[i], anchors=anchors)], axis=1)

        # confidence thresholding
        conf_mask = np.expand_dims((prediction[:,:,4] > self.conf_thres), axis=2)
        prediction = prediction * conf_mask
        prediction = prediction[np.nonzero(prediction[:, :, 4])]

        # rearrange results
        img_result = np.zeros((prediction.shape[0], 6))
        max_conf_cls = np.argmax(prediction[:, 5:5+self.num_classes], 1)
        #max_conf_score = np.amax(prediction[:, 5:5+num_classes], 1)

        img_result[:, :4] = prediction[:, :4]
        img_result[:, 4] = max_conf_cls
        img_result[:, 5] = prediction[:, 4]     
        #img_result[:, 5] = max_conf_score
        
        # non-maxima suppression
        result = []

        img_result = img_result[img_result[:, 5].argsort()[::-1]] 

        ind = 0
        while ind < img_result.shape[0]:
            bbox_cur = np.expand_dims(img_result[ind], 0)
            ious = self.bbox_iou(bbox_cur, img_result[(ind+1):])
            nms_mask = np.expand_dims(ious < self.nms_threshold, axis=2)
            img_result[(ind+1):] = img_result[(ind+1):] * nms_mask
            img_result = img_result[np.nonzero(img_result[:, 5])]
            ind += 1
        
        for ind in range(img_result.shape[0]):
            pt1 = [int(img_result[ind, 0]), int(img_result[ind, 1])]
            pt2 = [int(img_result[ind, 2]), int(img_result[ind, 3])]
            cls, prob = int(img_result[ind, 4]), img_result[ind, 5]
            result.append((pt1, pt2, cls, prob))

        return result
    
    ### get the mapping from index to classname 
    def get_classname_mapping(self, classfile):
        mapping = dict()
        with open(classfile, 'r') as fin:
            lines = fin.readlines()
            for ind, line in enumerate(lines):
                mapping[ind] = line.strip()
        return mapping

