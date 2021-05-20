import numpy as np
import cv2
import torch
import math

from lib.itn_cpm import create_san

#image_convert_param = {'pre_crop_expand' : 0.2, 'crop_width' : 180, 'crop_height' : 180}
class LandmarkDetector():
    def __init__(self, num_points = 68, path = None):
        if path == None:
            self.model, self.img_conver_param, message = create_san(num_points)
        else:
            self.model, self.img_conver_param, message = create_san(num_points, path)

        self.err_message = ''
        self.num_points = num_points

        if self.model == None:
            self.err_message = message

        self.is_cuda = False
        if self.model != None:
            self.is_cuda = torch.cuda.is_available()
        
        if self.model != None and self.is_cuda:
            self.model.cuda()

        if self.model != None:
            self.model.eval()

        
    def preprocess_image(self, img_np, box):
        
        if isinstance(img_np, np.ndarray):

            #checking box
            if not isinstance(box, list):
                self.err_message = 'box is not list'
                return None, None

            if len(box) != 4:
                self.err_message = 'box len must be = 4'
                return None, None

            for coord in box:
                if not isinstance(coord, float) and not isinstance(coord, int):
                    self.err_message = 'box coordinates contains not float value'
                    return None, None

            for i in range(len(box)):
                try:
                    box[i] = float(box[i])
                except Exception:
                    self.err_message = 'Cant convert ' + str(type(box[i])) + ' in float value'
                    return None, None
            
            #if x1 > x2 or y1 > y2
            if box[0] >= box[2] or box[1] >= box[3]:
                self.err_message = 'x1 > x2 or y1 > y2'
                return None, None

            if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] <0:
                self.err_message = 'negative value in box box[i] < 0'
                return None, None

            if len(img_np.shape) == 3:
                if box[2] > img_np.shape[1] or box[3] > img_np.shape[0]:
                    self.err_message = 'box value out of image size'
                    return None, None
            

            
            
            #checking image
            if len(img_np.shape) != 3 or img_np.shape[2] != 3:
                if len(img_np.shape) != 3:
                    self.err_message = 'input img dim != 3'
                    return None, None
                else:
                    self.err_message = 'channel need be = 3, input_image.shape[2] != 3'
                    return None, None

            if img_np.dtype != 'int32' and img_np.dtype != 'uint8':
                self.err_message = 'image must be int32 or uint8 type'
                return None, None

            if np.max(img_np) > 255 or np.min(img_np) < 0:
                self.err_message = 'image values must be in range [0, 255]'
                return None, None


            def san_github_normalize(img):
                img = img/255.0
                zero_mean_img = np.subtract(img, np.array([0.5, 0.5, 0.5]))
                std_arr = np.array([0.5, 0.5, 0.5])
                output_img = zero_mean_img/std_arr
                return output_img

            crop_ratio = self.img_conver_param['pre_crop_expand']
            reshape_size_x = self.img_conver_param['crop_width']
            reshape_size_y = self.img_conver_param['crop_height']

            face_ex_w, face_ex_h = (box[2] - box[0]) * crop_ratio, (box[3] - box[1]) * crop_ratio

            w, h = img_np.shape[:2]
            x1, y1 = int(max(math.floor(box[0]-face_ex_w), 0)), int(max(math.floor(box[1]-face_ex_h), 0))
            x2, y2 = int(min(math.ceil(box[2]+face_ex_w), w)), int(min(math.ceil(box[3]+face_ex_h), h))
            
            img = img_np[y1:y2, x1:x2,:]
            
            w_resize_coef = img.shape[1] / reshape_size_y
            h_resize_coef = img.shape[0] / reshape_size_x
            
            img = cv2.resize(img, (reshape_size_x, reshape_size_y))
            
            img = san_github_normalize(img)

            input_img = torch.tensor(np.expand_dims(np.transpose(img, (2, 0, 1)), 0), dtype = torch.float32)
            if self.is_cuda:
                input_img = input_img.cuda()

            return input_img, [x1, y1, w_resize_coef, h_resize_coef]
            
        else:
            self.err_message = 'input is not numpy ndarray'
            return None, None

    def predict(self, input_dict, is_point_for_original = True):
        if self.err_message != '':
            return {'landmarks' : [], 'error_message' : self.err_message}
        else:
            
            input_img, convert_original_param = self.preprocess_image(input_dict['image'], input_dict['box'])
            if input_img == None:
                return {'landmarks' : [], 'error_message' : self.err_message}

            points, probs = self.model(input_img)
            answ = np.zeros((self.num_points, 3))

            if is_point_for_original:
                def convert_point_origin_coord(point, parameters):
                    x1, y1, w_resize_coef, h_resize_coef = parameters
                    return [point[0]*w_resize_coef + x1, point[1]*h_resize_coef + y1]

                for i in range(self.num_points):
                    x, y = convert_point_origin_coord(list(points[0][i]), convert_original_param)
                    answ[i] = np.array([float(x), float(y), float(probs[0][i])])

            else:
                for i in range(self.num_points):
                    x, y = list(points[0][i])
                    answ[i] = np.array([float(x), float(y), float(probs[0][i])])

            return {'landmarks' : answ, 'error_message' : self.err_message}



