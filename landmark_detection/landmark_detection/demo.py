from landmark_detector import LandmarkDetector
import cv2

def main():
    img = cv2.cvtColor(cv2.imread('img.png'), cv2.COLOR_BGR2RGB)
    box = [200, 10, 580, 500]
    landmark_detector = LandmarkDetector()
    input_dict = {'image' : img, 'box' : box}
    
    answ_dict = landmark_detector.predict(input_dict)
    print(answ_dict['landmarks'])
    print(answ_dict['error_message'])

if '__name__' == 'main':
    main()