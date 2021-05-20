# landmark_detection

## Description
This repository contains lib for infer SAN model. **WARNING!!! this lib doesnt work whithout two additional element:**
1. [Minimal lib for create SAN model](https://github.com/PososikTeam/SAN_lib/tree/main/lib)
2. Weights for SAN in state_dict format example you can finde [there](https://drive.google.com/file/d/1rEQuGkAPFnnVscofZDmfQkRXWOcr_HEW/view?usp=sharing)
This repo is closely related with [SAN lib repo](https://github.com/PososikTeam/SAN_lib).

## Install guide
1. first you need to clone repo (git clone https://github.com/PososikTeam/landmark_detection)
2. install landmark_detection (python install setup.py) this command must be ran in cloned repo dir
3. download [minimal lib for create SAN model](https://github.com/PososikTeam/SAN_lib/tree/main/lib), **dir name for this library must be lib**
4. download [weights](https://drive.google.com/file/d/1rEQuGkAPFnnVscofZDmfQkRXWOcr_HEW/view?usp=sharing) or convert from checkpoint by using this [notebook](https://github.com/PososikTeam/SAN_lib/tree/main/create_state_dict)
5. cut weights into minimal lib for create SAN model, weights name file must be san.pth
6. now you can use this lib with following directory structur:
```bash
├── lib
│   ...
│   └── san.pth
├── code.py
```
where code.py is the place there you want to use landmark_detection.

## Quick start
```python
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
```
```answ_dict['landmarks']``` contains x, y coordinates point and it probability, so shape = (num_points, 3),
```answ_dict['error_message']``` contains error message.

Class LandmarkDetector contains **path** as input parameter, this path is weights directory path, so you can add weights file in any place with add this place directory as input for LandmarkDetector class. Also that class have **num_points** parameter- counts of model nedd to detect point.
```python
class LandmarkDetector():
    def __init__(self, num_points = 68, path = None):
```


## Evaluetion example
![Image of Yaktocat](https://github.com/PososikTeam/SAN_lib/blob/main/images/input.png)
![Image of Yaktocat](https://github.com/PososikTeam/SAN_lib/blob/main/images/output.png)


## Testing 
![Image of Yaktocat](https://github.com/PososikTeam/SAN_lib/blob/main/images/tests.png)
