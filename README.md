# landmark_detection

## Description
This repository contains lib for infer SAN model. **WARNING!!! this lib dosent work whithout two additional element:**
1. [Minimal lib for create SAN model](https://github.com/PososikTeam/SAN_lib/tree/main/lib)
2. Weights for SAN in state_dict format example you can finde [there](https://drive.google.com/file/d/1rEQuGkAPFnnVscofZDmfQkRXWOcr_HEW/view?usp=sharing)

## Install guide
1. first you need to clone repo (git clone https://github.com/PososikTeam/landmark_detection)
2. install lib (python install setup.py) this command must be ran in cloned repo dir
3. download [minimal lib for create SAN model], dir for this library must be lib
4. download [weights](https://drive.google.com/file/d/1rEQuGkAPFnnVscofZDmfQkRXWOcr_HEW/view?usp=sharing) or convert from checkpoint by using this [notebook](https://github.com/PososikTeam/SAN_lib/tree/main/create_state_dict)
5. cut weights into minimal lib for create SAN model, weights name file must be san.pth
6. now you can use this lib with following directory structur:

├── lib
│   .
│   └── san.pth
├── code.py

![Image of Yaktocat](https://github.com/PososikTeam/SAN_lib/blob/main/images/input.png)
![Image of Yaktocat](https://github.com/PososikTeam/SAN_lib/blob/main/images/output.png)


## Testing
![Image of Yaktocat](https://github.com/PososikTeam/SAN_lib/blob/main/images/tests.png)
