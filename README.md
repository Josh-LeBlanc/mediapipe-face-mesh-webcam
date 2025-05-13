# usage
I would recommend creating a venv to install the requirements to. python3.12 is the most recent version of python supported by mediapipe at this point. Then install the dependencies.
```bash
git clone https://github.com/josh-le/mediapipe-face-mesh-webcam.git
cd mediapipe-face-mesh-webcam
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Then, just run it:
```bash
python main.py
```
