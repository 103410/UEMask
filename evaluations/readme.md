
## ISM & FDFR Evaluation



First, navigate to the `evaluations` directory and install the required dependencies:

```bash
cd deepface
pip install -e .

cd retinaface
pip install -e .
```
Note
To accelerate the evaluation process, please install TensorFlow following the official instructions:
https://www.tensorflow.org/install/pip

Without TensorFlow, the evaluation code will run on CPU only, which is significantly slower.



If you encounter network or proxy issues during automatic weight downloading, please manually download the required model weights:

DeepFace (ArcFace)
https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5

RetinaFace
https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5


## SER-FIQ Evaluation

Navigate to the SER-FIQ directory:
```bash
cd FaceImageQuality
```

Download the pre-trained model files and place them in:

./insightface/model/


Download link:
https://drive.google.com/file/d/17fEWczMzTUDzRTv9qN3hFwVbkqRD7HE7/view?usp=sharing

Install dependencies:
```bash
pip install -r requirements.txt
```

Important: Ensure that numpy==1.22.0 is used for compatibility.

## BRISQUE Evaluation
```bash
pip install brisque
```



