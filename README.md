# posture-detection-backend

Built on top of YOLOv3. Using Python 3.7 > and compiled to Cython.

## Project setup
#### For production
Deploy using Dockerfile.

#### For development

1. Install pip requirements
1. Build for Cython
3. Run and test locally.<br>
   app.py for web server, local.py for reading local directory.

```
pip install -r requirements
python setup.py build_ext --inplace
```

```
python app.py
python local.py
```
