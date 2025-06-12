@echo off
set LOGFILE=setup_deepfake_env.log

echo [START] Logging to %LOGFILE% > %LOGFILE%
echo ------------------------------ >> %LOGFILE%

REM Step 1: Create and activate environment
echo [STEP 1] Creating Conda environment >> %LOGFILE%
call conda create -n deepfake-env8 python=3.9 -y >> %LOGFILE% 2>&1
call conda activate deepfake-env8 >> %LOGFILE% 2>&1

REM Step 2: Install PyTorch stack with CUDA 12.1 (GPU-enabled)
echo [STEP 2] Installing PyTorch, torchvision, torchaudio with CUDA 12.1 >> %LOGFILE%
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y >> %LOGFILE% 2>&1

REM Step 3: Install OpenCV
echo [STEP 3] Installing OpenCV >> %LOGFILE%
call conda install -c conda-forge opencv -y >> %LOGFILE% 2>&1

REM Step 4: Install TensorFlow
echo [STEP 4] Installing TensorFlow >> %LOGFILE%
pip install tensorflow==2.10 >> %LOGFILE% 2>&1

REM Step 5: Install DeepFace (no torch overwrite)
echo [STEP 5] Installing DeepFace and face detection libs >> %LOGFILE%
pip install deepface --no-deps >> %LOGFILE% 2>&1
pip install retina-face keras gdown mtcnn >> %LOGFILE% 2>&1

REM Step 6: Install InsightFace and ONNX
echo [STEP 6] Installing InsightFace + ONNX + image libs >> %LOGFILE%
pip install insightface --no-deps >> %LOGFILE% 2>&1
pip install onnxruntime-gpu --no-deps >> %LOGFILE% 2>&1
pip install onnx scikit-image albumentations >> %LOGFILE% 2>&1

REM Step 7: Install SimSwap dependencies
echo [STEP 7] Installing SimSwap dependencies >> %LOGFILE%
pip install cython moviepy==1.0.3 >> %LOGFILE% 2>&1

REM Step 8: Main packages (safe install without torch overwrite)
echo [STEP 8] Installing core packages >> %LOGFILE%
pip install pandas numpy==1.24.4 scikit-learn matplotlib seaborn==0.12.2 scipy==1.10.1 ffmpeg-python streamlit pyngrok fpdf tqdm >> %LOGFILE% 2>&1

REM Step 9: Safe install of torch-sensitive packages
echo [STEP 9] Installing timm and facenet-pytorch (no deps) >> %LOGFILE%
pip install timm --no-deps >> %LOGFILE% 2>&1
pip install facenet-pytorch --no-deps >> %LOGFILE% 2>&1

REM Step 10: Fix protobuf version
echo [STEP 10] Installing protobuf==3.20.* >> %LOGFILE%
pip install protobuf==3.20.* >> %LOGFILE% 2>&1

REM Step 11: EfficientNet and LipForensics Dependencies 
echo [STEP 11] Installing huggingface_hub and dependencies for EfficientNet and LipForensics >> %LOGFILE%
pip install huggingface_hub safetensors sympy==1.13.1 coloredlogs >> %LOGFILE% 2>&1

REM Step 12: Build mesh_core_cython Cython extension
echo [STEP 12] Building Cython extension for mesh_core_cython >> setup_deepfake_env.log

REM Navigate to the Cython build folder
cd insightface\thirdparty\face3d\mesh\cython || exit /b 1

REM Build the extension
python setup.py build_ext --inplace >> ..\..\..\cython_build.log 2>&1

REM Return to original directory
cd ..\..\..\..\..

echo [COMPLETE] Setup finished. Check %LOGFILE% for details. >> %LOGFILE%
