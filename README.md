# Mountainifier

## Running the model

The model can be trained with Train.py
The pipeline can be run with App.py
Mountains can be generated on a 1-off basis with ControlledTest.py
When using the model, all variables can be changed in GAN/config.py and model versions and datasets can be selected in GAN/config_local.py. 


## Setting up virtual environment (to avoid dependency hell)
open cmd/powershell, navigate to directory and run:
`virtualenv --python="C:\Users\Charl\AppData\Local\Programs\Python\Python311\python.exe" python311`  (edit this to your python 3.11 path)
Activate the virtual environment (do this whenever working on stuff for this course in python) ù
`python311\Scripts\activate.ps1` (.bat in cmd)
Then install the required packages:
`pip install -r requirements.txt`

You should be able to run test.py without errors.