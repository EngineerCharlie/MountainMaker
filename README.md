# NightSkyify

## Being a good githubber guide

- Work on personal stuff relating to the project in your own folder
- Use the Python Black extension so we have consistent formatting
- import numpy as np, import cv2 as cv2
- Variable naming
  - CamelCase for classes
  - camelCase for functions
  - snake_case for variables
  - ALL_CAPS for constants
  - Use descriptive variable names for code readability


## Setting up virtual environment (to avoid dependency hell)
open cmd/powershell, navigate to directory and run:
`virtualenv --python="C:\Users\Charl\AppData\Local\Programs\Python\Python311\python.exe" python311`  (edit this to your python 3.11 path)
Activate the virtual environment (do this whenever working on stuff for this course in python) ù
`python311\Scripts\activate.ps1` (.bat in cmd)
Then install the required packages:
`pip install -r requirements.txt`

You should be able to run test.py without errors.