import pandas as pd
from preprocessing import preprocessing
from plotting import plotting
from OCR import OCR
import os
import shutil

data_directory = 'raw'
preprocessed_output_directory = 'preprocessed_output'
recordings_directory = 'Recordings'
output_directory = 'output'

for filename in os.listdir(data_directory):
    file_path = os.path.join(data_directory, filename)
    if os.path.isfile(file_path):
        print(file_path)

    OCR_data_file = OCR(file_path)
    data = preprocessing(OCR_data_file)
    data.to_csv(file_path + '_processed.csv', index=False)
    shutil.move(file_path + '_processed.csv', output_directory)
    shutil.move(file_path, recordings_directory)
    plotting(data, file_path)

