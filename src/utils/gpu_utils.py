import os
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)


def gpu_info():
    gpu_info = os.system("nvidia - smi")
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        logging.info('Not connected to a GPU')
    else:
        logging.info(gpu_info)
