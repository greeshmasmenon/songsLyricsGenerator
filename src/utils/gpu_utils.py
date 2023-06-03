import subprocess
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)


def is_gpu_available():
    try:
        gpu_info: str = subprocess.run('nvidia-smi', check=True, stdout=subprocess.PIPE).stdout
        gpu_info = '\n'.join(gpu_info)
        if gpu_info.find('failed') >= 0:
            logging.info('Not connected to a GPU')
        else:
            logging.info(gpu_info)
    except:
        logging_info("GPU enabled device not found. Looks like it is not a Unix device")
