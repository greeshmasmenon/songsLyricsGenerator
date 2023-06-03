import subprocess
import pprint
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)


def is_gpu_available():
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass
    pprint.pprint(out_dict)
