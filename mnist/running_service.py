import tempfile
import os
import subprocess

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))

print(MODEL_DIR)

os.environ["MODEL_DIR"] = MODEL_DIR
cmd='nohup tensorflow_model_server \
  --rest_api_port=8502 \
  --model_name=fashion_model \
  --model_base_path="${MODEL_DIR}" >server.log 2>&1'
subprocess.call(cmd,shell=True)