from huggingface_hub import hf_hub_download
import joblib
import os

if __name__ == '__main__':
    model = joblib.load(
        hf_hub_download(repo_id=os.environ['HF_REPO_ID'], filename=os.environ['MODEL_NAME'])
    )
