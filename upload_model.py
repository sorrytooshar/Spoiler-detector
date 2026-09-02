from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./spoiler_model",
    repo_id="toosharm/spoiler-detector",
    repo_type="model"
)