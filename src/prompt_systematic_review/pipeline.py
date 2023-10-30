from huggingface_hub import HfFileSystem
import pandas as pd
import os




"""
READ THIS
https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions
https://huggingface.co/docs/huggingface_hub/v0.18.0.rc0/guides/hf_file_system

"""




class Pipeline:

    def __init__(self, token=None):
        self.token = token
        self.root = "datasets/PromptSystematicReview/Prompt_Systematic_Review_Dataset/"
        if token is not None:
            self.fs = HfFileSystem(token=token)
        else:
            self.fs = HfFileSystem()

    def is_logged_in(self):
        return self.token is not None
    
    def login(self, token):
        if self.token is not None:
            raise ValueError("Already Logged In")
        else:
            self.fs = HfFileSystem(token=self.token)
            self.token = token

    def get_all_files(self):
        return self.fs.ls(self.root, detail=False)

    def get_all_data_files(self):
        return self.fs.glob(self.root+"**.csv")
    
    def read_from_file(self, fileName):
        return pd.read_csv(self.fs.open(os.path.join(self.root, fileName)))
    
    def write_to_file(self, fileName, dataFrame):
        path = os.path.join(self.root, fileName)
        self.fs.write_text(path, dataFrame.to_csv(index=False))




