from huggingface_hub import (
    HfFileSystem,
    login,
    HfApi,
    hf_hub_download,
    snapshot_download,
)
import pandas as pd
from io import StringIO
import os


"""
READ THIS
https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions
https://huggingface.co/docs/huggingface_hub/v0.18.0.rc0/guides/hf_file_system

"""


class Pipeline:
    def __init__(
        self,
        revision="main",
        repo="PromptSystematicReview/Prompt_Systematic_Review_Dataset",
    ):
        try:
            self.token = os.getenv("HF_TOKEN")
            self.fs = HfFileSystem(token=self.token)
            self.api = HfApi(token=self.token)
            self.repo = repo
            self.repo_name = repo.split("/")[1]

        except:
            raise ValueError("Token not found")

        self.root = f"{repo}@{revision}/"
        self.root_without_revision = f"{repo}/"
        self.root_with_type = f"datasets/{self.root}"
        self.root_with_type_without_revision = f"datasets/{self.root_without_revision}"
        self.root_url = f"hf://datasets/{self.root}"
        self.root_url_without_revision = f"hf://datasets/{self.root_without_revision}"

        self.revision = revision

    def get_revision(self):
        return self.revision

    def set_revision(self, revision):
        try:
            assert revision.isalnum()
            self.revision = revision
            self.root = f"{self.repo}@{revision}/"
            self.root_without_revision = f"{self.repo}/"
            self.root_with_type = f"datasets/{self.root}"
            self.root_with_type_without_revision = (
                f"datasets/{self.root_without_revision}"
            )
            self.root_url = f"hf://datasets/{self.root}"
            self.root_url_without_revision = (
                f"hf://datasets/{self.root_without_revision}"
            )
        except:
            raise ValueError("Revision must be alphanumeric")

    def get_root_files(self):
        return self.api.list_repo_files(
            self.repo, revision=self.revision, repo_type="dataset"
        )

    def read_from_file(self, fileName):
        return pd.read_csv(os.path.join(self.root_url, fileName))

    def write_to_file(self, fileName, dataFrame):
        path = os.path.join(self.root_url, fileName)
        print(path)
        dataFrame.to_csv(path, index=False)

    def upload_file(self, filePath, folderInRepo=""):
        fileName = os.path.basename(filePath)
        path = os.path.join(self.repo, fileName)
        self.api.upload_file(
            repo_id=self.repo,
            path_in_repo=fileName,
            path_or_fileobj=fileName,
            commit_message=f"Add {fileName}",
            repo_type="dataset",
            revision=self.revision,
        )

    def download_file(self, fileName, downloadPath="./"):
        if self.root_without_revision[-1] == "/":
            repoId = self.root_without_revision[:-1]
        else:
            repoId = self.root_without_revision
        hf_hub_download(
            repo_id=repoId,
            filename=fileName,
            repo_type="dataset",
            revision=self.revision,
            local_dir=downloadPath,
        )

    def delete_file(self, fileName):
        path = os.path.join(self.repo, fileName)
        self.api.delete_file(
            fileName,
            self.repo,
            token=self.token,
            commit_message=f"Delete {fileName}",
            repo_type="dataset",
            revision=self.revision,
        )

    def upload_folder(self, folderPath, folderInRepo=""):
        if folderInRepo != "":
            folderInRepo = folderPath
        self.api.upload_folder(
            repo_id=self.repo,
            folder_path=folderPath,
            path_in_repo=folderInRepo,
            commit_message=f"Add {folderPath}",
            repo_type="dataset",
            revision=self.revision,
        )

    def delete_folder(self, folderPath):
        self.api.delete_folder(
            folderPath,
            self.repo,
            token=self.token,
            commit_message=f"Delete {folderPath}",
            repo_type="dataset",
            revision=self.revision,
        )

    def download_dataset(self, downloadPath="./"):
        if self.root_without_revision[-1] == "/":
            repoId = self.root_without_revision[:-1]
        else:
            repoId = self.root_without_revision

        snapshot_download(
            repo_id=repoId,
            repo_type="dataset",
            revision=self.revision,
            local_dir=downloadPath,
        )
