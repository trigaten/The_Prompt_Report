from datasets import load_dataset, Dataset
import os


def is_authenticated_with_huggingface():
    """
    Check if the user is authenticated with huggingface-cli.
    Returns:
    - bool: True if authenticated, False otherwise.
    """
    token_path = os.path.join(os.path.expanduser("~"), ".huggingface", "token")
    # Check if the token file exists
    if not os.path.exists(token_path):
        return False
    # Check if the token file has content (i.e., the token)
    with open(token_path, 'r') as f:
        token = f.read().strip()
        if not token:
            return False
    
    return True


#make sure
