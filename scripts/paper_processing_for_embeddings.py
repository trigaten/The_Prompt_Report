import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text() is not None])
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words


def preprocess_and_read(file_path):
    try:
        document_text = read_pdf(file_path)
        words = preprocess_text(document_text)
        return words, file_path
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], file_path
