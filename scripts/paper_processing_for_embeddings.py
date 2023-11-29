import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text() is not None])
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)  # Joining the words into a single string



def preprocess_and_read(file_path):
    try:
        document_text = read_pdf(file_path)
        words = preprocess_text(document_text)
        return words, file_path
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], file_path


def preprocess_and_read_sentences(file_path):
    try:
        document_text = read_pdf(file_path)
        if not document_text:
            raise ValueError("No text extracted from PDF")
        
        # Ensure text is a string before tokenization
        if not isinstance(document_text, str):
            print(f"Document text is not a string: {type(document_text)}")
            return [], file_path

        preprocessed_text = preprocess_text(document_text)  # This should return a string
        sentences = sent_tokenize(preprocessed_text)  # Tokenizing into sentences

        # Debug: Check if sentences is a list of strings
        if not all(isinstance(sentence, str) for sentence in sentences):
            raise TypeError("One or more sentences are not string type")

        return sentences, file_path
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], file_path
