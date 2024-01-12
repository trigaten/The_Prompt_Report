import os


DataFolderPath = os.path.abspath("./data")
DotenvPath = os.path.abspath("./.env")
hasDownloadedPapers = True


def setDownloadedPapers(hasDownloadedPapers):
    hasDownloadedPapers = hasDownloadedPapers


def setDataFolderPath(p):
    DataFolderPath = os.path.abspath(p)


def getDataPath():
    return os.path.abspath(DataFolderPath)


def concatPath(filename):
    return os.path.join(DataFolderPath, filename)
