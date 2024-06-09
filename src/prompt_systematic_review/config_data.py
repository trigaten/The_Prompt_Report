import os


DataFolderPath = os.path.abspath("./data")
DotenvPath = os.path.abspath("./.env")
hasDownloadedPapers = False


def setDownloadedPapers(hasDownloadedPapers):
    """
    Set the value of hasDownloadedPapers.

    :param hasDownloadedPapers: The new value for hasDownloadedPapers.
    :type hasDownloadedPapers: bool
    """
    hasDownloadedPapers = hasDownloadedPapers


def setDataFolderPath(p):
    """
    Set the value of DataFolderPath.

    :param p: The new path for DataFolderPath.
    :type p: str
    """
    DataFolderPath = os.path.abspath(p)


def getDataPath():
    """
    Get the absolute path of DataFolderPath.

    :return: The absolute path of DataFolderPath.
    :rtype: str
    """
    return os.path.abspath(DataFolderPath)
