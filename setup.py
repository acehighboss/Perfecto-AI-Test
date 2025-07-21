import nltk
import ssl

def download_nltk_data():
    """
    Downloads the 'punkt' tokenizer models from NLTK.
    Handles SSL certificate verification errors.
    """
    try:
        # SSL 인증서 검증 비활성화 (일부 환경에서 필요)
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    try:
        print("NLTK 'punkt' 모델을 확인하고 있습니다...")
        nltk.data.find('tokenizers/punkt')
        print("'punkt' 모델이 이미 존재합니다.")
    except nltk.downloader.DownloadError:
        print("'punkt' 모델을 찾을 수 없어 다운로드를 시작합니다...")
        nltk.download('punkt')
        print("다운로드가 완료되었습니다.")

if __name__ == "__main__":
    download_nltk_data()