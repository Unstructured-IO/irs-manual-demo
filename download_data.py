import sys
import urllib
import requests
from bs4 import BeautifulSoup
import re
import zipfile


def get_zip_urls(base="https://www.irs.gov/downloads/irm", start_page=1, max_page=74):
    urls = []
    for page_num in range(start_page, max_page + 1):
        url = f"{base}?page={page_num}"
        response = requests.get(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        for link in soup.find_all("a", href=re.compile(r"\.zip$")):
            urls.append(link.get("href"))
    return urls


def download_and_unzip(urls, unzip_dir):
    for zip_url in urls[:10]:
        filename = zip_url.split("/")[-1]
        urllib.request.urlretrieve(zip_url, filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                # check if the file has a PDF extension
                if file_info.filename.lower().endswith(".pdf"):
                    # extract the file to the PDF directory
                    zip_ref.extract(file_info, unzip_dir)


if __name__ == "__main__":
    base_url = sys.argv[1]
    page_start = int(sys.argv[2])
    page_max = int(sys.argv[3])
    pdf_dir = sys.argv[4]
    print(f"Grabbing zip urls from {base_url}")
    zip_urls = get_zip_urls(base_url, page_start, page_max)
    print(
        f"Found {len(zip_urls)} zip urls, downloading and unzipping pdfs into {pdf_dir}"
    )
    download_and_unzip(zip_urls, pdf_dir)
    print(f"Finished unzipping")
