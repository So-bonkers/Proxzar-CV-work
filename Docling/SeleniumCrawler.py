# ******************************************************************************************************************
# *Purpose:
#   This python script crawls Crawls a given url using selenium and bs4, built for NIPR site .
#   Also, this python script is used a module and imported into the main script 'CrawlAndExtractJsons.py'.
#
#   Usage:
#       python SeleniumCrawler.py [URL]
#
# *Created By:
#   Sricharan
#
# *Current Version on LIVE:
#  v1.0
#
# *Product Roadmap Implementation:
#
#
# *History:
#  Date         Changed By          Change Description
#  12/07/24                       Initial creation.
#
# ******************************************************************************************************************

#
# import requests
# from bs4 import BeautifulSoup
# import sys
# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.common.exceptions import WebDriverException
# import time
# import json
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from urllib.parse import urljoin, urlparse
# from tenacity import retry, wait_exponential, stop_after_attempt
# import re
# import logging
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.by import By
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# # Retry decorator for network requests
# @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
# def fetchUrl(url):
#     """Fetches a URL using requests library with retry mechanism."""
#     response = requests.get(url)
#     response.raise_for_status()
#     return response
#
# def isValidUrl(url):
#     """Checks if a given URL is valid."""
#     parsedUrl = urlparse(url)
#     return bool(parsedUrl.scheme) and bool(parsedUrl.netloc)
#
# def cleanText(text):
#     """Cleans up text by removing unwanted characters and extra spaces."""
#     text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/tabs with a single space
#     return text.strip()
#
# def SeleniumCrawler(url):
#     """Scrapes content from URLs using Selenium WebDriver."""
#     try:
#         # Fetch the initial URL and parse it with BeautifulSoup
#         reqs = fetchUrl(url)
#         soup = BeautifulSoup(reqs.text, "html.parser")
#         baseUrl = url
#
#         urls = []
#         # Extract all valid URLs from the initial page
#         for link in soup.find_all("a", href=True):
#             href = link.get("href")
#             fullUrl = urljoin(baseUrl, href)
#             if isValidUrl(fullUrl) and not fullUrl.startswith('javascript'):
#                 urls.append(fullUrl)
#
#         # Configure Chrome WebDriver options for headless mode
#         options = Options()
#         options.headless = True
#         driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#         logging.info("Successfully started Chrome web driver.")
#
#         scrapedData = []
#
#         # Iterate over each URL and scrape its content
#         for url in urls:
#             try:
#                 logging.info(f"Accessing URL: '{url}'..")
#                 driver.get(url)
#
#                 # Wait until the page is loaded
#                 WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
#
#                 # Parse the loaded page with BeautifulSoup
#                 soup = BeautifulSoup(driver.page_source, "html.parser")
#                 title = cleanText(soup.title.string) if soup.title else "No title"
#                 categorizedContent = []
#
#                 currentHeader = None
#                 currentParagraphs = []
#
#                 # Define a function to extract text content based on element types
#                 def extractElements(element):
#                     content = ""
#                     if element.name == 'p':
#                         content += element.get_text(strip=True) + " "
#                     elif element.name in ['ul', 'ol']:
#                         for li in element.find_all('li'):
#                             content += "• " + li.get_text(strip=True) + " "
#                     return cleanText(content)
#
#                 # Iterate through relevant HTML elements to categorize content
#                 for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']):
#                     if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
#                         if currentHeader:
#                             categorizedContent.append({
#                                 "header": cleanText(currentHeader),
#                                 "paragraphs": [cleanText(p) for p in currentParagraphs]
#                             })
#                         currentHeader = element.get_text(strip=True)
#                         currentParagraphs = []
#                     elif element.name in ['p', 'ul', 'ol']:
#                         content = extractElements(element)
#                         if content:
#                             currentParagraphs.append(content)
#
#                 # Append categorized content to scrapedData list
#                 if currentHeader:
#                     categorizedContent.append({
#                         "header": cleanText(currentHeader),
#                         "paragraphs": [cleanText(p) for p in currentParagraphs]
#                     })
#
#                 # Prepare data dictionary for each URL
#                 data = {
#                     "url": url,
#                     "title": title,
#                     "content": categorizedContent
#                 }
#                 scrapedData.append(data)
#                 logging.info(f"Scraped data from '{url}'")
#
#             except WebDriverException as e:
#                 logging.error(f"WebDriverException occurred: {e}")
#                 continue
#
#             except Exception as e:
#                 logging.error(f"Exception occurred: {e}")
#                 continue
#
#         # Quit the WebDriver session after scraping all URLs
#         driver.quit()
#
#         # Save scraped data to a JSON file
#         json_file = "scrapedData.json"
#         with open(json_file, "w", encoding="utf-8") as f:
#             json.dump(scrapedData, f, indent=4, ensure_ascii=False)
#         logging.info(f"Saved scraped data to '{json_file}'")
#
#     except requests.RequestException as e:
#         logging.error(f"Requests error occurred: {e}")
#
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#
# if __name__ == "__main__":
#     # If script is run as a standalone module, expect URL as command-line argument
#     if len(sys.argv) < 2:
#         logging.error("Usage: python SeleniumCrawler.py <URL>")
#         sys.exit(1)
#
#     url = sys.argv[1]
#     SeleniumCrawler(url)






import requests
from bs4 import BeautifulSoup
import sys
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException
import time
import json
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from urllib.parse import urljoin, urlparse
from tenacity import retry, wait_exponential, stop_after_attempt
import re
import logging
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retry decorator for network requests
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def fetchUrl(url):
    """Fetches a URL using requests library with retry mechanism."""
    response = requests.get(url)
    response.raise_for_status()
    return response

def isValidUrl(url):
    """Checks if a given URL is valid."""
    parsedUrl = urlparse(url)
    return bool(parsedUrl.scheme) and bool(parsedUrl.netloc)

def cleanText(text):
    """Cleans up text by removing unwanted characters and extra spaces."""
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/tabs with a single space
    return text.strip()

def SeleniumCrawler(url):
    """Scrapes content from URLs using Selenium WebDriver."""
    try:
        # Fetch the initial URL and parse it with BeautifulSoup
        reqs = fetchUrl(url)
        soup = BeautifulSoup(reqs.text, "html.parser")
        baseUrl = url

        # Commented out the section that adds child URLs to the list
        # urls = []
        # for link in soup.find_all("a", href=True):
        #     href = link.get("href")
        #     fullUrl = urljoin(baseUrl, href)
        #     if isValidUrl(fullUrl) and not fullUrl.startswith('javascript'):
        #         urls.append(fullUrl)

        # Configure Chrome WebDriver options for headless mode
        options = Options()
        #options.binary_location = "/opt/google/chrome/chrome"
        options.add_argument("--no-sandbox")
        options.headless = True


        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        logging.info("Successfully started Chrome web driver.")

        scrapedData = []

        # Process only the given seed URL
        try:
            # logging.info(f"Accessing URL: '{url}'..")
            driver.get(url)

            # Wait until the page is loaded
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

            # Parse the loaded page with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")
            title = cleanText(soup.title.string) if soup.title else "No title"
            categorizedContent = []

            currentHeader = None
            currentParagraphs = []

            # Define a function to extract text content based on element types
            def extractElements(element):
                content = ""
                if element.name == 'p':
                    content += element.get_text(strip=True) + " "
                elif element.name in ['ul', 'ol']:
                    for li in element.find_all('li'):
                        content += "• " + li.get_text(strip=True) + " "
                return cleanText(content)

            # Iterate through relevant HTML elements to categorize content
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']):
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if currentHeader:
                        categorizedContent.append({
                            "header": cleanText(currentHeader),
                            "paragraphs": [cleanText(p) for p in currentParagraphs]
                        })
                    currentHeader = element.get_text(strip=True)
                    currentParagraphs = []
                elif element.name in ['p', 'ul', 'ol']:
                    content = extractElements(element)
                    if content:
                        currentParagraphs.append(content)

            # Append categorized content to scrapedData list
            if currentHeader:
                categorizedContent.append({
                    "header": cleanText(currentHeader),
                    "paragraphs": [cleanText(p) for p in currentParagraphs]
                })

            # Prepare data dictionary for the URL
            data = {
                "url": url,
                "title": title,
                "content": categorizedContent
            }
            scrapedData.append(data)
            logging.info(f"Scraped data from '{url}'")

        except WebDriverException as e:
            logging.error(f"WebDriverException occurred: {e}")

        except Exception as e:
            logging.error(f"Exception occurred: {e}")

        # Quit the WebDriver session after scraping the URL
        driver.quit()

        # Save scraped data to a JSON file
        json_file = "scrapedData.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(scrapedData, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved scraped data to '{json_file}'")

    except requests.RequestException as e:
        logging.error(f"Requests error occurred: {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # If script is run as a standalone module, expect URL as command-line argument
    if len(sys.argv) < 2:
        logging.error("Usage: python SeleniumCrawler.py <URL>")
        sys.exit(1)

    url = sys.argv[1]
    SeleniumCrawler(url)
