import requests
from bs4 import BeautifulSoup
import time
import random
import urllib.parse

class ScholarScraper:
    def __init__(self):
        # Initialize headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.visited_urls = set()  # Keep track of visited URLs

    def scrape_search_results(self, search_url):
        """
        Scrape the search results page for PDF links
        """
        print(f"Scraping search results from: {search_url}")
        pdf_links = []
        
        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all paper links on the search results page
            for result in soup.find_all('h3', class_='gs_rt'):
                link = result.find('a')
                if link and 'href' in link.attrs:
                    paper_url = link['href']
                    # Find PDF links from the paper's page
                    pdf_links.extend(self.find_pdf_links(paper_url))
                    
            # Add random delay between requests
            time.sleep(random.uniform(1, 3))
                    
        except Exception as e:
            print(f"Error scraping search results: {str(e)}")
            
        return pdf_links

    def find_pdf_links(self, url):
        """
        Crawl a webpage for PDF links
        """
        if not url or url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        pdf_links = []
        
        try:
            print(f"Crawling: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links on the page
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.lower().endswith('.pdf'):
                    pdf_links.append(href)
                    
            # Add random delay between requests
            time.sleep(random.uniform(1, 3))
                    
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            
        return pdf_links

def main():
    scraper = ScholarScraper()
    
    # Get search URL from user
    search_query = input("Enter your search query (e.g., 'firetruck' or 'air'): ")
    search_url = f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={urllib.parse.quote(search_query)}&btnG="
    
    # Scrape PDF links from the search results
    print("\nScraping PDF links...")
    pdf_links = scraper.scrape_search_results(search_url)
    
    # Print found PDF links
    if pdf_links:
        print(f"\nFound {len(pdf_links)} PDF links:")
        for pdf_link in pdf_links:
            print(f"   - {pdf_link}")
    else:
        print("No PDF links found.")

if __name__ == "__main__":
    main() 