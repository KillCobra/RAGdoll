import requests
from bs4 import BeautifulSoup

def scrape_google_scholar_pdfs(query, max_results=20):
    query = query.replace(' ', '+')
    base_url = f"https://scholar.google.com/scholar?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
    }
    response = requests.get(base_url, headers=headers)
    
    if response.status_code != 200:
        print("Failed to retrieve content. Status code:", response.status_code)
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_links = []
    for link in soup.find_all('a', href=True):
        if '[PDF]' in link.text or '.pdf' in link['href']:
            pdf_links.append(link['href'])
            if len(pdf_links) >= max_results:
                break
    return pdf_links


# Example usage
if __name__ == "__main__":
    search_query = input("Enter your search query: ")
    pdf_links = scrape_google_scholar_pdfs(search_query)
    if pdf_links:
        print("\nPDF Links Found:")
        for i, link in enumerate(pdf_links, 1):
            print(f"{i}: {link}")
    else:
        print("No PDF links found.")
