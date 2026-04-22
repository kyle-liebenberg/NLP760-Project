import requests
from bs4 import BeautifulSoup
import time

def get_live_article_links(base_url):
    """Finds article links directly from the live Isolezwe homepage."""
    print(f"Fetching live links from {base_url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all hyperlink tags on the homepage
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Filter for links that look like actual news articles
            if '/izindaba/' in href or '/ezemidlalo/' in href: 
                # Ensure it's a full URL
                if href.startswith('/'):
                    href = base_url + href
                if href not in links:
                    links.append(href)
                    
        return links[:5] # Return just the first 5 to test
    except Exception as e:
        print(f"Failed to fetch homepage: {e}")
        return []

def scrape_article(url):
    """Visits a live article URL and extracts the author and full text."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return "Unknown", "Page blocked or missing"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Try to find the author in the meta tags or byline classes
        author = "Unknown"
        # Isolezwe often uses standard author meta tags, or specific classes like 'author-name'
        author_meta = soup.find('meta', attrs={'name': 'author'}) 
        if author_meta:
            author = author_meta.get('content')
            
        # 2. Extract the full text
        paragraphs = soup.find_all('p')
        full_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        return author, full_text
        
    except Exception as e:
        return "Error", str(e)

def main():
    base_url = "https://www.isolezwe.co.za"
    
    # 1. Get current links straight from the homepage
    live_links = get_live_article_links(base_url)
    
    if not live_links:
        print("Could not find any article links. The site structure might have changed.")
        return

    print(f"\n--- Scraping {len(live_links)} Live Articles ---")
    for index, url in enumerate(live_links):
        print(f"\nScraping Article {index + 1}: {url}")
        author, text = scrape_article(url)
        
        print(f"Author: {author}")
        print(f"Text Snippet: {text[:150]}...") 
        time.sleep(2) # Polite scraping delay

if __name__ == "__main__":
    main()