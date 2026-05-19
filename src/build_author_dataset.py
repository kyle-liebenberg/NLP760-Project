import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# Dictionary of our target authors and their profile URLs
# I've added 'www.' to ensure the requests route correctly
AUTHOR_PROFILES = {
    "Mhlengi Shangase": "https://www.isolezwe.co.za/authors/journalist/",
    "Zimbili Vilakazi": "https://www.isolezwe.co.za/authors/zee/",
    "Sabelo Nsele": "https://www.isolezwe.co.za/authors/mangethe/",
    "Simangaliso Ntshangase": "https://www.isolezwe.co.za/authors/health-and-lifestyle-journalist/",
    "Fanelesibonge Bengu": "https://www.isolezwe.co.za/authors/fanele-bengu/",
    "Nonkululeko Nhlapo": "https://www.isolezwe.co.za/authors/online/",
    "Nokubongwa Phenyane": "https://www.isolezwe.co.za/authors/maphenyane/",
    "Zakhele Xaba": "https://isolezwe.co.za/authors/zakhele-xaba/",
    "Sibusiso Mdlalose": "https://isolezwe.co.za/authors/mdlalose-wezemidlalo/",
    "Mthokozisi Mncuseni": "https://isolezwe.co.za/authors/your-football-guy/",
    "Charles Khuzwayo": "https://isolezwe.co.za/authors/entertainment/"

    # Dropped Zwelakhe Ngcobo to prevent severe class imbalance
}

def get_article_links(profile_url):
    """Scrapes an author's profile page and returns a list of their article URLs."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    print(f"  -> Visiting profile: {profile_url}")
    
    try:
        response = requests.get(profile_url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"  -> Failed to load profile. Status: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        article_links = []
        
        # Find all links on the profile page
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Heuristic to filter out general nav links. 
            # Isolezwe article links usually contain a date (e.g., 2026-04-22)
            # or are generally long strings ending in a word.
            if ('/izindaba/' in href or '/ezemidlalo/' in href or '/ezokungcebeleka/' in href) and re.search(r'\d{4}-\d{2}-\d{2}', href):
                # Make sure it's a full URL
                if href.startswith('/'):
                    href = "https://www.isolezwe.co.za" + href
                
                # Prevent duplicates and ignore category homepages
                if href not in article_links and len(href) > 40:
                    article_links.append(href)
                    
        return article_links
        
    except Exception as e:
        print(f"  -> Error fetching links: {e}")
        return []

def scrape_article_text(url):
    """Visits an article and extracts the paragraph text."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        
        # Join paragraphs. We ignore very short paragraphs to filter out photo credits/ads
        full_text = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
        return full_text
        
    except Exception as e:
        return None

def main():
    print("Starting Authorship Dataset Builder...")
    
    dataset = [] # List to hold our scraped data
    
    for author, profile_url in AUTHOR_PROFILES.items():
        print(f"\n[Scraping Author: {author}]")
        
        # 1. Get all article links from the profile
        links = get_article_links(profile_url)
        print(f"  -> Found {len(links)} potential articles.")
        
        # 2. Limit to max 50 to keep classes balanced
        links = links[:50] 
        
        # 3. Scrape the text for each link
        successful_scrapes = 0
        for i, link in enumerate(links):
            text = scrape_article_text(link)
            
            if text and len(text) > 150: # Only keep articles with actual content
                dataset.append({
                    'author': author,
                    'url': link,
                    'text': text
                })
                successful_scrapes += 1
                
            # Be polite to the server to avoid getting IP banned
            time.sleep(1.5)
            
            # Print a little progress bar
            if (i + 1) % 5 == 0:
                print(f"     ...scraped {successful_scrapes} texts so far...")
                
        print(f"  -> Finished {author}. Successfully grabbed {successful_scrapes} articles.")

    # 4. Save everything to a CSV
    if dataset:
        print("\nSaving dataset to data/raw/isizulu_authors_dataset.csv...")
        df = pd.DataFrame(dataset)
        
        # Ensure data folder exists!
        df.to_csv('data/raw/isizulu_authors_dataset.csv', index=False, encoding='utf-8')
        
        print("\n--- Summary ---")
        print(df['author'].value_counts())
        print("Done!")
    else:
        print("\nNo data was scraped. Check the logs above.")

if __name__ == "__main__":
    main()