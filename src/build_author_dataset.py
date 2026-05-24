import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Dictionary of our target authors and their profile URLs
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
}

# Configuration Constants
MAX_ARTICLES_PER_AUTHOR = 80  # Target depth to fit your 50-100 proposal metric
CLICKS_NEEDED = 4              # 1 initial load (25) + 4 clicks (~20-25 per click) = ~100 potential links
OUTPUT_DIR = 'data/raw'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'isizulu_authors_dataset.csv')

def get_article_links_selenium(profile_url, clicks=CLICKS_NEEDED):
    """Visits an author profile page, clicks 'Load More' multiple times to 
    bypass the 25-article rendering cap, and collects all unique article links."""
    print(f"  -> Launching headless browser for: {profile_url}")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    article_links = []
    
    try:
        driver.get(profile_url)
        time.sleep(4)  # Generous wait for initial layout render
        
        for click in range(clicks):
            try:
                # Target the 'Load more' element typical across IOL layouts
                load_more_btn = driver.find_element(By.XPATH, "//*[contains(text(), 'Load more') or contains(text(), 'Load More')]")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_btn)
                time.sleep(1)
                load_more_btn.click()
                print(f"    [+] Clicked 'Load More' button ({click + 1}/{clicks})")
                time.sleep(3)  # Give AJAX call time to pull downstream blocks
            except Exception:
                print("    [-] No further 'Load More' actions clickable or available.")
                break
                
        # Parse the expanded DOM tree
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Explicit category validation and absolute path verification
            if ('/izindaba/' in href or '/ezemidlalo/' in href or '/ezokungcebeleka/' in href) and re.search(r'\d{4}-\d{2}-\d{2}', href):
                if href.startswith('/'):
                    href = "https://www.isolezwe.co.za" + href
                
                if href not in article_links and len(href) > 40:
                    article_links.append(href)
                    
    except Exception as e:
        print(f"    [!] Browser automation error on profile: {e}")
    finally:
        driver.quit()
        
    return article_links

def scrape_article_text(url):
    """Visits a raw article URL and extracts the paragraph texts while stripping
    image descriptors and baseline structural metadata noise."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        
        cleaned_paragraphs = []
        for p in paragraphs:
            p_text = p.get_text().strip()
            
            # Filter out short fragments (like single-word tags, share buttons, photo descriptors)
            if len(p_text) <= 20:
                continue
                
            # Clean off media credits string structures (e.g., 'Image: Name / Independent Newspapers')
            # This protects your CNN-LSTM from building biases around photographer credits
            if p_text.startswith("Image:"):
                p_text = re.sub(r'^Image:.*?([Nn]ewspapers|[Mm]edia|[Ii]solezwe|[Nn]ewspaper)\s*', '', p_text).strip()
                
            if p_text:
                cleaned_paragraphs.append(p_text)
                
        full_text = " ".join(cleaned_paragraphs)
        return full_text if len(full_text) > 150 else None
        
    except Exception:
        return None

def main():
    print("=" * 60)
    print("  ISIZULU AUTHOR ATTRIBUTION DATASET BUILDER (SELENIUM UPGRADE)")
    print("=" * 60)
    
    dataset = []
    
    for author, profile_url in AUTHOR_PROFILES.items():
        print(f"\n[Processing Author: {author}]")
        
        # 1. Gather deep historical collection via automated page expansions
        links = get_article_links_selenium(profile_url, clicks=CLICKS_NEEDED)
        print(f"  -> Discovered {len(links)} cumulative article links.")
        
        # 2. Slice to the maximum limit to guarantee dynamic balance 
        links = links[:MAX_ARTICLES_PER_AUTHOR]
        
        # 3. Request each separate text document block
        successful_scrapes = 0
        for i, link in enumerate(links):
            text = scrape_article_text(link)
            
            if text:
                dataset.append({
                    'author': author,
                    'url': link,
                    'text': text
                })
                successful_scrapes += 1
                
            # Anti-ban throttle
            time.sleep(1.2)
            
            if (i + 1) % 10 == 0:
                print(f"     Processed {i + 1}/{len(links)} URLs... Saved: {successful_scrapes}")
                
        print(f"  -> Finished {author}. Total clean records: {successful_scrapes}")

    # 4. Save and summarize outputs
    if dataset:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.DataFrame(dataset)
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        
        print("\n" + "=" * 50)
        print("  DATA COLLECTION SUMMARY")
        print("=" * 50)
        print(df['author'].value_counts())
        print(f"\nSaved file location: {OUTPUT_FILE}")
        print("Dataset verification complete. Ready for Weeks 3-4 Model Development!")
    else:
        print("\n[!] Execution completed with 0 records scraped. Verify selector nodes.")

if __name__ == "__main__":
    main()