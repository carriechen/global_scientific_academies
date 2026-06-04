import requests
import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup
from urllib.parse import urljoin,urlparse
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path

def is_archived(url):
    """Check if a URL is already archived in the Wayback Machine."""
    api_url = f"https://archive.org/wayback/available?url={url}"
    try:
        response = requests.get(api_url, timeout=10)
        data = response.json()
        return data['archived_snapshots'] != {}
    except Exception as e:
        print(f"Error checking archive status for {url}: {e}")
        return False

def submit_to_wayback(url):
    """Submit a URL to the Wayback Machine for archiving."""
    save_url = "https://web.archive.org/save/"
    try:
        response = requests.post(save_url, data={"url": url}, timeout=30)
        if response.status_code == 200:
            print(f"✓ {url} has been submitted to the Wayback Machine.")
            return True
        else:
            print(f"✗ Failed to submit {url}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error submitting {url}: {e}")
        return False

def extract_urls_from_archive(archive_url):
    """Extract archived URLs from the Wayback Machine CDX API."""
    try:
        # Construct the CDX API URL
        cdx_api_url = f"http://web.archive.org/cdx/search/cdx?url={archive_url}&matchType=prefix&collapse=urlkey&filter=statuscode:200&output=json"
        response = requests.get(cdx_api_url, timeout=30)
        response.encoding = 'utf-8'
        data = response.json()
        
        extracted_urls = set()
        
        for entry in data[1:]:  # Skip the header row
            if len(entry) >= 3:
                captured_url = entry[2]  # The URL is in the third column
                timestamp = entry[1]  # Timestamp is in the second column
                extracted_urls.add(f"http://web.archive.org/web/{timestamp}/{captured_url}")
        
        return extracted_urls
    except Exception as e:
        print(f"Error extracting URLs from archive for {archive_url}: {e}")
        return set()

def save_sitemap(urls, filename):
    """Save URLs as an XML sitemap."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
            for url in urls:
                f.write(f"  <url><loc>{url}</loc></url>\n")
            f.write('</urlset>')
        print(f"✓ Sitemap saved as {filename} ({len(urls)} URLs)")
        return True
    except Exception as e:
        print(f"✗ Error saving sitemap {filename}: {e}")
        return False

def load_processed_urls(log_file="processed_urls.log"):
    """Load previously processed URLs from log file."""
    processed = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    url = line.strip()
                    if url:
                        processed.add(url)
        except Exception as e:
            print(f"Error loading processed URLs: {e}")
    return processed

def log_processed_url(url, action, log_file="processed_urls.log"):
    """Log a processed URL to avoid reprocessing."""
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{url}\n")
    except Exception as e:
        print(f"Error logging processed URL: {e}")

def main():
    """Main function to process academy websites."""
    # Load the academy data
    input_file = "global_science_academies_final.xlsx"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    print(f"Loading data from {input_file}...")
    df = pd.read_excel(input_file)
    
    if 'website_url' not in df.columns:
        print(f"Error: 'website_url' column not found in {input_file}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Extract website URLs
    urls = df['website_url'].dropna().values
    urls = [url for url in urls if url and str(url).strip() and str(url).strip() != '-']
    
    print(f"Found {len(urls)} valid website URLs to process")
    
    # Load previously processed URLs
    processed_urls = load_processed_urls()
    print(f"Skipping {len(processed_urls)} previously processed URLs")
    
    # Filter out already processed URLs
    urls_to_process = [url for url in urls if url not in processed_urls]
    print(f"Processing {len(urls_to_process)} new URLs")
    
    # Create sitemaps directory
    sitemaps_dir = Path("./sitemaps")
    sitemaps_dir.mkdir(exist_ok=True)
    
    # Process URLs
    results = []
    
    for url in tqdm(urls_to_process, desc="Processing URLs"):
        url = str(url).strip()
        
        if not url or url == '-':
            continue
        
        try:
            # Check if URL is already archived
            if not is_archived(url):
                print(f"Submitting {url} for archiving...")
                success = submit_to_wayback(url)
                if success:
                    results.append((url, 'submitted_for_archiving'))
                    log_processed_url(url, 'submitted')
                else:
                    results.append((url, 'submission_failed'))
                    log_processed_url(url, 'failed')
            else:
                print(f"Extracting archived URLs for {url}...")
                url_list = extract_urls_from_archive(url)
                
                if url_list:
                    filename = sitemaps_dir / f"{urlparse(url).netloc}_sitemap.xml"
                    if save_sitemap(url_list, str(filename)):
                        results.append((url, str(filename)))
                        log_processed_url(url, 'sitemap_created')
                    else:
                        results.append((url, 'sitemap_failed'))
                        log_processed_url(url, 'sitemap_failed')
                else:
                    print(f"No archived URLs found for {url}")
                    results.append((url, 'no_archive_found'))
                    log_processed_url(url, 'no_archive')
            
            # Add delay to be respectful to the APIs
            import time
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            results.append((url, f'error: {str(e)}'))
            log_processed_url(url, 'error')
    
    # Save results summary
    results_file = "internet_archive_results.csv"
    results_df = pd.DataFrame(results, columns=['url', 'status'])
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Print summary statistics
    status_counts = results_df['status'].value_counts()
    print("\nProcessing Summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    print(f"\nTotal URLs processed: {len(results)}")
    print(f"Sitemaps created: {len([r for r in results if 'sitemap' in r[1] and 'failed' not in r[1]])}")

if __name__ == "__main__":
    main()
