import pandas as pd
import requests
import time
from pathlib import Path
import os
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

def extract_urls_with_sitemap_package(url):
    """Try to extract URLs using the sitemap package or create a custom sitemap."""
    try:
        # Ensure URL has proper protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Method 1: Try to find existing sitemap
        sitemap_urls = []
        common_locations = [
            '/sitemap.xml',
            '/sitemap_index.xml', 
            '/sitemaps.xml',
            '/sitemap.html'
        ]
        
        for location in common_locations:
            sitemap_url = url.rstrip('/') + location
            try:
                response = requests.head(sitemap_url, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    # Try to parse the sitemap
                    sitemap_response = requests.get(sitemap_url, timeout=30)
                    if sitemap_response.status_code == 200:
                        try:
                            root = ET.fromstring(sitemap_response.content)
                            extracted_urls = []
                            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                                loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                                if loc is not None:
                                    extracted_urls.append(loc.text)
                            if extracted_urls:
                                return extracted_urls, 'found_existing'
                        except ET.ParseError:
                            pass
            except:
                continue
        
        # Method 2: Crawl the website and generate sitemap
        print(f"  Crawling website to generate sitemap...")
        discovered_urls = []
        visited_urls = set()
        urls_to_visit = [url]
        
        max_pages = 100  # Limit to avoid infinite crawling
        max_depth = 3    # Limit crawl depth
        
        while urls_to_visit and len(discovered_urls) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            
            try:
                response = requests.get(current_url, timeout=15)
                if response.status_code == 200:
                    discovered_urls.append(current_url)
                    
                    # Extract links from the page
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            absolute_url = urljoin(url, href)
                        elif href.startswith('http'):
                            absolute_url = href
                        else:
                            continue
                        
                        # Only include URLs from the same domain
                        if urlparse(absolute_url).netloc == urlparse(url).netloc:
                            if absolute_url not in visited_urls and absolute_url not in urls_to_visit:
                                urls_to_visit.append(absolute_url)
                
            except Exception as e:
                print(f"    Error crawling {current_url}: {e}")
                continue
            
            time.sleep(1)  # Be respectful
        
        if discovered_urls:
            return discovered_urls, 'generated_crawl'
        
    except Exception as e:
        print(f"Error with sitemap package approach: {e}")
    
    return [], 'failed'

def save_sitemap(urls, filename):
    """Save URLs as an XML sitemap."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
            for url in urls:
                f.write(f'  <url><loc>{url}</loc></url>\n')
            f.write('</urlset>')
        
        return True
    except Exception as e:
        print(f"Error saving sitemap {filename}: {e}")
        return False

def get_domain_from_url(url):
    """Extract domain from URL."""
    if not url or url == '-':
        return None
    
    url = str(url).strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    parsed = urlparse(url)
    return parsed.netloc

def python_sitemap_retrieval():
    """Use python-sitemap approaches to retrieve missing sitemaps."""
    
    # Load missing entries
    mapping_df = pd.read_csv('sitemap_mapping.csv')
    missing_entries = mapping_df[mapping_df['sitemap_file'] == 'No sitemap'].copy()
    
    print(f"Attempting python-sitemap retrieval for {len(missing_entries)} missing entries...")
    
    # Create sitemaps directory
    sitemaps_dir = Path("./sitemaps")
    sitemaps_dir.mkdir(exist_ok=True)
    
    results = []
    
    for index, row in missing_entries.iterrows():
        acad_id = row['acad_id']
        acad_name = row['acad_name_en']
        website_url = row['website_url']
        
        print(f"\n=== Processing: {acad_name} ===")
        print(f"URL: {website_url}")
        
        if not website_url or website_url == '-':
            results.append({
                'acad_id': acad_id,
                'acad_name_en': acad_name,
                'website_url': website_url,
                'method': 'no_valid_url',
                'status': 'failed',
                'sitemap_file': '',
                'urls_found': 0
            })
            continue
        
        # Try to extract URLs using sitemap package approaches
        urls, method = extract_urls_with_sitemap_package(website_url)
        
        if urls:
            # Remove duplicates
            unique_urls = list(set(urls))
            
            # Generate sitemap filename
            domain = get_domain_from_url(website_url)
            if domain:
                sitemap_filename = f"{domain}_python_sitemap.xml"
                sitemap_path = sitemaps_dir / sitemap_filename
                
                if save_sitemap(unique_urls, str(sitemap_path)):
                    print(f"  ✓ SUCCESS: Created sitemap with {len(unique_urls)} URLs")
                    print(f"    Method: {method}")
                    print(f"    File: {sitemap_filename}")
                    
                    results.append({
                        'acad_id': acad_id,
                        'acad_name_en': acad_name,
                        'website_url': website_url,
                        'method': method,
                        'status': 'success',
                        'sitemap_file': sitemap_filename,
                        'urls_found': len(unique_urls)
                    })
                else:
                    print(f"  ✗ FAILED: Could not save sitemap")
                    results.append({
                        'acad_id': acad_id,
                        'acad_name_en': acad_name,
                        'website_url': website_url,
                        'method': method,
                        'status': 'save_failed',
                        'sitemap_file': '',
                        'urls_found': len(unique_urls)
                    })
            else:
                print(f"  ✗ FAILED: Could not extract domain")
                results.append({
                    'acad_id': acad_id,
                    'acad_name_en': acad_name,
                    'website_url': website_url,
                    'method': method,
                    'status': 'domain_extraction_failed',
                    'sitemap_file': '',
                    'urls_found': len(unique_urls)
                })
        else:
            print(f"  ✗ FAILED: No URLs found")
            results.append({
                'acad_id': acad_id,
                'acad_name_en': acad_name,
                'website_url': website_url,
                'method': method,
                'status': 'no_urls_found',
                'sitemap_file': '',
                'urls_found': 0
            })
        
        # Add delay to be respectful
        time.sleep(2)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = "python_sitemap_retrieval_results.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n=== PYTHON-SITEMAP RETRIEVAL SUMMARY ===")
    print(f"Total processed: {len(results)}")
    
    status_counts = results_df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"{status}: {count}")
    
    method_counts = results_df[results_df['status'] == 'success']['method'].value_counts()
    if not method_counts.empty:
        print(f"\nSuccessful methods:")
        for method, count in method_counts.items():
            print(f"  {method}: {count}")
    
    sitemaps_created = len(results_df[results_df['status'] == 'success'])
    print(f"\nNew sitemaps created: {sitemaps_created}")
    print(f"Results saved to: {results_file}")
    
    return results_df

if __name__ == "__main__":
    python_sitemap_retrieval()
