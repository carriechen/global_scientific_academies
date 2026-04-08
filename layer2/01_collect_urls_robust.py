from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import Optional, List
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
import chardet

EXCLUDE_EXTENSIONS = {
    ".js", ".css", ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".pdf", ".woff", ".woff2", ".ttf", ".eot",
}

def parse_sitemap_standard(sitemap_path: Path) -> List[str]:
    """Standard XML parsing method."""
    try:
        tree = ET.parse(sitemap_path)
        root = tree.getroot()
        
        urls = []
        for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
            loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            if loc is not None:
                urls.append(loc.text.strip())
        
        return urls
    except Exception:
        return []

def parse_sitemap_regex(sitemap_path: Path) -> List[str]:
    """Regex-based URL extraction for malformed XML."""
    try:
        # Try to detect encoding first
        with sitemap_path.open('rb') as f:
            raw = f.read()
        
        encoding = chardet.detect(raw).get('encoding') or 'utf-8'
        
        # Try different encodings
        for enc in [encoding, 'utf-8', 'latin-1', 'cp1252']:
            try:
                content = raw.decode(enc, errors='ignore')
                break
            except:
                content = raw.decode('utf-8', errors='ignore')
        
        # Extract URLs using regex patterns
        urls = []
        
        # Pattern 1: Standard <loc> tags
        pattern1 = r'<loc[^>]*>([^<]+)</loc>'
        matches1 = re.findall(pattern1, content, re.IGNORECASE)
        urls.extend([match.strip() for match in matches1])
        
        # Pattern 2: URLs in text (fallback)
        pattern2 = r'https?://[^\s<>"\'\)]+'
        matches2 = re.findall(pattern2, content)
        urls.extend(matches2)
        
        # Clean and filter URLs
        cleaned_urls = []
        for url in urls:
            url = url.strip()
            if url.startswith(('http://', 'https://')) and len(url) > 10:
                # Remove common XML artifacts
                url = re.sub(r'<![^\[]*>', '', url)
                url = re.sub(r'&[^;]*;', '', url)
                cleaned_urls.append(url)
        
        return list(set(cleaned_urls))  # Remove duplicates
        
    except Exception:
        return []

def parse_sitemap_line_by_line(sitemap_path: Path) -> List[str]:
    """Line-by-line parsing for severely corrupted XML."""
    try:
        urls = []
        
        with sitemap_path.open('rb') as f:
            raw = f.read()
        
        encoding = chardet.detect(raw).get('encoding') or 'utf-8'
        
        try:
            content = raw.decode(encoding, errors='ignore')
        except:
            content = raw.decode('utf-8', errors='ignore')
        
        # Split by lines and process each line
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for URLs in each line
            if 'http' in line:
                # Extract URL patterns
                url_patterns = [
                    r'https?://[^\s<>"\'\)]+',
                    r'<loc[^>]*>([^<]+)</loc>',
                ]
                
                for pattern in url_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        url = match.strip()
                        if url.startswith(('http://', 'https://')) and len(url) > 10:
                            urls.append(url)
        
        return list(set(urls))
        
    except Exception:
        return []

def parse_sitemap_fallback(sitemap_path: Path) -> List[str]:
    """Ultimate fallback - extract any HTTP URLs from the file."""
    try:
        with sitemap_path.open('rb') as f:
            raw = f.read()
        
        # Try different encodings
        for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                content = raw.decode(enc, errors='ignore')
                break
            except:
                content = raw.decode('utf-8', errors='ignore')
        
        # Find all HTTP URLs
        url_pattern = r'https?://[^\s<>"\'\)]+(?=\s|$|<|>|"|\')'
        urls = re.findall(url_pattern, content, re.IGNORECASE)
        
        # Filter and clean
        cleaned_urls = []
        for url in urls:
            url = url.strip()
            if len(url) > 10 and not any(ext in url.lower() for ext in EXCLUDE_EXTENSIONS):
                cleaned_urls.append(url)
        
        return list(set(cleaned_urls))
        
    except Exception:
        return []

def is_valid_html_url(url: str) -> bool:
    """Check if URL is valid HTML (not a file download)."""
    lower = url.lower()
    if any(ext in lower for ext in EXCLUDE_EXTENSIONS):
        return False
    return True

def get_domain_from_url(url: str) -> Optional[str]:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return None

def parse_sitemap_robust(sitemap_path: Path) -> tuple[List[str], str]:
    """Parse sitemap using multiple methods in order of preference."""
    
    # Method 1: Standard XML parsing
    urls = parse_sitemap_standard(sitemap_path)
    if urls:
        return urls, 'standard_xml'
    
    # Method 2: Regex extraction
    urls = parse_sitemap_regex(sitemap_path)
    if urls:
        return urls, 'regex_extraction'
    
    # Method 3: Line-by-line parsing
    urls = parse_sitemap_line_by_line(sitemap_path)
    if urls:
        return urls, 'line_by_line'
    
    # Method 4: Ultimate fallback
    urls = parse_sitemap_fallback(sitemap_path)
    if urls:
        return urls, 'fallback_extraction'
    
    return [], 'failed'

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sitemap-mapping", required=True, help="sitemap_mapping.csv")
    parser.add_argument("--sitemap-dir", required=True, help="Directory containing sitemap XML files")
    parser.add_argument("--output-dir", default="websiteanalysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sitemap mapping
    mapping_df = pd.read_csv(args.sitemap_mapping)
    
    # Filter to entries that have sitemaps
    sitemap_entries = mapping_df[mapping_df['sitemap_file'] != 'No sitemap'].copy()
    
    print(f"Processing {len(sitemap_entries)} sitemap files...")
    
    # Create sitemap dictionary mapping file paths to domains
    sitemap_dict = {}
    for _, row in sitemap_entries.iterrows():
        sitemap_file = row['sitemap_file']
        website_url = row['website_url']
        
        if website_url and str(website_url) != '-':
            domain = get_domain_from_url(str(website_url))
            if domain:
                sitemap_path = Path(args.sitemap_dir) / sitemap_file
                sitemap_dict[str(sitemap_path)] = domain
    
    # Process all sitemap files
    all_rows = []
    parsing_stats = {
        'standard_xml': 0,
        'regex_extraction': 0,
        'line_by_line': 0,
        'fallback_extraction': 0,
        'failed': 0
    }
    
    sitemap_dir = Path(args.sitemap_dir)
    
    for sitemap_path in tqdm(list(sitemap_dir.glob("*.xml")), desc="Processing sitemaps"):
        if str(sitemap_path) not in sitemap_dict:
            continue
            
        domain = sitemap_dict[str(sitemap_path)]
        urls, method = parse_sitemap_robust(sitemap_path)
        
        parsing_stats[method] += 1
        
        for url in urls:
            if url and url.startswith(("http://", "https://")) and is_valid_html_url(url):
                all_rows.append({
                    'site_name': sitemap_path.name,
                    'loc': url,
                    'sitedomain': domain,
                    'parsing_method': method
                })
    
    # Create DataFrame
    df_alldom = pd.DataFrame(all_rows)
    
    if df_alldom.empty:
        print("No valid URLs found!")
        return
    
    # Remove duplicates
    df_alldom = df_alldom.drop_duplicates(subset=['loc'], keep='first')
    
    # Filter to only URLs from valid domains
    valid_domains = set(sitemap_dict.values())
    df_alldom['ifdomain'] = df_alldom['loc'].apply(lambda x: any(domain in x.lower() for domain in valid_domains))
    df_alldom = df_alldom[df_alldom['ifdomain']].copy()
    
    # Save results
    output_path = output_dir / "dfalldom.feather"
    df_alldom.reset_index(drop=True).to_feather(output_path)
    
    print(f"Saved: {output_path} ({len(df_alldom):,} rows)")
    print(f"Unique domains: {df_alldom['sitedomain'].nunique()}")
    print(f"Unique sitemaps: {df_alldom['site_name'].nunique()}")
    
    print("\n=== PARSING METHOD STATISTICS ===")
    for method, count in parsing_stats.items():
        print(f"{method:20} {count:3} sitemaps")
    
    # Save parsing statistics
    stats_df = pd.DataFrame(list(parsing_stats.items()), columns=['method', 'count'])
    stats_path = output_dir / "parsing_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Parsing stats saved to: {stats_path}")

if __name__ == "__main__":
    main()
