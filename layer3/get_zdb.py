import pandas as pd
import requests
import time
import xml.etree.ElementTree as ET
import csv
import os

# --- CONFIGURATION ---
INPUT_EXCEL = 'global_science_academies_final.xlsx'  # Updated filename
OUTPUT_DIR = 'zdb_result'
OUTPUT_XML = os.path.join(OUTPUT_DIR, 'zdb_academies_collection.xml')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'zdb_academies_readable.csv')

# Namespace map for parsing XML
NS = {
    'sru': 'http://www.loc.gov/zing/srw/',
    'marc': 'http://www.loc.gov/MARC21/slim'
}
ET.register_namespace('', "http://www.loc.gov/MARC21/slim")

# Mapping Dictionary: Converts MARC Codes -> Human Readable Headers
MARC_MAPPING = {
    # Custom Tracking Fields
    'acad_id': 'Academy_ID',
    'Source_Academy_Query': 'Matched_Search_Term',
    
    # Standard MARC Fields
    '001': 'ZDB-ID',
    '005': 'Last Updated',
    '008': 'Fixed Data',
    '022_a': 'ISSN',
    '024_a': 'Other ID (DOI/URN)',
    '024_2': 'ID Type',
    '035_a': 'System Control No',
    '040_a': 'Cataloging Agency',
    '041_a': 'Language',
    '044_c': 'Country Code',
    '110_a': 'Corporate Author',
    '245_a': 'Title',
    '245_b': 'Subtitle',
    '245_c': 'Responsibility',
    '246_a': 'Alternate Title',
    '264_a': 'Place of Publication',
    '264_b': 'Publisher',
    '264_c': 'Date',
    '362_a': 'Volume/Date Range',
    '533_a': 'Reproduction Note',
    '655_a': 'Genre',
    '710_a': 'Corporate Body Added',
    '776_t': 'Other Format Title',
    '780_t': 'Preceding Title',
    '785_t': 'Succeeding Title',
    '856_u': 'URL'
}

def fetch_records_for_query(query_string):
    """
    Queries ZDB for a specific string using the Corporate Body index (koe).
    """
    if not query_string or pd.isna(query_string):
        return []

    # Clean the string (strip whitespace)
    query_string = str(query_string).strip()
    if not query_string:
        return []

    base_url = "https://services.dnb.de/sru/zdb"
    cql_query = f'koe="{query_string}"'
    
    records_found = []
    start_record = 1
    max_per_request = 100
    
    print(f"  --> Searching: '{query_string}'...", end=" ")

    while True:
        params = {
            'version': '1.1',
            'operation': 'searchRetrieve',
            'query': cql_query,
            'recordSchema': 'MARC21-xml',
            'maximumRecords': max_per_request,
            'startRecord': start_record
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"[API Error: {response.status_code}]")
                break
                
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                print("[XML Parse Error]")
                break

            # Check count on first page
            if start_record == 1:
                count_node = root.find('.//sru:numberOfRecords', NS)
                total = int(count_node.text) if count_node is not None else 0
                print(f"Found {total} hits.")
                if total == 0:
                    break

            batch = root.findall('.//marc:record', NS)
            if not batch:
                break
                
            records_found.extend(batch)
            
            # Pagination
            count_node = root.find('.//sru:numberOfRecords', NS)
            total = int(count_node.text) if count_node is not None else 0
            
            if start_record + len(batch) > total:
                break
            
            start_record += len(batch)
            time.sleep(0.5) 

        except Exception as e:
            print(f"[Error: {e}]")
            break
            
    return records_found

def parse_record_to_dict(xml_record, acad_id, source_query):
    """
    Flattens MARC XML to dict, attaching the ID and Query used.
    """
    data = {
        'acad_id': acad_id,
        'Source_Academy_Query': source_query
    }
    
    # 1. Control Fields
    for cf in xml_record.findall('marc:controlfield', NS):
        tag = cf.get('tag')
        if tag:
            data[tag] = cf.text

    # 2. Data Fields
    for df in xml_record.findall('marc:datafield', NS):
        tag = df.get('tag')
        for sf in df.findall('marc:subfield', NS):
            code = sf.get('code')
            text = sf.text
            if tag and code and text:
                key = f"{tag}_{code}"
                if key in data:
                    data[key] += f" | {text}"
                else:
                    data[key] = text
    return data

def main():
    # 0. Create Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Read Excel File
    print(f"Reading {INPUT_EXCEL}...")
    try:
        df = pd.read_excel(INPUT_EXCEL)
        
        # Check for required columns
        required_cols = ['acad_id', 'acad_name', 'acad_name_en']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns in Excel: {missing_cols}")
            return
            
        print(f"Processing {len(df)} rows.")
        
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # 2. Iterate Rows
    full_xml_collection = ET.Element("{http://www.loc.gov/MARC21/slim}collection")
    all_flat_records = []
    
    # Use iterrows to get ID and both names row by row
    for index, row in df.iterrows():
        acad_id = row['acad_id']
        name_native = row['acad_name']
        name_english = row['acad_name_en']
        
        print(f"\nProcessing ID: {acad_id}")

        # List of queries to run for this row
        # We filter out empty/NaN values
        queries = []
        if pd.notna(name_native) and str(name_native).strip():
            queries.append(name_native)
        
        # Avoid duplicate search if English name is same as Native
        if pd.notna(name_english) and str(name_english).strip():
            if str(name_english).strip() != str(name_native).strip():
                queries.append(name_english)

        # Run Search for each valid name
        for q in queries:
            xml_records = fetch_records_for_query(q)
            
            for rec in xml_records:
                # Add to XML Collection
                full_xml_collection.append(rec)
                
                # Parse for CSV (Injecting ID and Query Term)
                flat_rec = parse_record_to_dict(rec, acad_id, q)
                all_flat_records.append(flat_rec)

    # 3. Save Master XML
    print(f"\nSaving {len(all_flat_records)} total records to {OUTPUT_XML}...")
    tree = ET.ElementTree(full_xml_collection)
    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)

    # 4. Save Mapped CSV
    print(f"Saving readable CSV to {OUTPUT_CSV}...")
    if all_flat_records:
        df_out = pd.DataFrame(all_flat_records)
        df_out.rename(columns=MARC_MAPPING, inplace=True)
        
        # Sort Columns: IDs first
        cols = list(df_out.columns)
        priority_cols = ['Academy_ID', 'Matched_Search_Term', 'ZDB-ID', 'Title', 'ISSN', 'Publisher', 'Date']
        sorted_cols = [c for c in priority_cols if c in cols] + \
                      sorted([c for c in cols if c not in priority_cols])
        
        df_out = df_out[sorted_cols]
        df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        
        print("Done!")
    else:
        print("No records found.")
    
if __name__ == "__main__":
    main()