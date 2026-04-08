import xml.etree.ElementTree as ET
import pandas as pd

def parse_marc_xml_to_excel(xml_file_path, output_excel_path):
    # Register the MARCXML namespace
    namespaces = {'marc': 'http://www.loc.gov/MARC21/slim'}
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    results = []

    # Iterate through each bibliographic record
    for record in root.findall('marc:record', namespaces):
        zdb_id = "N/A"
        
        # 1. Extract ZDB-ID from Datafield 016 (Source DE-600)
        for datafield in record.findall('marc:datafield[@tag="016"]', namespaces):
            source_code = None
            value_code = None
            for subfield in datafield.findall('marc:subfield', namespaces):
                code = subfield.get('code')
                if code == '2':
                    source_code = subfield.text
                elif code == 'a':
                    value_code = subfield.text
            
            if source_code == 'DE-600' and value_code:
                zdb_id = value_code
                break
        
        # 2. Extract Corporate Bodies from tags 110, 710, and 610
        target_tags = ['110', '710', '610']
        for tag in target_tags:
            for datafield in record.findall(f'marc:datafield[@tag="{tag}"]', namespaces):
                corp_name_parts = []
                koe_ref = "N/A"
                
                for subfield in datafield.findall('marc:subfield', namespaces):
                    code = subfield.get('code')
                    if code == 'a':
                        corp_name_parts.append(subfield.text)
                    elif code == 'b':
                        corp_name_parts.append(f"({subfield.text})")
                    elif code == '0':
                        koe_ref = subfield.text
                
                if corp_name_parts:
                    full_name = " ".join(corp_name_parts)
                    results.append({
                        "ZDB-ID": zdb_id,
                        "Corporate Body": full_name,
                        "koeRef": koe_ref
                    })

    # Create a DataFrame and export to Excel
    df = pd.DataFrame(results)
    
    # Exporting to Excel
    # index=False prevents the row numbers (0, 1, 2...) from being saved as a column
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"Successfully saved {len(df)} rows to '{output_excel_path}'")

# Execute the function
input_file = './zdb_result/zdb_academies_collection.xml'
output_file = './zdb_result/zdb_academies_mapping.xlsx'

parse_marc_xml_to_excel(input_file, output_file)