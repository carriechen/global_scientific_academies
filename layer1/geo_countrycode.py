import pandas as pd
import requests
import time
import pycountry
import langcodes

# Function to get geocoding information
def get_geocoding_info(address, api_key):
    if pd.isna(address) or address == '' or address == 'N/A':
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
    
    api_url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        'q': address,
        'key': api_key,
        'language': 'en',
        'pretty': 1
    }
    
    try:
        response = requests.get(api_url, params=params)
        data = response.json()
        
        if data['results']:
            result = data['results'][0]
            city = result['components'].get('city', result['components'].get('town', result['components'].get('village', 'N/A')))
            country = result['components'].get('country', 'N/A')
            continent = result['components'].get('continent', 'N/A')
            latitude = result['geometry']['lat']
            longitude = result['geometry']['lng']
            
            return city, country, continent, latitude, longitude
        else:
            return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
    except Exception as e:
        print(f"Error geocoding address '{address}': {e}")
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

# Read the Excel file with academy data
input_file = "global_science_academies V1_filled_ids_entity_ids_checked.xlsx"
df = pd.read_excel(input_file)

print(f"Loaded {len(df)} records from {input_file}")
print(f"Columns: {df.columns.tolist()}")

# Your OpenCage Geocoding API key
api_key = "fcbbb4936a5b4b9fa9bf90955cccf2ed"  # Replace with your actual API key

# Count rows that need geocoding
needs_geocoding = df[(df['latitude'].isna()) | (df['longitude'].isna()) | 
                     (df['latitude'] == 'N/A') | (df['longitude'] == 'N/A')].copy()
print(f"Found {len(needs_geocoding)} rows that need geocoding")

# Process each row that needs geocoding
for index, row in needs_geocoding.iterrows():
    # Try address first, then headquarters
    address_to_geocode = None
    if pd.notna(row['address']) and row['address'] != '' and row['address'] != 'N/A':
        address_to_geocode = row['address']
    elif pd.notna(row['headquarters']) and row['headquarters'] != '' and row['headquarters'] != 'N/A':
        address_to_geocode = row['headquarters']
    
    if address_to_geocode:
        print(f"Geocoding row {index + 1}: {address_to_geocode}")
        city, country, continent, latitude, longitude = get_geocoding_info(address_to_geocode, api_key)
        
        # Update the dataframe
        df.at[index, 'city'] = city if city != 'N/A' else df.at[index, 'city']
        df.at[index, 'country'] = country if country != 'N/A' else df.at[index, 'country']
        df.at[index, 'continent'] = continent if continent != 'N/A' else df.at[index, 'continent']
        df.at[index, 'latitude'] = latitude if latitude != 'N/A' else df.at[index, 'latitude']
        df.at[index, 'longitude'] = longitude if longitude != 'N/A' else df.at[index, 'longitude']
        
        time.sleep(1)  # Sleep to respect API rate limits
    else:
        print(f"No valid address found for row {index + 1}")

# Save the updated dataframe back to Excel
output_file = "global_science_academies_updated_geocoding.xlsx"
df.to_excel(output_file, index=False)

print(f"Updated geocoding data saved to {output_file}")
print(f"Total records processed: {len(df)}")
print(f"Records with valid coordinates: {len(df[df['latitude'].notna() & df['longitude'].notna()])}")

# Function to get country language information
def getcountry(x):
    language_code=''
    language=''
    try:
        if pd.notna(x) and x != '' and x != 'N/A':
            alpha2=pycountry.countries.search_fuzzy(x)[0].alpha_2
            language=pycountry.languages.get(alpha_2=alpha2).name
            language_code=langcodes.find(language).language
    except:
        pass
    return language_code,language

# Update language information based on country
if 'country' in df.columns:
    a=df.country.apply(getcountry)
    
    if 'origin_lang' in df.columns:
        ta=pd.concat([df[['origin_lang']],a.apply(lambda x:x[1]).to_frame()],axis=1)
        ta['origin_lang_a']=ta.fillna('').apply(lambda x:x[1] if x[1] else x[0],axis=1)
        df['origin_lang']=ta['origin_lang_a']
    
    if 'website_lang' in df.columns:
        ta=pd.concat([df[['website_lang']],a.apply(lambda x:x[0]).to_frame()],axis=1)
        ta['website_lang_a']=ta.fillna('').apply(lambda x:x[1] if x[1] else x[0],axis=1)
        df['website_lang']=ta['website_lang_a']
    
    # Save the final updated file
    final_output_file = "global_science_academies_final.xlsx"
    df.to_excel(final_output_file, index=False)
    print(f"Final updated data with languages saved to {final_output_file}")
