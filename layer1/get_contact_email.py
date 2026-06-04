import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Read the URLs from the uploaded file
file_path = 'website.txt'

# Function to extract contact email from a website
def get_contact_email(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all email-like strings from the webpage
        emails = set()
        for a_tag in soup.find_all('a', href=True):
            if 'mailto:' in a_tag['href']:
                email = a_tag['href'].split('mailto:')[1]
                emails.add(email)
        
        return ', '.join(emails) if emails else 'No email found'
    except requests.RequestException:
        return 'Error accessing website'

# Read URLs from the file
with open(file_path, 'r') as file:
    urls = [line.strip() for line in file.readlines()]

# Prepare CSV data
csv_data = [['Website', 'Contact Email']]
for url in urls:
    email = get_contact_email(url)
    csv_data.append([url, email])

# Output file path
output_csv_path = 'website_contact_emails.csv'

# Write CSV data to a file
with open(output_csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

output_csv_path
