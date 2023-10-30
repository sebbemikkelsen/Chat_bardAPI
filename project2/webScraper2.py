import requests
from bs4 import BeautifulSoup
import os
import re

# Function to sanitize filenames
def sanitize_filename(filename):
    # Remove characters that are not allowed in filenames
    sanitized_name = re.sub(r'[\\/:"*?<>|]', '_', filename)
    # Remove spaces and parentheses
    sanitized_name = re.sub(r'[\s()\[\]{}]', '', sanitized_name)
    return sanitized_name

# URL of the main page with the table of subpage links
main_page_url = 'https://www4.skatteverket.se/rattsligvagledning/edition/2023.14/2151.html'

# Create a directory to save the text files
if not os.path.exists('subpage_texts2'):
    os.mkdir('subpage_texts2')

# Send an HTTP GET request to the main page
main_page_response = requests.get(main_page_url)

# Check if the request was successful (status code 200)
if main_page_response.status_code == 200:
    main_page_soup = BeautifulSoup(main_page_response.text, 'html.parser')

    # Find all the links to subpages within the table
    subpage_links = main_page_soup.find_all('a', class_='brodtextxreferens', href=True)

    # Loop through the subpage links and scrape text from each subpage
    for link in subpage_links:
        subpage_url = 'https://www4.skatteverket.se' + link['href']  # Construct the absolute URL
        subpage_response = requests.get(subpage_url)
       
        if subpage_response.status_code == 200:
            subpage_soup = BeautifulSoup(subpage_response.text, 'html.parser')
           
            # Find the div with class "body searchable-content" and scrape its text
            content_div = subpage_soup.find('div', class_='body searchable-content')
           
            if content_div:
                # Get all the paragraphs within the content div
                paragraphs = content_div.find_all('p')
               
                # Combine the text from all paragraphs into a single string
                text = '\n'.join([p.get_text() for p in paragraphs])

                # Sanitize the filename
                sanitized_filename = sanitize_filename(link.text.strip())
               
                # Create a unique filename for each subpage
                filename = f'subpage_texts2/{sanitized_filename}.txt'
               
                # Write the text to the file
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(text)
        else:
            print(f"Failed to retrieve subpage {subpage_url}. Status code: {subpage_response.status_code}")
else:
    print(f"Failed to retrieve the main page. Status code: {main_page_response.status_code}")