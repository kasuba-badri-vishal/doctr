import requests
from bs4 import BeautifulSoup
import os

username = 'iitb'
password = 'iitb123'
# URL pattern for the pages
base_url = "https://ilocr.iiit.ac.in/bhashini-iiith/dataset/"
login_url = 'https://ilocr.iiit.ac.in/bhashini-iiith/user/login/?next=/bhashini-iiith/dataset/'


session = requests.Session()

# Login to the website
login_data = {
    'username': username,
    'password': password
}

# Send GET request to get CSRF token
login_page = session.get(login_url)
soup = BeautifulSoup(login_page.content, 'html.parser')
csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'}).get('value')

# Include CSRF token in login data
login_data['csrfmiddlewaretoken'] = csrf_token

# Send POST request to login with CSRF token
login_response = session.post(login_url, data=login_data)

# Loop through all the pages
# Loop through all the pages
for page_num in range(87,207):  # Change the range to loop through all pages
    url = f"{base_url}{page_num}/"
    print(f"Scraping {url}")

    # Send a GET request to the page
    response = session.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        print(soup)
        # Find all download links
        download_links = soup.find_all('a', {'class': 'btn btn-success', 'download': True})
        print(download_links)
        # Create a directory to store the downloaded files
        directory = f"/raid/ganesh/badri/RECOGNITION/data/bhashini_baselines"
        os.makedirs(directory, exist_ok=True)

        # Download each file
        for link in download_links:
            file_url = link['href']
            print(file_url)
            if file_url.endswith('.json'):  # Check if the link ends with .json
                file_name = file_url.split('/')[-1]
                file_path = os.path.join(directory, file_name)

                # Download the file
                file_url = f"https://ilocr.iiit.ac.in{file_url}"
                print(file_url)
                file_response = session.get(file_url)
                print(f"File URL: {file_url}")
                print(f"File Response Status Code: {file_response.status_code}")
                if file_response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(file_response.content)
                    print(f"Downloaded {file_name}")
                else:
                    print(f"Failed to download {file_name}")
    else:
        print(f"Failed to fetch {url}")