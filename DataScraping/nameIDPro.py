from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options  # Import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from random import randint
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import csv

# Set up Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")  # Enable headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
chrome_options.add_argument("--window-size=1920x1080")  # Set the window size

# load the driver with headless options
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# get the web page
driver.get('https://www.ratemyprofessors.com/search/professors/1257?q=*')

print("Passed line 25:")

# Click the button to load more professors
for n in range(800):  # Adjust the number of clicks as needed
    try:
        button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".Buttons__Button-sc-19xdot-1.PaginationButton__StyledPaginationButton-txi1dr-1.eUNaBX"))
        )
        driver.execute_script("arguments[0].click();", button)  # Click using JavaScript
        time.sleep(randint(1, 5))
    except (TimeoutException, NoSuchElementException) as e:
        print(f"Element not found or clickable: {e}")
        break  #

# Get the updated page source
page = driver.page_source
print("Passed line 35:")
# Parse the page using BeautifulSoup
soup = BeautifulSoup(page, "html.parser")
professors = soup.find_all('a', class_='TeacherCard__StyledTeacherCard-syjs0d-0')

# Extract names and IDs
print(professors)
names_and_ids = []
with open("nameIDPro.csv", "w+", newline="") as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)
    
    # Write the header row
    csvwriter.writerow(["Name", "ID"])
    
    for professor in professors:
        print("Looping...:")
        name = professor.find('div', class_='CardName__StyledCardName-sc-1gyrgim-0').text.strip()
        department = professor.find('div', class_='CardSchool__Department-sc-19lmz2k-0').text.strip()
        id = professor['href'].split('/')[-1]
        if department == "Computer Science":
            names_and_ids.append((name, id))
            csvwriter.writerow([name, id])
            print(name, ": ", id)

print("There are " + str(len(names_and_ids)) + " professors in the list.")

# Quit the driver
driver.quit()
