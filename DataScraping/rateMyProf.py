from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# load the driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# get the web page
driver.get('https://www.ratemyprofessors.com/professor/879690')

# Define the files for different scores
files = {
    "1": open("oneRate.txt", "w+"),
    "2": open("twoRate.txt", "w+"),
    "3": open("threeRate.txt", "w+"),
    "4": open("fourRate.txt", "w+"),
    "5": open("fiveRate.txt", "w+"),
}

# try:
#     while True:  # This will keep executing until the "Show More" button is not found
#         # Wait for the button to be clickable and click it
#         show_more_button = WebDriverWait(driver, 10).until(
#             EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Show More')]"))
#         )
#         show_more_button.click()
#         time.sleep(randint(1, 3))  # Shorter sleep for demonstration purposes
# except Exception as e:
#     print("No more 'Show More' buttons or an error occurred:", e)

# Find all elements containing ratings and comments
ratings = driver.find_elements(By.CSS_SELECTOR, 'div.Rating__RatingBody-sc-1rhvpxz-0')

# Extract ratings and comments
for rating in ratings:
    try:                                              #   CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 gcFhmN
        score = rating.find_element(By.CSS_SELECTOR, 'div.CardNumRating__CardNumRatingNumber-sc-17t4b9u-2').text[0]  # Assuming score is the first character
                                                        #   Comments__StyledComments-dzzyvm-0 gRjWel
        comment = rating.find_element(By.CSS_SELECTOR, 'div.Comments__StyledComments-dzzyvm-0').text.strip()
        if score in files:
            files[score].write(comment + '\n')
    except Exception as e:
        print("Error processing a rating:", e)

# Close all files
for file in files.values():
    file.close()

driver.quit()