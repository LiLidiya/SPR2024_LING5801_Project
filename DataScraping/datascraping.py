from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options  # Import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from random import randint

chrome_options = Options()
chrome_options.add_argument("--headless")  # Enable headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
chrome_options.add_argument("--window-size=1920x1080")  # Set the window size

# load the driver with headless options
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

files = {
    "1": open("oneRate.txt", "w+"),
    "2": open("twoRate.txt", "w+"),
    "3": open("threeRate.txt", "w+"),
    "4": open("fourRate.txt", "w+"),
    "5": open("fiveRate.txt", "w+"),
}

fd = open("professorsURL.txt","r")  # Open a file that contains professor's review page URL
num_prof = 1    # To count number of professor
# get the web page
for profUrl in fd:
    print(num_prof,"prof:",end="")
    print(profUrl[:len(profUrl)-1])
    driver.get(profUrl[:len(profUrl)-1])    # Move to the URL
    for n in range(800):  # Adjust the number of clicks as needed
        try:
            button = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.Buttons__Button-sc-19xdot-1.PaginationButton__StyledPaginationButton-txi1dr-1.eUNaBX"))
            )
            driver.execute_script("arguments[0].click();", button)  # Click using JavaScript
            time.sleep(randint(1, 5))
            print("Button Clicked")
        except:
            print(f"Element not found or clickable")
            break  #

    ratings = driver.find_elements(By.CSS_SELECTOR, 'div.Rating__RatingBody-sc-1rhvpxz-0')  # Get all rating data after clicking button
    
    for rating in ratings:
        try:                                              
            score = rating.find_element(By.CSS_SELECTOR, 'div.CardNumRating__CardNumRatingNumber-sc-17t4b9u-2').text[0]  # Assuming score is the first character
            comment = rating.find_element(By.CSS_SELECTOR, 'div.Comments__StyledComments-dzzyvm-0').text.strip()
            if score in files:
                files[score].write(comment + '\n')
        except Exception as e:
            print("***************************Error processing a rating*********************************:", e)
    print(len(ratings), "ratings were written to txt file")
    print("---------------------Finished one professor -----------------------------")
    num_prof += 1
    driver.back()   # Navigate back in the browser
    

fd.close()

for file in files.values():
    file.close()

driver.quit()
