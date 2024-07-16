import time
from datetime import datetime
import requests
import polars as pl
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


# VARS
email_string = "tman_irishfan@yahoo.com"
ccnum = "5524331656090227"


chrome_options  = webdriver.ChromeOptions()
chrome_options .add_argument('--ignore-certificate-errors')
chrome_options .add_argument('--ignore-ssl-errors')

start_time = time.time()
driver = webdriver.Chrome(options=chrome_options)
driver.get('https://mshealthandfit.com/vote/verify/2024/julia-poppenberg')

# Wait for the input field to be present
try:
    email_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "purchase-email"))
    )
    card_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "purchase-card-number"))
    )
    name_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "purchase-card-name"))
    )
    # Input the email
    email_input.send_keys(email_string)  # Replace with the desired email
    # Input the credit card number
    card_input.send_keys(ccnum)  # Replace with the desired credit card number


    time.sleep(20)
    print("Credit card number input successful")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Always quit the driver
    driver.quit()

