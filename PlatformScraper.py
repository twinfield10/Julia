import time
from datetime import datetime
import requests
import polars as pl
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options  = webdriver.ChromeOptions()
chrome_options .add_argument('--ignore-certificate-errors')
chrome_options .add_argument('--ignore-ssl-errors')

url = 'https://platformalexandria.com/floorplans'
start_time = time.time()
driver = webdriver.Chrome(options=chrome_options)
driver.get(url)


wait = WebDriverWait(driver, 10)

links_list = []
for e in driver.find_elements(By.CSS_SELECTOR, 'div[class*="units_grid"]'):
    link = e.get_attribute('href')
    if link:
        links_list.append(f"{url}/{link}")

dfs = []
for i in links_list:
    driver.get(i)
    print(i)

    # Locate the table element using CSS selector
    table_element = driver.find_element(By.CSS_SELECTOR, 'div[class*="units_table"]')

    # Get the inner HTML of the table element
    table_html = table_element.get_attribute('outerHTML')

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(table_html, 'html.parser')

    # Find the table
    time.sleep(5)
    table = soup.select_one('table')
    #print(table)

    # Extract headers
    #headers = []
    #for th in table.find('thead').find_all('th'):
    #    print(th.get_text(strip=True))
    #    headers.append(th.get_text(strip=True))
    headers = ['Unit', 'Size', 'Price', 'Finish', 'Available', 'Link']
    # Extract rows
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = tr.find_all('td')
        row = [cell.get_text(strip=True) for cell in cells]
        print(row)
        rows.append(row)

    # Create DataFrame
    df = pl.DataFrame(rows, schema=headers)
    dfs.append(df)

final = pl.concat(dfs)
print(final.head())

#df = pl.concat(df_list)
#print(df.head())


end_time = time.time()
elap_time = round((end_time - start_time)/60, 2)
print(f"Elapsed Time: {elap_time} Minutes")
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))