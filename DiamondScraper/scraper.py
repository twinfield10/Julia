# BrilliantEarthScrape.py

import os
import re
from time import perf_counter, sleep
import traceback
from io import StringIO
import json

import pandas as pd
from gazpacho import Soup
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options

# set timestamp
timestamp = f"{pd.Timestamp('today'):%Y-%m-%d %I-%M %p}"

# create project directories
os.makedirs('data', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

# define CSV path
cwd = os.getcwd()
csv_path = ''.join((cwd, '/data/', timestamp, '.csv'))


def create_driver():
    """Creates headless Chrome WebDriver instance."""
    chrome_options  = webdriver.ChromeOptions()
    chrome_options .add_argument('--ignore-certificate-errors')
    chrome_options .add_argument('--ignore-ssl-errors')
    return webdriver.Chrome(options=chrome_options)


driver = create_driver()


def take_screenshot():
    """Saves a screenshot of the current window in the 'screenshots' directory."""
    path = ''.join((
        './screenshots/', 'screenshot ', timestamp, '.png')
    )
    driver.save_screenshot(path)


def make_soup():
    """Makes soup on raw html to enable parsing."""
    html = driver.page_source
    return Soup(html)


def load_url(diamond_type: str):
    """Navigates to Brilliant Earth's diamonds search page."""
    base = 'https://www.brilliantearth.com/'
    natural_url = base + "/loose-diamonds/search/"
    lab_url = base + "lab-diamonds-search/"
    if diamond_type == 'natural':
        driver.get(natural_url)
    else:
        driver.get(lab_url)


def close_marketing_box():
    """Closes the marketing box when first loading the page."""
    # wait a maximum of 60 seconds to close the box
    try:
        WebDriverWait(driver, 15).until(
            ec.presence_of_element_located((By.CSS_SELECTOR, "button[name='bluecoreCloseButton']"))
        ).click()
    except:
        pass


def get_shapes():
    """Returns a list of available shapes."""
    soup = make_soup()
    a = soup.find('div', {'id': 'DiamondShape'})
    b = a.find('a')
    #return [shape.text.title() for shape in b if shape.text.title() in ['Elongated Cushion', 'Oval']]
    return [shape.text.title() for shape in b]


def select_shapes(ix: int):
    """Selects diamond shapes on the first pass."""
    if ix == 0:
        shapes = get_shapes()
        print(shapes)
        for shape in shapes:
            shape_element = f"//li[@data-shape='{shape}']//a"
            driver.find_element(By.XPATH, shape_element).click()
        sleep(3)
    else:
        pass

def show_list():
    list = driver.find_element(By.XPATH, "//div[@data-ga-text='Table View']")
    list.click()
    sleep(2)

def sort_page():
    dropdown = driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle')
    dropdown.click()
    price_high_to_low = driver.find_element(By.XPATH, "//a[@data-order-by='price' and @data-order-method='asc']")
    driver.execute_script("arguments[0].click();", price_high_to_low)
    sleep(3)


def perform_actions(element: str, box_input: str):
    """Takes actions on input box elements."""
    # find element
    e = driver.find_element(By.ID,  element)

    actions = ActionChains(driver)
    actions.move_to_element(e)
    actions.click()
    actions.send_keys(Keys.BACKSPACE * 10)
    actions.send_keys(box_input + Keys.RETURN)
    actions.perform()
    sleep(1)

    # click header
    header = driver.find_element(By.TAG_NAME, 'h1')
    header.click()


def set_max_price():
    """Re-adjusts the max price box in the results table."""
    perform_actions('max_price', '10000000')

def set_min_price():
    """Re-adjusts the max price box in the results table."""
    perform_actions('min_price', '1000')


def set_max_carat():
    """Re-adjusts the carat box in the results table."""
    perform_actions('max_carat', '50')

def set_min_carat():
    """Re-adjusts the carat box in the results table."""
    perform_actions('min_carat', '1')

def table_scroll_dynamic():
    """Scrolls down the diamond data table dynamically."""
    prev_n_items = 0
    scroll_by = 6800  # Increment in pixels

    while True:
        # Make soup and find items
        soup = make_soup()
        items = soup.find('div', {'class': 'inner item'})
        n_items = len(items)

        print(f"Current items: {n_items}, Previous items: {prev_n_items}")

        if n_items == prev_n_items:
            print("No new items loaded. Exiting scroll.")
            break

        prev_n_items = n_items
        driver.execute_script("document.querySelector('#diamond_search_wrapper').scrollTop += 1000")  # Replace 1000 with your desired scroll amount
        sleep(1)

    print(f"Total items loaded: {n_items}")


def table_scroll():
    """Scrolls down the diamond data table.
    The table loads a maximum of 200 items per position.
    """
    base_script = "document.querySelector('#diamond_search_wrapper').scrollTop += 1000"
    positions = ['6766', '20366', '33966',
                         '47566', '61166',
                         '74766', '88366'
                         ]
    prev_n_items = 0
    scroll_by = 1000

    for p in positions:
        # make soup & find items
        soup = make_soup()
        items = soup.find('div', {'class': 'inner item'})

        # check if 'items' is a list
        if isinstance(items, list):
            n_items = len(items)
            diff = n_items - prev_n_items

            print(n_items)
            print(prev_n_items)
            print(diff)

            # if 300 items loaded, track 'n_items' & scroll down to load more
            if diff == 300:
                prev_n_items = n_items
                scroll_by = ''.join((base_script, p))
                driver.execute_script(scroll_by)
                sleep(3)
            else:
                # if there are fewer than 300 items, all items have been loaded
                break
        # if 'items' is not a list (a single item), break
        else:
            break

def create_detail_dataframe():
    """Returns Data in JSON Format containing Diamond Details
       Converts to Pandas DataFrame"""
    
    # Keep Columns
    keep_cols = ['id', 'upc',
                 'shape', 'price', 'carat', 'cut', 'color', 'clarity',
                 'report', 'origin', 'polish', 'symmetry', 'measurements', 'fluorescence',
                 'has_cert', 'length', 'length_width_ratio', 'depth', 'table',
                 ]

    soup = make_soup()
    items = soup.find('div', {'class': 'inner item'})
    dfs = []

    for i in items:
        data_upc_str = i.attrs.get('data-upc')
        if data_upc_str:
            data_upc_dict = json.loads(data_upc_str)
            df = pd.json_normalize(data_upc_dict)
            #df['url'] = f"https://www.brilliantearth.com//loose-diamonds/view_detail/{df['id']}/"
            dfs.append(df[keep_cols])

    if dfs:
        final = pd.concat(dfs, ignore_index=True)
    else:
        print("No items found to process.")

    return final



def create_dataframe():
    """Returns pandas DataFrame from diamonds HTML page."""
    html = driver.page_source
    dfs = pd.read_html(StringIO(html))

    #print(dfs)

    # return the third table which contains target data
    return dfs[2]


def clean_table_df(df):
    """"Returns clean diamonds pandas DataFrame."""

    # Keep relevant columns
    keep_cols = ['id', 'upc',
                 'shape', 'price', 'carat', 'cut', 'color', 'clarity',
                 'report', 'origin', 'polish', 'symmetry', 'measurements', 'fluorescence',
                 'has_cert', 'length', 'length_width_ratio', 'depth', 'table',
                 ]
    df = df[keep_cols]
    
    # Filter Bad Values
    #df = df[df["price"] != "Need help finding the perfect diamond? try our diamond finder"]

    # remove '$' and commas, and convert float to int
    #df['price'] = df['price'].replace({'\\$': '', ',': ''}, regex=True)
    #df['price'] = df['price'].replace({'[a-zA-Z\s]': ''}, regex=True)
    #df['price'] = pd.to_numeric(df['price'], downcast='integer')

    return df


def get_url_list():
    """Returns list of html containing url sub-directories."""
    soup = make_soup()

    # find html with diamond url page and return it
    return soup.find('a', {'class': 'name td-n2'})


def create_url_df():
    """Returns DataFrame with diamond id and individual diamond urls."""

    url_list = get_url_list()
    url_dict = {}
    base = 'https://www.brilliantearth.com/'

    # extract url sub-directory & id and add to dict
    for ix, i in enumerate(url_list[:-1], start=1):
        href = i.attrs.get('href')

        # add diamond id and url to dict
        url_dict[ix] = {'url': base + href}

    # construct pandas DataFrame from url_dict and return it
    return pd.DataFrame.from_dict(url_dict, orient='index')


def merge_dfs(left_df, right_df):
    """Merges 'df' and 'url_df' and returns merged DataFrame."""
    return pd.merge(left_df, right_df, left_index=True, right_index=True)


def adjust_price(max_price: str):
    """Filters diamonds results based on price range."""
    perform_actions('min_price', max_price)
    sleep(3)


def final_cleaning(df, diamond_type):
    """Returns DataFrame - removes duplicates, adds 'type' & 'date_fetched' columns."""
    clean_df = df.copy()
    clean_df = clean_df.drop_duplicates()
    clean_df['type'] = diamond_type
    clean_df['date_fetched'] = timestamp
    clean_df['url'] = "https://www.brilliantearth.com//loose-diamonds/view_detail/"+ clean_df['id'].astype(str) +"/"
    return clean_df


def to_csv(df):
    """Writes a CSV file in the 'data' directory."""
    df = df.drop_duplicates()
    df.to_csv(csv_path, index=False)


def get_max_price(df):
    """Returns string of the max 'price' in the DataFrame."""
    return str(df['price'].max())


def get_last_id(df):
    """Returns the 'id' of the last row in the DataFrame."""
    return df['id'].iloc[-1]


def main():
    """Run script."""
    print('Attempting to scrape diamonds data. This could take a while...')
    tic = perf_counter()
    diamond_type = ['natural', 'lab']
    final_df = pd.DataFrame()

    try:
        for ix, dt in enumerate(diamond_type):
            # first scrape attempt
            load_url(dt)
            close_marketing_box()
            select_shapes(ix)
            set_max_carat()
            set_min_carat()
            set_min_price()
            set_max_price()
            show_list()
            sort_page()
            table_scroll_dynamic()

            # create and clean DataFrame to append to
            raw_df = create_detail_dataframe()
            table_df = raw_df #clean_table_df(raw_df)
            #url_df = create_url_df()
            df1 = table_df #merge_dfs(url_df, table_df)

            # get max price & id from the DataFrame to filter diamonds for next scrape
            prev_max_price = get_max_price(df1)
            prev_last_id = get_last_id(df1)

            # scrape remaining rows by iterating the price range
            while True:
                # scrape diamonds table
                adjust_price(prev_max_price)
                table_scroll_dynamic()

                # create and clean DataFrame, and append to 'df1' (created in first pass)
                raw_df = create_detail_dataframe()
                table_df = raw_df #clean_table_df(raw_df)
                #url_df = create_url_df()
                merged_df = table_df #merge_dfs(url_df, table_df)
                df1 = pd.concat([df1, merged_df])

                # set current max price & id using the last row scraped
                current_max_price = get_max_price(df1)
                current_last_id = get_last_id(df1)

                # check if price and id of the last row have been scraped
                if current_max_price != prev_max_price and \
                   current_last_id != prev_last_id:
                    prev_max_price = current_max_price
                    prev_last_id = current_last_id
                if current_max_price == prev_max_price:
                    prev_max_price = str(int(current_max_price) + 1)
                # else there are no new diamond results, export DataFrame
                else:
                    clean_df = final_cleaning(df1, dt)
                    final_df = pd.concat([final_df, clean_df])
                    break
        else:
            to_csv(final_df)
            print(f"CSV path: {csv_path}")
    except:
        traceback.print_exc()
        take_screenshot()
    finally:
        driver.quit()
        toc = perf_counter()
        duration = (toc - tic) / 60
        print(f"Finished in {duration:0.1f} minutes")


if __name__ == '__main__':
    main()
