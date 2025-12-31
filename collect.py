import time
import json
import os
import signal
import sys
import random
import traceback
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database
from selenium.common.exceptions import TimeoutException
from sqlalchemy.sql import func 


WEBSITES = [
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://www.prothomalo.com",              # Most visited Bengali news site
    # "https://www.chaldal.com",                 # Online grocery
    # "https://www.dhakatribune.com",            # English news portal
]


TRACES_PER_SITE = 1000
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def setup_webdriver():
    """Set up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(os.path.join(os.path.dirname(ChromeDriverManager().install()), "chromedriver"))
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()

    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""

def collect_single_trace(driver, wait, website_url):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, etc.)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return success or failure status
    """
    try:
        driver.get(FINGERPRINTING_URL)
        wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Collect trace')]")))
        collect_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Collect trace')]")
        collect_button.click()

        try:
            wait.until(EC.text_to_be_present_in_element(
                (By.XPATH, "//div[@role='alert']"), "Collecting trace data..."
            ))
        except TimeoutException:
            status_element = driver.find_element(By.XPATH, "//div[@role='alert']")
            actual_status = status_element.text
            print(f"  - Expected 'Collecting trace data...', but found: '{actual_status}'")
            raise

        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        driver.get(website_url)

        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        for _ in range(random.randint(7, 10)):
            scroll_distance = random.randint(-1500, 1500)
            driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
            time.sleep(random.uniform(0.5, 1.5))

        driver.close()
        driver.switch_to.window(driver.window_handles[0])

        wait.until(EC.text_to_be_present_in_element(
            (By.XPATH, "//div[@role='alert']"), "Trace data collection complete!"
        ))

        print(f"  - Successfully collected trace for {website_url}")
        return True

    except Exception as e:
        print(f"  - Error collecting trace for {website_url}: {str(e)}")
        traceback.print_exc()
        if len(driver.window_handles) > 1:
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
        return False
        # raise  # Re-raise to trigger power loss simulation in collect_fingerprints
    
def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """
    wait = WebDriverWait(driver, 10)
    total_new_traces = 0

    # Calculate remaining traces needed for each website
    current_counts = database.db.get_traces_collected()
    if target_counts is None:
        target_counts = {website: TRACES_PER_SITE for website in WEBSITES}

    remaining_counts = {website: max(0, target_counts.get(website, TRACES_PER_SITE) - current_counts.get(website, 0))
                       for website in WEBSITES}

    print("Remaining traces to collect:", remaining_counts)
    
    try:
        # Iterate over each website
        for website in WEBSITES:
            if remaining_counts[website] <= 0:
                print(f"Skipping {website}: Target number of traces already collected.")
                continue

            print(f"\nCollecting traces for {website} ({TRACES_PER_SITE - remaining_counts[website] + 1}/{TRACES_PER_SITE})")
            
            # Collect all required traces for this website
            while remaining_counts[website] > 0:
                print(f"  Collecting trace {TRACES_PER_SITE - remaining_counts[website] + 1}/{TRACES_PER_SITE}")
                success = collect_single_trace(driver, wait, website)
                if success:
                    remaining_counts[website] -= 1
                else:
                    print(f"  - Retrying trace collection for {website}")

                time.sleep(1)  # Brief pause to avoid overwhelming the server

            # Retrieve and save traces for this website
            
            traces = retrieve_traces_from_backend(driver)
            if not traces:
                print("  - Failed to retrieve traces from backend")
                print(f"  Clearing traces for {website} from backend...")
                clear_trace_results(driver, wait)
                continue
            

            # Process traces if retrieved, otherwise save placeholders
            collected_traces = []
            for i, trace in enumerate(traces):
                website_index = WEBSITES.index(website)
                collected_traces.append({
                    'website': website,
                    'website_index': website_index,
                    'trace_data': trace
                })
                total_new_traces += 1
            
            # Save all traces for this website in a batch
            print(f"  Saving {len(collected_traces)} traces for {website} to the database...")
            session = database.db.Session()
            try:
                for trace in collected_traces:
                    database.db.save_trace(
                        trace['website'],
                        trace['website_index'],
                        trace['trace_data']
                    )
                session.commit()
                print(f"  All traces for {website} saved successfully.")
            except Exception as e:
                session.rollback()
                print(f"  Error saving traces for {website} to database: {str(e)}")
                raise
            finally:
                session.close()

            # Clear traces from the backend
            print(f"  Clearing traces for {website} from backend...")
            clear_trace_results(driver, wait)

    except Exception as e:
        print(f"(exception occurred in fingerprint collection): {str(e)}")
        traceback.print_exc()
        raise

    return total_new_traces



def main():
    # Step 1: Check if the Flask server is running
    if not is_server_running():
        print("Flask server is not running. Please start the server at http://localhost:5000")
        sys.exit(1)

    # Step 2: Initialize the database
    print("Initializing database...")
    database.db = Database(WEBSITES)
    database.db.init_database()  # Create the tables

    # Step 3: Set up the WebDriver
    print("Setting up WebDriver...")
    driver = None
    try:
        driver = setup_webdriver()
        wait = WebDriverWait(driver, 10)

        # Step 4: Start the collection process
        max_attempts = 3
        attempt = 1
        while not is_collection_complete() and attempt <= max_attempts:
            print(f"\nCollection attempt {attempt}/{max_attempts}")
            try:
                new_traces = collect_fingerprints(driver)
                print(f"Collected {new_traces} new traces in this attempt")
            except Exception as e:
                print(f"Error during collection attempt {attempt}: {str(e)}")
                traceback.print_exc()
            attempt += 1

        # Step 5: Check if collection is complete
        if is_collection_complete():
            print(f"\nCollection complete! Collected {TRACES_PER_SITE} traces for each website.")
        else:
            print(f"\nCollection incomplete after {max_attempts} attempts. Check logs for errors.")

    except Exception as e:
        print(f"Fatal error during collection: {str(e)}")
        traceback.print_exc()
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
        print("Exporting data to JSON...")
        try:
            database.db.export_to_json(OUTPUT_PATH)
            print(f"Data exported to {OUTPUT_PATH}")
        except Exception as e:
            print(f"Error exporting data to JSON: {str(e)}")
        print("WebDriver closed.")


if __name__ == "__main__":
    main()
    
