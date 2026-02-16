from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time


def main() -> None:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        print("Visiting app...")
        url = "https://aryanasignment-sayjr7rvushwkwhsewnoxx.streamlit.app"
        for attempt in range(3):
            try:
                driver.get(url)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"Load failed ({attempt + 1}/3): {exc}")
                time.sleep(5)

        time.sleep(5)
        print(f"Page Title: {driver.title}")

        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                if "Yes, get this app back up!" in btn.text:
                    print("Found Wake Button. Clicking...")
                    btn.click()
                    time.sleep(10)
                    print("Clicked!")
                    break
        except Exception as exc:
            print(f"Button check skipped: {exc}")
    finally:
        driver.quit()
        print("Done.")


if __name__ == "__main__":
    main()
