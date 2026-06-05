"""
TGDPS Ground Truth Rainfall Scraper (Cloud Version)
====================================================
Scrapes daily rainfall for 7 study stations from TGDPS Telangana portal.
Designed to run via GitHub Actions (cloud, no laptop needed).

Source: https://tgdps.telangana.gov.in/mandaldata.jsp?s1=23
Stations: Rangareddy district mandals
"""
import requests
from bs4 import BeautifulSoup
import csv, re, os
from datetime import datetime, date
from pathlib import Path


URL = "https://tgdps.telangana.gov.in/mandaldata.jsp?s1=23"
STATIONS = ["Chevella", "Hayathnagar", "Ibrahimpatnam",
            "Kondurg", "Maheshwaram", "Saroornagar", "Yacharam"]

# CSV path — works both locally and in GitHub Actions
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = SCRIPT_DIR / "groundtruth_june2026.csv"


def scrape():
    """Scrape Rangareddy mandal page and return (date, {station: rainfall_mm})."""
    r = requests.get(URL, verify=False, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    # Extract date from page
    date_match = re.search(r'(\d{2}/\d{2}/\d{4})', soup.get_text())
    data_date = datetime.strptime(date_match.group(1), "%d/%m/%Y").date() if date_match else date.today()

    # Parse mandal data — each entry: Sno, Name, TodayActual, TodayNormal, Dev%, CumulActual, CumulNormal, CumulDev%, Status
    today_data = {}
    for table in soup.find_all('table'):
        for row in table.find_all('tr'):
            cells = [c.get_text(strip=True) for c in row.find_all('td')]
            i = 0
            while i < len(cells) - 2:
                if cells[i].isdigit():
                    name = cells[i + 1]
                    try:
                        rainfall = float(cells[i + 2])
                    except ValueError:
                        i += 1
                        continue
                    if name in STATIONS:
                        today_data[name] = rainfall
                    # Skip to next entry (8-9 fields per mandal)
                    sno = int(cells[i])
                    i += 3
                    while i < len(cells):
                        if cells[i].isdigit():
                            try:
                                if int(cells[i]) > sno:
                                    break
                            except ValueError:
                                pass
                        if 'Average' in cells[i]:
                            i = len(cells)
                            break
                        i += 1
                else:
                    i += 1

    return data_date, today_data


def save_csv(data_date, today_data):
    """Append row to CSV. Skip if date exists."""
    date_str = data_date.strftime("%Y-%m-%d")
    header = ["Date"] + STATIONS

    # Read existing dates
    existing_dates = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, 'r') as f:
            for row in csv.DictReader(f):
                existing_dates.add(row.get("Date", ""))

    if date_str in existing_dates:
        print(f"[SKIP] {date_str} already exists in CSV")
        return False

    # Create header if new file
    if not CSV_PATH.exists():
        with open(CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(header)

    # Append today's row
    row = [date_str] + [today_data.get(s, "") for s in STATIONS]
    with open(CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow(row)
    return True


def main():
    import warnings, urllib3
    urllib3.disable_warnings()
    warnings.filterwarnings("ignore")

    print(f"TGDPS Scraper — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    data_date, today_data = scrape()
    print(f"Data date: {data_date}")

    print(f"\n{'Station':18s} | {'Rainfall (mm)':>13s}")
    print(f"{'─'*18}-+-{'─'*13}")
    for s in STATIONS:
        val = today_data.get(s, "N/A")
        print(f"{s:18s} | {val:>13}")

    found = sum(1 for s in STATIONS if s in today_data)
    print(f"\nFound: {found}/{len(STATIONS)} stations")

    if found > 0:
        saved = save_csv(data_date, today_data)
        if saved:
            print(f"\n✓ Appended {data_date} to {CSV_PATH.name}")

            # Show full CSV content
            print(f"\nFull CSV ({CSV_PATH.name}):")
            with open(CSV_PATH, 'r') as f:
                print(f.read())
        else:
            print(f"\n→ {data_date} was already in CSV")
    else:
        print("\n✗ No data found — website may be down or format changed")
        exit(1)


if __name__ == "__main__":
    main()
