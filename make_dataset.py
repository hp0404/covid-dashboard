import os
import requests
import logging
import numpy as np
import pandas as pd
from datetime import datetime


URL = "https://raw.githubusercontent.com/VasiaPiven/covid19_ua/master/covid19_by_area_type_hosp_dynamics.csv"
API = "https://api.github.com/repos/VasiaPiven/covid19_ua/branches/master"
TOKEN = os.environ.get("TOKEN", "")
TODAY = datetime.today().strftime("%Y-%m-%d")
CURRENT_LOCATION = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_LOCATION = os.path.join(CURRENT_LOCATION, "data", "outputs")

LOG = os.path.join(OUTPUTS_LOCATION, "make_dataset.log")
DATASET = os.path.join(OUTPUTS_LOCATION,  f"monitoring_v5_{TODAY}.csv")


def make_dataset():
    """ Додає АР Крим до таблиці. """
    data = pd.read_csv(URL)
    unique_dates = data["zvit_date"].unique()
    crimea = pd.DataFrame(
        {
            "zvit_date": unique_dates,
            "registration_area": ["Автономна Республіка Крим"],
        },
        index=np.arange(len(unique_dates))
    )
    df = data.append(crimea)
    df.to_csv(DATASET, sep=";", index=False)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(LOG),
            logging.StreamHandler()
        ]
    )

    r = requests.get(API, headers={"Authorization": f"Bearer {TOKEN}"}).json()
    latest_commit = r["commit"]["commit"]["committer"]["date"]
    commit_date = datetime.strptime(latest_commit, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

    if not os.path.isfile(DATASET) and commit_date == TODAY:
        make_dataset()
        logging.info("Скрипт виконався успішно")
    else:
        logging.info("Скрипт не виконувався")
