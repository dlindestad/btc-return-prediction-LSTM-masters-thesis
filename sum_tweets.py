import pandas as pd
import numpy as np
from datetime import datetime

START_DATE = "2012-01-01"
END_DATE = "2022-09-01"


def monthsplit(searchstring):
    searchlist = []
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    for idx, _ in enumerate(dates):
        if idx != len(dates) - 1:
            searchlist.append(
                f"{searchstring} since:{dates[idx].date()} until:{dates[idx+1].date()}"
            )
    return searchlist


def list_filelist(searchstrings):
    filtered_searchstrings = []
    for search in searchstrings:
        t_list = []
        for file_name in monthsplit(search):
            if ":" in file_name:
                file_name = file_name.replace(":", "_")
            if '"' in file_name:
                file_name = file_name.replace('"', "")
                file_name = file_name + "QUOTESEARCH"
            t_list.append(file_name)
        filtered_searchstrings.append(t_list)
    return filtered_searchstrings


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def main():
    searchstrings = ["btc", "#BTC", "ethereum", "eth", "bitcoin"]
    final_count = {search: 0 for search in searchstrings}
    for search in searchstrings:
        count = 0
        print(f'\n{ts()}: Counting "{search}"')
        search_list = [search]
        for i, filename in enumerate(list_filelist(search_list)[0]):
            print(f"{ts()}: Progress - {i}/{len(list_filelist(search_list)[0])}")
            dft = pd.read_csv(f"Data/Tweets/{filename}.csv")
            final_count[search] += len(dft)
    final_count["sum"] = sum(final_count.values())
    print(f"\n{ts()}: Final tweet count:")
    print(final_count)
    return


if __name__ == "__main__":
    main()
