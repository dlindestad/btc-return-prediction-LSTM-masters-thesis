import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from statistics import mean, stdev
import matplotlib.pyplot as plt

START_DATE = "2012-01-01"
END_DATE = "2022-09-01"


def clean_tweet(tweet):
    """
    Cleans tweet text
    """
    tweet = re.sub("RT", "", tweet)  # remove 'RT'
    tweet = re.sub("#[A-Za-z0-9]+", "", tweet)  # remove '#'
    tweet = re.sub("\\n", "", tweet)  # remove '\n'
    tweet = re.sub("https?:\/\/\S+", "", tweet)  # remove hyperlinks
    tweet = re.sub("@[\S]*", "", tweet)  # remove @mentions
    tweet = re.sub("^[\s]+|[\s]+$", "", tweet)  # remove leading and trailing whitespace
    return tweet


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


def fast_group_days(df):
    keys, values = df.sort_values("Date").values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    df2 = pd.DataFrame({"Date": ukeys, "Text": [list(a) for a in arrays]})
    return df2


def analyze_sentiment(sentences, ignore_neutral=True):
    """
    Analyzes sentiments
    """
    analyzer = SentimentIntensityAnalyzer()
    vslist = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        if ignore_neutral:
            if vs["compound"] != 0:
                vslist.append(vs["compound"])
        else:
            vslist.append(vs["compound"])
    return vslist


def main():
    searchstrings = ["btc", "#BTC", "ethereum", "eth", "bitcoin"]
    df = pd.DataFrame(
        columns=searchstrings, index=pd.date_range(start=START_DATE, end=END_DATE).date
    )
    tweetcount = df.copy()
    df_sd, df_in, df_in_sd = df.copy(), df.copy(), df.copy()
    bands_excl = np.zeros(201, dtype=np.int32)
    bands_incl = np.zeros(201, dtype=np.int32)
    for search in searchstrings:
        print(f'\n{ts()}: Analyzing sentiments for "{search}"')
        search_list = [search]
        for i, filename in enumerate(list_filelist(search_list)[0]):
            print(f"{ts()}: Progress - {i}/{len(list_filelist(search_list)[0])}")
            dft = pd.read_csv(f"Data/Tweets/{filename}.csv")

            dft = dft.drop(
                columns=[
                    "Unnamed: 0",
                    "Tweet Id",
                    "Username",
                    "Replies",
                    "Retweets",
                    "Likes",
                    "Quotes",
                ]
            )
            dft["Datetime"] = pd.to_datetime(dft["Datetime"]).dt.date
            dft = dft.rename(columns={"Datetime": "Date"})
            dft = fast_group_days(dft)
            for _, row in dft.iterrows():
                tweetcount.loc[row["Date"], search] = len(row["Text"])
                # without neutral sentiments
                s_in = analyze_sentiment(row["Text"], ignore_neutral=False)
                s_in_ar = np.array(s_in)
                for v in s_in:
                    bands_incl[round((v + 1) * 100)] += 1
                s = s_in_ar[s_in_ar != 0]
                for v in s:
                    bands_excl[round((v + 1) * 100)] += 1
                if len(s) == 0:
                    df.loc[row["Date"], search] = np.nan
                    df_sd.loc[row["Date"], search] = np.nan
                elif len(s) == 1:
                    df.loc[row["Date"], search] = mean(s)
                    df_sd.loc[row["Date"], search] = np.nan
                else:
                    df.loc[row["Date"], search] = mean(s)
                    df_sd.loc[row["Date"], search] = stdev(s)

                # with neutral sentiments
                if len(s_in) == 0:
                    df_in.loc[row["Date"], search] = np.nan
                    df_in_sd.loc[row["Date"], search] = np.nan
                elif len(s_in) == 1:
                    df_in.loc[row["Date"], search] = mean(s_in)
                    df_in_sd.loc[row["Date"], search] = np.nan
                else:
                    df_in.loc[row["Date"], search] = mean(s_in)
                    df_in_sd.loc[row["Date"], search] = stdev(s_in)

    df.to_csv("Data/Sentiments/sentiments.csv")
    df_sd.to_csv("Data/Sentiments/sentiments_sd.csv")
    df_in.to_csv("Data/Sentiments/sentiments_incl_neutral.csv")
    df_in_sd.to_csv("Data/Sentiments/sentiments_incl_neutral_sd.csv")
    tweetcount.fillna(0)
    tweetcount.to_csv("Data/Tweets/tweetcount.csv")
    histogram_df = pd.DataFrame(
        bands_excl, columns=["bands_excl"], index=np.linspace(-1, 1, num=201)
    )
    histogram_df["bands_incl"] = bands_incl
    histogram_df.to_csv("Data/Sentiments/histogram_df.csv")
    print(f"\n{ts()}: Success, exiting.")
    return


if __name__ == "__main__":
    main()
