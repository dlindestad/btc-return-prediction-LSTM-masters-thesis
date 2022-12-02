import pandas as pd
import snscrape.modules.twitter as sntwitter
import threading
from time import perf_counter, sleep
from datetime import datetime
import os

START_DATE = "2012-01-01"
END_DATE = "2022-09-01"

# Scrapes twitter for tweets with searchstrings ["btc", "#BTC", "ethereum", "eth", "bitcoin"]
# WARNING - running this program in full at default thread limit will likely take over 120 hours.
# If threads crash with error 403 this is caused by twitter, reduce thread_limit and restart,
# the program will resume.


def monthsplit(searchstring):
    searchlist = []
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    for idx, _ in enumerate(dates):
        if idx != len(dates) - 1:
            searchlist.append(
                f"{searchstring} since:{dates[idx].date()} until:{dates[idx+1].date()}"
            )
    return searchlist


def tweetscrape(searchstring):
    """
    Scrapes twitter, passing searchstring to snscrape
    """
    tweets = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(searchstring).get_items()):
        if i == 0:
            ts = datetime.now().strftime("[%H:%M:%S]")
            print(f'{ts}: Started search, searchstring: "{searchstring}".')
        if i % 10000 == 0:
            ts = datetime.now().strftime("[%H:%M:%S]")
            print(
                f'{ts}: Current count for "{searchstring}" Results: {human_format(i)}.'
            )
        tweets.append(
            [
                tweet.date,
                tweet.id,
                tweet.content,
                tweet.user.username,
                tweet.replyCount,
                tweet.retweetCount,
                tweet.likeCount,
                tweet.quoteCount,
            ]
        )
    return tweets


def save_dataframe(tweets, searchstring):
    """
    Saves pandas dataframe to .csv file, naming according to searchstring
    """
    if ":" in searchstring:
        searchstring = searchstring.replace(":", "_")
    if '"' in searchstring:
        searchstring = searchstring.replace('"', "")
        searchstring = searchstring + "QUOTESEARCH"
    dirname = os.getcwd()
    filename = dirname + "\Data\Tweets\{}.csv".format(searchstring)
    tweets_df = pd.DataFrame(
        tweets,
        columns=[
            "Datetime",
            "Tweet Id",
            "Text",
            "Username",
            "Replies",
            "Retweets",
            "Likes",
            "Quotes",
        ],
    )
    tweets_df.to_csv(filename)
    ts = datetime.now().strftime("[%H:%M:%S]")
    print(f"\n\n{ts}: SAVED SEARCH! Searchstring: {searchstring}. FILEPATH: {filename}")
    return


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def run_search(searchstring):
    """
    Performs the actual search
    """
    result = tweetscrape(searchstring)
    save_dataframe(result, searchstring)
    ts = datetime.now().strftime("[%H:%M:%S]")
    print(f'\n\n{ts}: Scrape completed for "{searchstring}"')
    return


def find_missing_searchresults(search):
    """
    Finds all monthly searchresults for a given search for which a .csv files does not yet exist, returns list of missing searchterms
    """
    full_list_unclean = monthsplit(search)
    full_list = []
    for unclean_term in full_list_unclean:
        if ":" in unclean_term:
            full_list.append(unclean_term.replace(":", "_"))
        else:
            full_list.append(unclean_term)

    dirname = os.getcwd()
    dirname = dirname + "\Data\Tweets"
    result_list = []
    file_list = os.listdir(dirname)
    for searchterm in full_list:
        if not any([s.startswith(searchterm) for s in file_list]):
            result_list.append(searchterm)
    return result_list


def get_search_terms(searchlist):
    """
    Adds back ":" at appropriate spot for correct search terms
    """
    search_terms = []
    for item in searchlist:
        search_terms.append(item.replace("_", ":"))
    return search_terms


def main():
    thread_limit = 25
    start_time = perf_counter()
    searchstrings = ["btc", "#BTC", "ethereum", "eth", "bitcoin"]
    filenames = []
    for search in searchstrings:
        filenames_temp = find_missing_searchresults(search)
        filenames.append([get_search_terms(filenames_temp), search])

    inactive_thread_list = []
    active_thread_list = []
    max_search_length = len(monthsplit(searchstrings[0]))
    for searchlist, searchterm in filenames:
        for search in searchlist:
            inactive_thread_list.append(
                threading.Thread(target=run_search, args=[search])
            )
        ts = datetime.now().strftime("[%H:%M:%S]")
        print(
            f"{ts}: Search for {searchterm}. Created {len(searchlist)} threads. Progress: {int(max_search_length - len(searchlist))}/{max_search_length}."
        )

    thread_queue = len(inactive_thread_list)
    while thread_queue > 0:
        if (
            len(threading.enumerate()) < thread_limit + 1
        ):  # Main thread counts as a thread in threading.enumerate()
            active_thread_list.append(inactive_thread_list[0])
            inactive_thread_list.pop(0)
            active_thread_list[-1].daemon = True
            active_thread_list[-1].start()
            thread_queue -= 1
        else:
            sleep(10)

    for thread in active_thread_list:
        thread.join()

    end_time = perf_counter()
    ts = datetime.now().strftime("[%H:%M:%S]")
    print(f"{ts}: Scrape took {end_time- start_time: 0.2f} second(s) to complete.")
    return


if __name__ == "__main__":
    main()
