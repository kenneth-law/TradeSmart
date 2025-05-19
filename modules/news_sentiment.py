"""
News Sentiment Module

This module contains functions for retrieving and analyzing news sentiment for stocks.
"""

import os
import re
import requests
import openai
from bs4 import BeautifulSoup
from modules.utils import log_message

def get_news_sentiment_with_timeframes(ticker_symbol, company_name):
    """
    Analyzes the sentiment of market news headlines for a given company over various
    timeframes using search results obtained from Google News. The sentiment is derived from
    the headlines sourced for the company and adjusted based on their respective timeframes.
    The function also cleans the company name to standardize its representation and accounts
    for ASX-specific stock ticker formatting.

    The sentiment is provided as a numerical score and corresponding label, with an
    additional textual comment summarizing the findings.

    Parameters:
        ticker_symbol (str): The stock ticker symbol for the company being analyzed. Must
            include '.AX' as a suffix for ASX stocks if applicable.
        company_name (str): The full name of the company to analyze, including potential
            stock designations like "LTD" or "GROUP".

    Returns:
        tuple: A five-element tuple consisting of:
            - sentiment_score (float): The computed sentiment score, ranging from -1.0
              (very negative) to 1.0 (very positive). A score of 0 is neutral.
            - sentiment_label (str): A descriptive label corresponding to the sentiment score
              (e.g., "Neutral", "Positive").
            - sentiment_comment (str): A brief textual explanation summarizing the sentiment
              and its implications based on the analyzed headlines.
            - raw_sentiment_score (float): The unweighted sentiment score before timeframe
              adjustments are applied.
            - headlines (list): A list of headlines that were found and analyzed.

    Raises:
        Exception: An error may be raised for issues occurring during network requests,
        HTML parsing, or the API call for sentiment analysis. Such instances will be logged
        as warnings without halting the process.

    """

    return 0, "Neutral", "DEBUG SKIPPED", 0, []


    print(f"Starting news analysis for {company_name} ({ticker_symbol})")

    # Clean the company name by removing stock designations like "FPO", "[TICKER]", etc.
    clean_company_name = company_name

    # Remove "FPO" designation
    clean_company_name = clean_company_name.replace(" FPO", "")

    # Remove ticker in brackets like "[BHP]"
    clean_company_name = re.sub(r'\s*\[[A-Z]+\]', '', clean_company_name)

    # Remove other common stock designations
    common_designations = [" LIMITED", " LTD", " GROUP", " CORPORATION", " CORP", " INC", " INCORPORATED", " PLC", " N.V.", " S.A."]
    for designation in common_designations:
        if designation in clean_company_name.upper():
            clean_company_name = clean_company_name.replace(designation, "")
            clean_company_name = clean_company_name.replace(designation.lower(), "")
            clean_company_name = clean_company_name.replace(designation.title(), "")

    # Trim whitespace
    clean_company_name = clean_company_name.strip()

    print(f"Cleaned company name: '{clean_company_name}' (original: '{company_name}')")

    # Special handling for ASX stocks
    is_asx_stock = ticker_symbol.endswith('.AX')

    # Define timeframes to search, from most recent to oldest
    timeframes = [
        {"name": "Past 24 hours", "query_param": "qdr:d", "weight": 1.0},
        {"name": "Past week", "query_param": "qdr:w", "weight": 0.85},
        {"name": "Past month", "query_param": "qdr:m", "weight": 0.7},
    ]

    all_headlines = []
    weights = []
    timeframe_used = None

    # Generate search terms - multiple options for ASX stocks
    search_terms = []
    if is_asx_stock:
        base_ticker = ticker_symbol.replace('.AX', '')
        search_terms = [
            f"{clean_company_name} stock",  # Company name + stock
            f"{clean_company_name} ASX",  # Company name + ASX
            f"{base_ticker} ASX",  # Ticker + ASX
        ]
        print(f"Using ASX-specific search terms: {search_terms}")
    else:
        search_terms = [f"{clean_company_name} stock news"]

    # Try each timeframe until we find enough headlines
    for timeframe in timeframes:
        if len(all_headlines) >= 3:
            break

        for search_term in search_terms:
            if len(all_headlines) >= 3:
                break

            try:
                print(f"Searching for '{search_term}' in {timeframe['name']}")
                news_url = f"https://www.google.com/search?q={search_term}&tbm=nws&source=lnt&tbs={timeframe['query_param']}"
                print(f"Search URL: {news_url}")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/'
                }

                response = requests.get(news_url, headers=headers)

                if response.status_code != 200:
                    print(f"Failed to get response: {response.status_code}")
                    continue

                # Use BeautifulSoup to parse the HTML
                soup = BeautifulSoup(response.content, 'html.parser')

                # Try multiple selector patterns that Google might use
                headline_selectors = [
                    {"tag": "div", "attrs": {"class": "BNeawe vvjwJb AP7Wnd"}},
                    {"tag": "h3", "attrs": {"class": "LC20lb MBeuO DKV0Md"}},
                    {"tag": "div", "attrs": {"role": "heading"}},
                    {"tag": "div", "attrs": {"class": "mCBkyc y355M ynAwRc MBeuO jBgGLd OSrXXb"}},
                    {"tag": "div", "attrs": {"class": "n0jPhd ynAwRc MBeuO nDgy9d"}},
                    {"tag": "a", "attrs": {"class": "WlydOe"}},
                    {"tag": "h3", "attrs": {}},
                    {"tag": "a", "attrs": {"class": "DY5T1d RZIKme"}}
                ]

                headlines_found = []

                # Try each selector until we find some headlines
                for selector in headline_selectors:
                    tag_name = selector["tag"]
                    attrs = selector["attrs"]

                    found_elements = soup.find_all(tag_name, attrs)

                    for element in found_elements:
                        text = element.get_text(strip=True)
                        if text and len(text) > 15:  # Filter out very short texts
                            if text not in ["Customised date range", "All results", "News", "Images", 
                                           "Videos", "Maps", "Shopping", "More", "Search tools", "Any time"]:
                                headlines_found.append(text)

                # If we found headlines, add them to our list with the appropriate weight
                if headlines_found:
                    print(f"Found {len(headlines_found)} headlines with '{search_term}'")
                    # Get up to 5 headlines from this timeframe
                    for headline in headlines_found[:5]:
                        all_headlines.append(headline)
                        weights.append(timeframe["weight"])

                    timeframe_used = timeframe["name"]
                    print(f"Headlines: {headlines_found[:5]}")

                    # If we found at least 3 headlines, we can stop searching
                    if len(all_headlines) >= 3:
                        break

            except Exception as e:
                print(f"Error searching {search_term} in {timeframe['name']}: {str(e)}")
                continue

    # If we didn't find any headlines, return neutral sentiment
    if not all_headlines:
        print("No headlines found for any search term or timeframe")
        return 0, "Neutral", "No headlines found across any timeframe", 0, []

    # Combine the headlines into a single string, but note which timeframe they came from
    combined_headlines = "\n".join(all_headlines)

    # Calculate the average weight to apply to the sentiment
    avg_weight = sum(weights) / len(weights) if weights else 1.0

    print(f"Found {len(all_headlines)} headlines in total")

    # Now, if an OpenAI API key is available, analyse the sentiment using these headlines
    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        prompt = f"""
        Analyze the market sentiment for {company_name} ({ticker_symbol}) based on these news headlines:

        {combined_headlines}

        Note: These headlines are from {timeframe_used}.

        Rate the sentiment on a scale from -1.0 (very negative) to 1.0 (very positive).
        IF YOU DETERMINE THAT THE HEADLINES ARE NOT RELATED TO THE COMPANY WE ARE ANALYSING, FOR EXAMPLE:
        "Headlines: ['The 10 most shorted ASX stocks plus the biggest risers and fallers, Week 19', '2 Growth Stocks with All-Star Potential and 1 to Steer Clear Of', 'Kevin Hart Becomes the Laughingstock After Photo Next to Steph Curry', 'Main Vietnam stock exchange launches long-awaited trading system', 'The 10 most shorted ASX stocks plus the biggest risers and fallers Week 19']"
        When you are analysing SUNCORP stock, then logically, you should exclude "Kevin Hart Becomes the Laughingstock After Photo Next to Steph Curry" and consider the news not specific to the company as as vietnam trading system as noise and >0.1 weight.
        If you think there are no relevent infomation, simply return 0
        Also provide a summary label (Very Negative, Negative, Slightly Negative, Neutral, Slightly Positive, Positive, Very Positive).
        In addition, add a short comment on what you think based on the headline, ~50-100 words
        Format your response as exactly three lines:
        [sentiment_score]
        [sentiment_label]
        [sentiment_comment]
        """

        print("Sending request to OpenAI for sentiment analysis")

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.2,
            )

            result = response.choices[0].message.content.strip().split('\n')

            # Extract sentiment score and label
            if len(result) >= 3:
                try:
                    raw_sentiment_score = float(result[0].strip())
                    sentiment_label = result[1].strip()
                    sentiment_comment = str(result[2].strip())

                    # Apply the weight to the sentiment score based on recency
                    weighted_sentiment_score = raw_sentiment_score * avg_weight

                    print(f"Raw sentiment score: {raw_sentiment_score}")
                    print(f"Weighted sentiment score: {weighted_sentiment_score}")
                    print(f"Sentiment label: {sentiment_label}")
                    print(f"Timeframe: {timeframe_used}, Weight: {avg_weight}")
                    print(f"the comment by ai is {sentiment_comment}")

                    sentiment_label = sentiment_label + " - " + sentiment_comment

                    return weighted_sentiment_score, sentiment_label, f"Based on news from {timeframe_used}", raw_sentiment_score, all_headlines
                except Exception as e:
                    print(f"Failed to parse sentiment result: {str(e)}")
                    return 0, "Neutral", f"Failed to parse sentiment: {str(e)}", 0, []
            else:
                print("Incomplete sentiment analysis response")
                return 0, "Neutral", "Incomplete sentiment analysis", 0, []
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return 0, "Neutral", f"Error with sentiment analysis: {str(e)}", 0, []
    else:
        print("OpenAI API key not set - returning neutral sentiment")
        return 0, "Neutral", "OpenAI API key not set", 0, []
