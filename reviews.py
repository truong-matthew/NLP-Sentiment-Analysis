from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


# Headers for BeautifulSoup4 (just in case)
headers = {'user-agent': 'Chrome/108.0.5355.0'}

# Set options to run Selenium 'Headless'
options = Options()
#options.headless = True
#options.add_argument("--window-size=1920,1200")

# Set up driver path
DRIVER_PATH = '/usr/local/bin/chromedriver'
driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)

# List of Details
review = []
stars = []


# Page number (104 in total)
page_num = 1

# Get title for Selenium
time.sleep(5)

while page_num != 3:
    driver.get('https://ca.trustpilot.com/review/www.super.com?page=' + str(page_num))
    # Get source code for beautiful soup
    soup = BeautifulSoup(driver.page_source, 'html.parser') #
    time.sleep(2)

    # Find all product links
    prod_titles = soup.find_all('h2', {'class': 'typography_heading-s__f7029 typography_appearance-default__AAY17'})
    prod_captions = soup.find_all('p', {'class': 'typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn'})
    #prod_stars = soup.find_all('div', {'class': 'star-rating_starRating__4rrcf star-rating_medium__iN6Ty'})
    divs = soup.find_all('div', {"class": "star-rating_starRating__4rrcf star-rating_medium__iN6Ty"})
    #print(div.find('img').attrs['src'])

    collection = soup.findAll("img")
    poss_ratings = ['Rated 1 out of 5 stars', 'Rated 2 out of 5 stars', 'Rated 3 out of 5 stars', \
                    'Rated 4 out of 5 stars', 'Rated 5 out of 5 stars']
    '''
    stars_1 = soup.find_all('img', {'alt': 'Rated 1 out of 5 stars'})
    stars_2 = soup.find_all('img', {'alt': 'Rated 2 out of 5 stars'})
    stars_3 = soup.find_all('img', {'alt': 'Rated 3 out of 5 stars'})
    stars_4 = soup.find_all('img', {'alt': 'Rated 4 out of 5 stars'})
    stars_5 = soup.find_all('img', {'alt': 'Rated 5 out of 5 stars'})
    '''
    # Iterate through list of links to pull title, SKU and link
    for i in range(len(prod_titles)):
        title_string = prod_titles[i].text
        review.append(title_string)

    for i in range(len(divs)):
        target_div = divs[i]
        div_star = target_div.find('img').attrs['alt']
        if div_star in poss_ratings:
            stars.append(div_star)

#    for img in collection:
#        if 'alt' in img.attrs:#and img.attrs['alt'] in poss_ratings:
#            stars.append(img.attrs['alt'])
    else:
        print("Checked Page: " + str(page_num))
        page_num += 1


# Format data
print("# of reviews =", len(review))
print("# of stars =", len(stars))
scraped_data = pd.DataFrame([review, stars]).transpose()
scraped_data.columns = ["Reviews", "Stars"]
print(scraped_data.head())

# Save data into csv
scraped_data.to_csv("supershop_reviews.csv", index=False)

'''Issue is some reviews only have a title so we pull the star rating but no description so it messes up the order
as basically the code runs in two loops (inefficient tbh)'''
from wordcloud import WordCloud
all_neg = " ".join(review)
plt.figure(figsize =(20,20))
plt.imshow(WordCloud().generate(all_neg))
plt.show()