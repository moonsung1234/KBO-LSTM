
from bs4 import BeautifulSoup
import numpy as np
import requests

url = "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx"
new_url = "https://www.koreabaseball.com"
new_url_list = []

data = np.array([])

res = requests.get(url)

if res.status_code == 200 :
    html = res.text
    soup = BeautifulSoup(html, "html.parser")
    
    tr = soup.select("#cphContents_cphContents_cphContents_udpContent > div.record_result > table > tbody > tr")

    for e in tr :
        href = e.select_one("td:nth-child(2) > a")["href"]

        new_url_list.append(new_url + href.replace("Basic", "Total"))

for new_url in new_url_list :
    res = requests.get(new_url)

    id = new_url[-5:]

    if res.status_code == 200 :
        html = res.text
        soup = BeautifulSoup(html, "html.parser")

        tr = soup.select("#contents > div.sub-content > div.player_records > div > table > tbody > tr")

        for e in tr :
            td = e.select("td")[2:]
            td = list(map(lambda x : x.get_text().replace("-", "0"), td))

            td.insert(0, id)

            td = np.round(np.array(td, dtype=np.float).reshape(1, -1), 3)
            data = td if len(data) == 0 else np.concatenate((data, td), axis=0)

            print(td)

np.savetxt("./data.csv", data, delimiter=",")