# myanimelist-data-collector
Scrape and collect anime/manga data using MAL API / Jikan API
And this is only for studying purpose, without using direct automation web broswer to illegally scrape data.
# Not for commercial or any other illegal purposes.
# Abuot the data set 

# Installation 
## For content based model
Install requirements: 
```bash
pip install pandas numpy scikit-learn streamlit
```
Train:
``` bash
python content_based_filtering/train.py
```
Deploy: 
```bash
streamlit run content_based_filtering/app.py
```
