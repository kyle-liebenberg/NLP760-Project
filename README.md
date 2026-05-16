# NLP760-Project
This project applies subword embeddings to authorship attribution, aiming to identify an author’s distinctive writing style by analysing subword patterns rather than whole words.

---

## Data Scraping 
isiZulu articles were scraped from Isolezwe (`https://isolezwe.co.za`) for the following 8 authors
- Mhlengi Shangase: `https://isolezwe.co.za/authors/journalist/`
- Zimbili Vilakazi: `https://isolezwe.co.za/authors/zee/`
- Sabelo Nsele: `https://isolezwe.co.za/authors/mangethe/`
- Simangaliso Ntshangase: `https://isolezwe.co.za/authors/health-and-lifestyle-journalist/`
- Fanelesibonge Bengu: `https://isolezwe.co.za/authors/fanele-bengu/`
- Nonkululeko Nhlapo: `https://isolezwe.co.za/authors/online/`
- Zwelakhe Ngcobo: `https://isolezwe.co.za/authors/mapholoba/` (only has 10 articles)
- Nokubongwa Phenyane: `https://isolezwe.co.za/authors/maphenyane/`

## To run the python script:
```bash
python3 -m venv venv
source venv/bin/activate
pip install datasets pandas
python3 build_author_dataset.py
```