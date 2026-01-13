# myanimelist-data-collector
Scrape and collect anime/manga data using MAL API / Jikan API
And this is only for studying purpose, without using direct automation web broswer to illegally scrape data.
# Not for commercial or any other illegal purposes.

---

# Manga Recommendation System 

Đồ án môn học: Nhập môn Khoa học Dữ liệu (Group 03)
Hệ thống tích hợp 2 bộ gợi ý: Content-based và RAG Chatbot.

## Cấu trúc Dự án

Dự án được chia làm 2 thư mục chính tương ứng với 2 phân hệ:

1.  **`content_base_filtering/`**: Hệ thống gợi ý Manga sử dụng TF-IDF & Weighted Soup.
2.  **`semantic_analysis/`**: Hệ thống gợi ý Manga sử dụng RAG, ChromaDB & Multilingual Model.

---

## Hướng dẫn Cài đặt & Chạy

Trước khi chạy, vui lòng cài đặt các thư viện cần thiết:

```bash
pip install streamlit pandas numpy scikit-learn chromadb sentence-transformers

Hướng dẫn Sử dụng

1. Chạy mô hình Content-Based

Chức năng: Gợi ý các bộ Anime tương tự dựa trên lịch sử các bộ phim người dùng đã xem và đánh giá.

Bước 1: Di chuyển vào thư mục:

cd content_base_filtering

Bước 2: Huấn luyện mô hình (Tạo file .pkl):

python train.py

(Hệ thống sẽ xử lý dữ liệu và lưu model vào thư mục models/)

Bước 3: Khởi chạy ứng dụng:

streamlit run app.py

--- 

2. Chạy mô hình RAG Chatbot
Chức năng: Tìm kiếm Manga bằng ngôn ngữ tự nhiên (Hỗ trợ Tiếng Việt tốt) thông qua Chatbot.
Bước 1: Di chuyển vào thư mục:

cd semantic_analysis

Bước 2: Chuẩn bị dữ liệu (Chỉ cần chạy lần đầu hoặc khi mất file):Chạy EDA_Analyse.ipynb (Run All) để làm sạch dữ liệu và tạo file processed_manga.pkl.Chạy Model_vn.ipynb (Run All) để tạo Vector Database hỗ trợ tiếng Việt (manga_chroma_db_vn).

Bước 3: Khởi chạy Chatbot:

streamlit run app.py

## Pipeline & Phương pháp Kỹ thuật

Đối với Content-based Filtering: 

Feature Engineering: Sử dụng kỹ thuật "Weighted Soup" - kết hợp các đặc trưng Genres (x3), Themes (x2), Demographics (x2), Score và Synopsis.
Vectorization: Sử dụng TF-IDF (n-gram=1,2; max_features=10,000).
Similarity: Tính toán ma trận Cosine Similarity để tìm độ tương đồng.



Đối với Semantic Analysis & RAG Data Collection: 
Scrape_and_clean_new.ipynb (Thu thập dữ liệu từ Jikan API v4).
Cleaning & EDA: EDA_Analyse.ipynb (Xử lý dữ liệu nhiễu, loại bỏ nội dung nhạy cảm, vẽ biểu đồ phân phối và Knowledge Graph).

Embedding Models:Khuyên dùng (Tiếng Việt): Chạy Model_vn.ipynb sử dụng model paraphrase-multilingual-mpnet-base-v2.
Bản nhẹ (Tiếng Anh): Model_eng.ipynb sử dụng model all-MiniLM-L6-v2.

Retrieval: Sử dụng ChromaDB để truy vấn vector và thuật toán Hybrid Ranking (kết hợp điểm ngữ nghĩa và độ nổi tiếng).

3. Thành viên thực hiện (Nhóm 03)

1. Sần Dịch Anh - 21120411: Crawl Data, Data Cleaning, đặt câu hỏi
2. Nguyễn Văn Hậu - 21120449: EDA, xây dựng model Content-based Filtering
3. Nguyễn Trung Dũng - 21120228 & EDA, xây dựng Model RAG + Semantic Search, ChromaDB
4. Ngô Gia Long - 20120525: Code Giao diện Streamlit, Slide
```
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

