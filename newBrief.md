# BRIEF OPERASIONAL BARU
## Pipeline Penelitian Analisis Sentimen Komentar TikTok terhadap Program Makan Bergizi Gratis Menggunakan IndoBERT dan BiLSTM

Dokumen ini adalah **brief operasional target** yang disusun agar **selaras dengan metodologi penelitian pada naskah skripsi**, terutama alur **split 70% data latih dan 30% data uji** yang kemudian bercabang menjadi **jalur training** dan **jalur testing**.

Dokumen ini **bukan sekadar menyalin logic repository saat ini**. Dokumen ini dipakai sebagai **acuan refactor / penyesuaian codebase, diagram metodologi, dan narasi skripsi** agar konsisten satu sama lain.

---

## 1. Posisi Dokumen Ini

Gunakan brief ini ketika tujuan utama adalah:

1. Menyelaraskan pipeline implementasi dengan **metode penelitian yang tertulis di BAB III**.
2. Memastikan diagram metodologi, narasi penelitian, dan alur kode tidak saling bertentangan.
3. Menegaskan pemisahan yang jelas antara **jalur data latih** dan **jalur data uji**.
4. Menjaga agar **test set hanya dipakai untuk evaluasi akhir**, bukan untuk tuning.

Jika ada konflik antara:
- **brief lama** yang mengikuti codebase saat ini, dan
- **metodologi penelitian** yang menjadi dasar skripsi,

maka **untuk brief baru ini metodologi penelitian dianggap sebagai sumber kebenaran utama**.

---

## 2. Keputusan Desain Utama

Sebelum menulis alur detail, ada beberapa keputusan desain yang harus dinyatakan secara eksplisit.

### 2.1 Split utama penelitian adalah 70% data latih dan 30% data uji
Metodologi penelitian menyatakan bahwa dataset dibagi menjadi dua bagian utama:
- **Data latih (70%)**
- **Data uji (30%)**

Artinya, secara konseptual dan dokumentatif, alur utama penelitian harus digambarkan sebagai **dua jalur besar** setelah proses persiapan dataset selesai.

### 2.2 Jalur latih dan jalur uji harus dipisahkan secara tegas
Setelah dataset dibagi, kedua subset menempuh alur yang secara struktur serupa pada tahap preprocessing dan ekstraksi fitur, tetapi **tujuan akhirnya berbeda**:
- **Jalur latih** berakhir pada **pelatihan model dan penyimpanan bobot**.
- **Jalur uji** berakhir pada **prediksi sentimen dan evaluasi model** menggunakan bobot yang telah dilatih.

### 2.3 Test set tidak boleh dipakai untuk tuning
Semua keputusan model, termasuk hyperparameter, arsitektur final, dan pemilihan bobot terbaik, **tidak boleh menggunakan data uji**. Jika tuning tetap ingin dipertahankan, maka tuning harus dilakukan dengan salah satu pendekatan berikut:
- split internal dari data latih saja,
- cross-validation di dalam data latih,
- atau validation subset yang berasal dari train branch saja.

Validation internal ini boleh ada secara implementasi, tetapi **tidak perlu ditonjolkan sebagai jalur utama pada diagram metodologi**, karena metodologi penelitian kamu menekankan split utama 70:30.

### 2.4 Embedding kontekstual untuk BiLSTM harus berupa urutan token, bukan hanya satu vektor [CLS]
Karena classifier utama adalah **BiLSTM**, maka input classifier harus berupa **sequence contextual embeddings** hasil encoder IndoBERT, misalnya representasi token:
- `[CLS], T1, T2, ..., Tn, [SEP]`

Vektor `[CLS]` boleh disebut sebagai bagian dari keluaran encoder, tetapi **classifier BiLSTM tidak logis jika hanya menerima satu vektor [CLS] saja**, karena BiLSTM dirancang untuk memproses urutan token.

### 2.5 Jika metodologi menyebut Global Max Pooling dan Fully Connected, implementasi harus mengikuti
Jika naskah penelitian dan diagram metodologi menyatakan bahwa classifier BiLSTM terdiri dari:
- Dropout
- BiLSTM
- Global Max Pooling
- Fully Connected
- Softmax

maka implementasi final sebaiknya benar-benar mengikuti urutan tersebut. Jika codebase saat ini masih memakai pooling lain atau classifier yang lebih sederhana, maka:
- kode harus direvisi agar sama dengan metodologi, **atau**
- metodologi harus direvisi agar sama dengan implementasi.

Untuk brief ini, diasumsikan bahwa **metodologi penelitian yang baru adalah target desain yang akan diikuti**.

---

## 3. Prinsip Eksekusi

1. Semua angka, tabel, grafik, dan metrik harus berasal dari komputasi aktual.
2. Gunakan seed global tetap untuk menjaga reproducibility.
3. Pisahkan tegas proses training dan testing.
4. Simpan artefak penting di setiap tahap.
5. Jangan gunakan test set untuk pemilihan konfigurasi model.
6. Semua perubahan data harus terlacak pada log.
7. Jalur preprocessing untuk train dan test harus sama secara aturan.
8. Model evaluasi harus menggunakan bobot hasil training yang telah disimpan.

---

## 4. Struktur Alur Penelitian yang Direkomendasikan

Secara metodologis, alur penelitian yang paling konsisten adalah sebagai berikut:

### Fase A. Akuisisi dan Penyiapan Dataset
1. Komentar TikTok dikumpulkan melalui **web scraping**.
2. Dataset mentah disimpan ke file utama, misalnya `.xlsx`.
3. Label sentimen tersedia atau dilengkapi melalui proses pelabelan manual.
4. Dilakukan pemeriksaan integritas dataset agar data siap dibagi.

### Fase B. Pembersihan Integritas Sebelum Split
Walaupun pada narasi singkat kamu split diletakkan segera setelah dataset terkumpul, secara operasional lebih aman menempatkan **pembersihan integritas dasar** sebelum split. Tujuannya adalah menghindari kebocoran akibat komentar duplikat atau baris tidak valid yang tersebar ke train dan test.

Tahap ini mencakup:
- validasi kolom teks,
- validasi kolom label,
- penghapusan baris teks kosong,
- standarisasi label,
- penghapusan duplikasi exact,
- pembuatan dataset bersih final.

Hasil fase ini adalah **dataset bersih berlabel** yang siap dibagi menjadi train dan test.

> Catatan penting:
> Jika kamu ingin diagram metodologi tetap sederhana, bagian ini dapat tetap disebut sebagai bagian dari persiapan dataset sebelum percabangan train-test. Namun, secara implementasi, langkah ini sebaiknya dilakukan **sebelum split**.

### Fase C. Split Utama 70:30
Dataset bersih kemudian dibagi dengan **stratified split** menjadi:
- **70% data latih**
- **30% data uji**

Setelah titik ini, pipeline bercabang menjadi dua jalur besar.

---

## 5. Jalur 1: Data Latih (Training Pipeline)

Jalur ini digunakan untuk membangun model dan menghasilkan bobot terlatih.

### 5.1 Input Jalur Latih
Masukan jalur ini adalah **data latih hasil split 70%**.

### 5.2 Preprocessing Data Latih
Lakukan preprocessing terhadap data latih dengan aturan yang sama yang nantinya juga akan diterapkan pada data uji. Tahap ini terdiri atas:

1. **Cleaning teks**
   - menghapus URL,
   - menghapus mention,
   - menghapus simbol yang tidak relevan,
   - merapikan spasi,
   - case folding,
   - menangani karakter berulang.

2. **Normalisasi teks**
   - mengubah slang / singkatan / kata tidak baku ke bentuk yang lebih baku,
   - menjaga token domain tertentu bila perlu.

3. **Tokenisasi awal**
   - memecah teks agar siap diproses tokenizer model.

Output tahap ini adalah teks bersih dan ternormalisasi yang siap masuk ke IndoBERT tokenizer.

### 5.3 Tokenisasi Menggunakan IndoBERT Tokenizer
Data latih yang sudah dipraproses kemudian ditokenisasi menggunakan tokenizer IndoBERT.

Aturan tokenisasi:
- token khusus `[CLS]` ditambahkan di awal,
- token khusus `[SEP]` ditambahkan di akhir,
- hasil tokenisasi menghasilkan `input_ids`, `attention_mask`, dan jika dipakai `token_type_ids`.

### 5.4 Ekstraksi Fitur Menggunakan IndoBERT
Urutan token hasil tokenizer diproses oleh model IndoBERT.

Secara konseptual, tahap ini mencakup:
- Token Embedding,
- Segment Embedding,
- Position Embedding,
- pemrosesan melalui **Transformer Encoder 12 layer**.

Hasil akhir tahap ini adalah **embedding kontekstual berurutan**, yaitu representasi setiap token dalam urutan:
- `[CLS], T1, T2, ..., Tn, [SEP]`

Embedding kontekstual ini menjadi input untuk classifier BiLSTM.

### 5.5 Pelatihan Classifier BiLSTM
Embedding kontekstual dari data latih digunakan untuk melatih classifier BiLSTM.

Arsitektur target classifier pada brief ini adalah:
1. **Dropout**
2. **Bidirectional LSTM**
3. **Global Max Pooling**
4. **Fully Connected Layer**
5. **Softmax**

Penjelasan fungsi tiap bagian:
- **Dropout**: regularisasi awal agar model tidak terlalu mudah overfit.
- **BiLSTM**: membaca urutan token dari dua arah untuk menangkap dependensi sekuensial.
- **Global Max Pooling**: mengambil fitur paling dominan dari keluaran sekuens BiLSTM.
- **Fully Connected Layer**: memetakan fitur hasil pooling ke ruang klasifikasi.
- **Softmax**: menghasilkan probabilitas tiga kelas sentimen.

### 5.6 Training Loop
Training loop minimal harus mencakup:
- forward pass,
- perhitungan loss,
- backward pass,
- optimizer step,
- pencatatan metrik per epoch.

Jika diperlukan early stopping atau validation internal, gunakan subset yang berasal **hanya dari data latih**, bukan dari data uji utama.

### 5.7 Output Jalur Latih
Output utama jalur latih adalah:
- bobot model terlatih,
- checkpoint model terbaik,
- history training,
- konfigurasi model yang dipakai.

Pada diagram metodologi, jalur latih secara konseptual berakhir di:
- **Model/Bobot Tersimpan**

Titik ini lalu terhubung ke jalur uji melalui proses **load bobot model**.

---

## 6. Jalur 2: Data Uji (Testing Pipeline)

Jalur ini digunakan untuk inferensi dan evaluasi akhir model.

### 6.1 Input Jalur Uji
Masukan jalur ini adalah **data uji hasil split 30%**.

### 6.2 Preprocessing Data Uji
Tahap preprocessing pada data uji **harus sama** dengan preprocessing data latih, tanpa menambahkan aturan baru yang hanya muncul pada test.

Tahap ini meliputi:
- cleaning,
- normalisasi,
- tokenisasi awal.

Tujuannya adalah menjaga konsistensi ruang input antara train dan test.

### 6.3 Tokenisasi Menggunakan IndoBERT Tokenizer
Seperti pada data latih, data uji ditokenisasi menggunakan tokenizer IndoBERT dengan menambahkan `[CLS]` dan `[SEP]`.

### 6.4 Ekstraksi Fitur Menggunakan IndoBERT
Data uji diproses melalui encoder IndoBERT yang sama untuk menghasilkan embedding kontekstual berurutan.

### 6.5 Load Bobot Model
Pada tahap ini, classifier tidak dilatih ulang. Sistem hanya:
- memuat bobot model hasil training,
- menyusun ulang arsitektur model yang sama,
- menjalankan inferensi pada data uji.

### 6.6 Prediksi Sentimen
Model menghasilkan prediksi salah satu dari tiga kelas:
- Positif
- Negatif
- Netral

### 6.7 Evaluasi Model
Prediksi pada data uji kemudian dibandingkan dengan label sebenarnya untuk menghasilkan metrik evaluasi.

Metrik wajib yang dihasilkan:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Jika klasifikasi bersifat multi-kelas, sebaiknya juga dihasilkan:
- metrik per kelas,
- macro average,
- support per kelas.

### 6.8 Output Jalur Uji
Output utama jalur uji adalah:
- file prediksi data uji,
- confusion matrix,
- classification report,
- ringkasan metrik evaluasi,
- visualisasi confusion matrix.

Pada diagram metodologi, jalur ini berakhir di:
- **Evaluasi Model**

---

## 7. Titik Pertemuan Akhir (End Node)

Dalam diagram metodologi, jalur data latih dan jalur data uji bertemu pada satu titik akhir (end node / black dot), tetapi secara logika keduanya **tidak mengerjakan hal yang sama**.

Arti titik akhir tersebut adalah:
- jalur latih telah menghasilkan model terlatih,
- jalur uji telah menggunakan model tersebut untuk menghasilkan evaluasi,
- keseluruhan sistem penelitian selesai.

Jadi, end node bukan berarti data latih dan data uji diproses bersama kembali, melainkan menandakan bahwa kedua jalur telah menyelesaikan perannya masing-masing dalam eksperimen.

---

## 8. Struktur Logika Detail yang Harus Dijaga

Agar brief ini benar-benar operasional dan tidak ambigu, beberapa relasi logika berikut harus dipertahankan.

### 8.1 Urutan yang benar
Urutan logika yang disarankan adalah:
1. Web scraping
2. Penyiapan dataset mentah
3. Pembersihan integritas dasar
4. Split 70:30
5. Cabang train dan test
6. Training menghasilkan bobot
7. Testing memuat bobot
8. Evaluasi model

### 8.2 Yang tidak boleh terjadi
- test set dipakai menentukan hyperparameter,
- test set ikut melatih model,
- model uji memakai arsitektur berbeda dari model latih,
- preprocessing train dan test berbeda aturannya,
- data duplikat lintas train-test dibiarkan jika masih bisa dicegah sebelum split.

### 8.3 Posisi validation jika tetap dibutuhkan
Jika implementasi nyata tetap memerlukan validation untuk early stopping atau seleksi model, maka validation harus diperlakukan sebagai:
- **mekanisme internal training**, bukan split utama penelitian.

Dengan kata lain:
- di diagram utama tetap 70% train dan 30% test,
- di implementasi, data train 70% boleh dipecah lagi menjadi train-subset dan val-subset secara internal.

---

## 9. Rekomendasi Struktur File dan Artefak

Agar pipeline ini mudah dijalankan ulang, gunakan artefak berikut.

### 9.1 Data
- `data/dataset_raw.xlsx` atau `data/dataset.xlsx`
- `data/dataset_clean.csv`
- `data/train.csv`
- `data/test.csv`

### 9.2 Model
- `models/best_indobert_bilstm.pt`
- `models/train_config.json`

### 9.3 Output Evaluasi
- `outputs/final_metrics.json`
- `outputs/classification_report.csv`
- `outputs/confusion_matrix.csv`
- `outputs/test_predictions.csv`
- `outputs/figures/confusion_matrix.png`

### 9.4 Output Laporan
- `outputs/bab4_hasil_otomatis.md`
- `outputs/bab5_kesimpulan_saran_otomatis.md`

---

## 10. Mapping Implementasi yang Disarankan ke Codebase

Jika repository ingin diselaraskan dengan brief ini, maka perubahan yang disarankan adalah sebagai berikut.

### 10.1 Step split
Step split harus menghasilkan **dua file utama**:
- `train.csv`
- `test.csv`

Jika validation tetap diperlukan, ia dibuat secara internal di step training atau sebagai file tambahan yang tidak dijadikan pusat narasi metodologi.

### 10.2 Step training model utama
Step training harus menerima:
- `train.csv`
- konfigurasi IndoBERT
- konfigurasi BiLSTM classifier

Lalu menghasilkan:
- bobot model,
- history training,
- log eksperimen.

### 10.3 Step evaluasi
Step evaluasi harus menerima:
- `test.csv`
- bobot model tersimpan

Lalu menghasilkan:
- prediksi,
- metrik,
- confusion matrix,
- classification report.

### 10.4 Step baseline dan tuning
Baseline model dan tuning boleh tetap ada sebagai eksperimen tambahan, tetapi posisinya harus jelas:
- **bukan inti diagram utama metodologi**, dan
- **tidak boleh menggunakan test set untuk pemilihan model**.

---

## 11. Narasi Ringkas Metode Penelitian yang Konsisten dengan Brief Ini

Versi ringkas yang konsisten untuk dijadikan acuan narasi adalah:

"Dataset komentar TikTok dikumpulkan melalui web scraping, kemudian dilakukan pembersihan data dan standarisasi label. Dataset yang telah siap kemudian dibagi menjadi 70% data latih dan 30% data uji menggunakan stratified sampling. Pada jalur data latih, data dipraproses melalui cleaning, normalisasi, dan tokenisasi awal, kemudian ditokenisasi menggunakan IndoBERT tokenizer dengan token khusus [CLS] dan [SEP]. Urutan token tersebut diproses oleh IndoBERT untuk menghasilkan embedding kontekstual, yang selanjutnya digunakan sebagai input classifier BiLSTM. Model dilatih hingga menghasilkan bobot terbaik yang disimpan. Pada jalur data uji, data melalui preprocessing dan tokenisasi yang sama, kemudian diproses oleh IndoBERT untuk menghasilkan embedding kontekstual. Bobot model hasil training dimuat kembali untuk menghasilkan prediksi sentimen. Hasil prediksi kemudian dievaluasi menggunakan accuracy, precision, recall, F1-score, dan confusion matrix." 

---

## 12. Aturan Keras

1. Jangan mengarang metrik.
2. Jangan gunakan test set untuk tuning.
3. Jangan mencampur jalur training dan testing.
4. Jangan menyebut arsitektur yang tidak benar-benar diimplementasikan.
5. Jika naskah menyebut Global Max Pooling + Fully Connected, kode harus disesuaikan agar benar-benar memakainya.
6. Jika implementasi memakai validation internal, jelaskan bahwa validation berasal dari data latih, bukan split utama penelitian.
7. Semua file output penting harus tersimpan dan bisa ditelusuri ulang.

---

## 13. Kesimpulan Operasional

Brief baru ini menegaskan bahwa metodologi penelitian kamu secara logis adalah:
- **satu dataset utama**,
- **satu split utama 70:30**,
- **dua jalur besar setelah split**,
- **training menghasilkan bobot**,
- **testing menggunakan bobot untuk evaluasi**.

Dengan brief ini, diagram metode, narasi BAB III, implementasi pipeline, dan penulisan BAB IV/V dapat dibuat lebih konsisten dan lebih mudah dipertanggungjawabkan.