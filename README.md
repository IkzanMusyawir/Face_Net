# FaceNet Face Verification and Identification
Proyek ini menerapkan FaceNet untuk deteksi wajah, alignment, ekstraksi embedding 512 dimensi, verifikasi wajah satu lawan satu, dan identifikasi multi-kelas menggunakan SVM. Struktur dan alur kerja mengikuti modul praktikum FaceNet.
## Struktur Folder
``` 
FaceNet/
│
├── data/
│   ├── train/
│   │   ├── ikzan/
│   │   │   ├── a1.jpg
│   │   │   └── a2.jpg
│   │   └── stev/
│   │       ├── b1.jpg
│   │       └── b2.jpg
│   │
│   └── val/
│       ├── ikzan/
│       │   └── a1.jpg
│       └── stev/
│           └── b1.jpg
│
├── build_embeddings.py
├── eval_folder.py
├── facenet_svm.joblib
├── predict_one.py
├── train_classifier.py
├── train_knn.py
├── utils_facenet.py
├── verify_cli.py
├── verify_pair.py
├── X_train.npy
└── y_train.npy
``` 

## Tujuan Praktikum

* Memahami deteksi dan alignment wajah dengan MTCNN
* Menghasilkan embedding FaceNet dimensi 512
* Melakukan verifikasi wajah satu lawan satu
* Melakukan identifikasi wajah menggunakan SVM
* Mengevaluasi akurasi model pada data validasi

## Analisis Setiap Kode File
### build_embeddings.py
Berdasarkan analisis mendalam terhadap kode build_embeddings.py, dapat disimpulkan bahwa implementasi ini merupakan solusi yang robust dan efisien untuk pipeline ekstraksi embedding wajah dalam sistem FaceNet. Kode tersebut mengadopsi pendekatan modular yang terstruktur dengan baik, dimana fungsi iter_images berperan sebagai generator yang melakukan traversing direktori secara efisien dan otomatis mendeteksi kelas berdasarkan struktur folder, sementara fungsi build_matrix menangani proses inti ekstraksi embedding dengan integrasi progress tracking menggunakan tqdm dan mekanisme error handling yang memadai untuk mencatat gambar-gambar yang gagal diproses. Keunggulan utama kode ini terletak pada optimasi memory melalui penggunaan generator function, user experience yang baik dengan visualisasi progress bar, serta desain yang memungkinkan eksekusi baik sebagai standalone script maupun imported module, meskipun terdapat peluang peningkatan pada aspek file type validation dan fleksibilitas konfigurasi path untuk penggunaan yang lebih robust.

### eval_folder.py
Berdasarkan analisis kode eval_folder.py, dapat disimpulkan bahwa script ini berfungsi sebagai modul evaluasi kinerja model klasifikasi wajah dengan melakukan validasi terhadap dataset yang tersimpan dalam struktur folder terorganisir. Kode mengimplementasikan pipeline evaluasi yang komprehensif dimulai dari loading model SVM yang telah dilatih, ekstraksi embedding wajah menggunakan utilitas FaceNet, hingga proses prediksi dan perhitungan metrik akurasi secara keseluruhan maupun per-kelas. Keunggulan utama kode ini terletak pada kemampuan melakukan evaluasi granular melalui penggunaan defaultdict yang memungkinkan tracking performa individual setiap kelas, serta mekanisme filtering yang robust untuk menangani kasus dimana embedding wajah tidak berhasil diekstraksi. Namun, kode memiliki ketergantungan kuat pada struktur folder yang spesifik dan asumsi bahwa penamaan folder merepresentasikan label kelas yang valid, yang meskipun efektif untuk use case tertentu, dapat membatasi fleksibilitas dalam skenario deployment yang lebih kompleks.

### predict_one.py
Berdasarkan analisis kode predict_one.py, dapat disimpulkan bahwa script ini berfungsi sebagai modul inferensi real-time untuk sistem pengenalan wajah yang mengimplementasikan mekanisme prediksi tunggal dengan integrasi threshold-based rejection untuk menangani kasus wajah tidak dikenal. Kode tersebut melakukan loading model SVM yang telah dilatih sebelumnya, kemudian mengekstrak embedding wajah dari gambar input menggunakan utilitas FaceNet, dan menghasilkan prediksi kelas beserta confidence score melalui proses seleksi probabilitas tertinggi. Keunggulan utama implementasi ini terletak pada mekanisme unknown rejection yang cerdas melalui parameter threshold yang dapat dikonfigurasi, sehingga sistem tidak hanya mampu mengidentifikasi wajah yang dikenal tetapi juga secara otomatis mendeteksi dan menandai wajah-wajah yang tidak terdapat dalam dataset training ketika confidence score berada di bawah batas tertentu. Namun, arsitektur kode memiliki ketergantungan eksplisit pada path gambar yang hard-coded dan asumsi bahwa model SVM telah tersedia di filesystem, yang meskipun praktis untuk penggunaan dasar, dapat diperluas dengan menambahkan error handling untuk kasus file tidak ditemukan dan mekanisme konfigurasi path yang lebih fleksibel untuk environment production.

### train_classifier.py
Berdasarkan analisis kode train_classifier.py, dapat disimpulkan bahwa script ini berfungsi sebagai modul pelatihan model klasifikasi untuk sistem pengenalan wajah yang mengimplementasikan pipeline machine learning terstandarisasi dengan preprocessing StandardScaler dan classifier SVM berkernel RBF. Kode tersebut melakukan loading data training berupa embeddings wajah yang telah diekstraksi sebelumnya, membangun model klasifikasi dengan konfigurasi yang dioptimalkan untuk data wajah melalui parameter C=10, gamma='scale', dan penyeimbangan kelas otomatis, serta mengevaluasi performa model secara langsung pada data training. Keunggulan utama implementasi ini terletak pada penggunaan pipeline scikit-learn yang memastikan konsistensi preprocessing selama training dan inference, serta kemampuan menghasilkan probability estimates yang crucial untuk mekanisme rejection wajah tidak dikenal, meskipun pendekatan tanpa cross-validation dapat membatasi kemampuan generalisasi assessment pada dataset yang lebih kecil dan kompleks.

###  train_knn.py
Berdasarkan analisis kode train_knn.py, dapat disimpulkan bahwa script ini berfungsi sebagai alternatif modul pelatihan model klasifikasi untuk sistem pengenalan wajah dengan mengimplementasikan algoritma K-Nearest Neighbors (KNN) sebagai penanding untuk pendekatan SVM yang digunakan sebelumnya. Kode tersebut memanfaatkan embeddings wajah yang telah diekstraksi untuk membangun pipeline klasifikasi yang terdiri dari tahap standardisasi fitur menggunakan StandardScaler dilanjutkan dengan klasifikasi KNN menggunakan 3 tetangga terdekat dengan metrik jarak Euclidean. Keunggulan implementasi ini terletak pada kesederhanaan algoritma KNN yang efektif untuk data dengan distribusi kompleks dan kemampuan kerja yang baik pada dataset berukuran kecil, meskipun pendekatan ini mungkin menghadapi tantangan skalabilitas pada dataset yang sangat besar akibat kebutuhan komputasi yang meningkat seiring pertumbuhan data training.

### utils_facenet.py
Berdasarkan analisis kode utils_facenet.py, dapat disimpulkan bahwa script ini berfungsi sebagai utility core yang menyediakan seluruh fungsi fundamental untuk pipeline pengenalan wajah, mulai dari pembacaan gambar, deteksi dan alignment wajah menggunakan MTCNN, hingga ekstraksi embedding 512-dimensi dengan model InceptionResnetV1 yang telah dilatih sebelumnya pada dataset VGGFace2. Kode tersebut mengimplementasikan optimasi performa melalui dekorator @torch.no_grad() untuk menghilangkan overhead komputasi gradient selama inference, serta memiliki kemampuan adaptif untuk memanfaatkan GPU ketika tersedia sambil tetap maintain compatibility dengan CPU. Keunggulan utama implementasi ini terletak pada arsitektur modular yang memisahkan setiap tahap proses menjadi fungsi-fungsi spesifik dengan tanggung jawab terdefinisi dengan jelas, dilengkapi dengan error handling yang robust untuk menangani kegagalan pembacaan file dan kasus wajah tidak terdeteksi, serta menyertakan utilitas cosine similarity yang essential untuk perbandingan embedding wajah dalam proses verifikasi.

###  verify_cli.py
Berdasarkan analisis kode verify_cli.py, dapat disimpulkan bahwa script ini berfungsi sebagai command-line interface yang efisien untuk melakukan verifikasi wajah one-to-one dengan membandingkan dua gambar input dan menghitung similarity score menggunakan cosine similarity antara embedding wajah yang diekstraksi. Kode tersebut mengimplementasikan argument parsing yang robust dengan dukungan parameter threshold yang dapat dikustomisasi, serta menangani berbagai skenario penggunaan termasuk kasus dimana wajah tidak terdeteksi pada salah satu gambar input. Keunggulan utama implementasi ini terletak pada antarmuka command-line yang user-friendly yang memungkinkan integrasi mudah dengan workflow otomatisasi dan batch processing, dilengkapi dengan output yang informatif yang menampilkan similarity score numerik dan keputusan biner MATCH/NO MATCH berdasarkan threshold yang ditentukan, menjadikannya tool yang praktis untuk penggunaan production dan testing yang repetitif.

### verify_pair.py
Berdasarkan analisis kode verify_pair.py, dapat disimpulkan bahwa script ini berfungsi sebagai tool verifikasi wajah sederhana yang melakukan perbandingan langsung antara dua gambar yang telah ditentukan secara hard-coded dalam kode, menghitung similarity score menggunakan cosine similarity antara embedding wajah yang diekstraksi, dan memberikan keputusan biner berdasarkan threshold tetap sebesar 0.85. Kode tersebut mengimplementasikan mekanisme dasar verifikasi one-to-one dengan penanganan yang tepat untuk kasus dimana wajah tidak terdeteksi pada salah satu gambar, namun memiliki keterbatasan dalam fleksibilitas karena path gambar yang statis dan threshold yang tidak dapat dikonfigurasi tanpa modifikasi kode langsung, menjadikannya lebih cocok untuk testing dan demonstrasi cepat daripada penggunaan production yang membutuhkan dinamisme parameter.
## Cara Menjalankan
* Jalankan build_embeddings.py untuk membuat embedding training.
* Jalankan train_classifier.py untuk melatih model SVM.
* Jalankan predict_one.py untuk mengenali satu gambar.
* Jalankan verify_pair.py untuk membandingkan dua wajah.
* Jalankan verify_cli.py untuk verifikasi cepat lewat terminal.
* Jalankan eval_folder.py untuk mengevaluasi performa model pada folder val.
## Fungsi Output
* X_train.npy berisi embedding wajah untuk training.
* y_train.npy berisi label wajah untuk training.
* facenet_svm.joblib berisi model klasifikasi identitas.
* Training accuracy menunjukkan performa model pada data train.
* Hasil prediksi menunjukkan identitas wajah dan tingkat kepercayaan.
* Hasil verifikasi menunjukkan kekuatan kemiripan dua wajah.
* Akurasi evaluasi menunjukkan kemampuan model pada data uji.
