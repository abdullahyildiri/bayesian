Bu 3 Python kodu, diyabet hastalığının olup olmadığını belirlemek için Naive Bayes sınıflandırıcısını kullanarak bir makine öğrenimi modeli geliştirmeyi amaçlar. Bu kod, bir diyabet veri seti üzerinde çalışır ve veriyi eğitim ve test setlerine böler, ardından farklı algoritmalar kullanarak en iyi hiperparametreleri bulur. Bu 3 kodun sonuçlarında doğruluk oranına göre hangi algoritmayı kullanmamız gerektiğini öğreniriz

GRED SEARCH
1. Giriş (Özet)
Bu Python kodu, diyabet hastalığının olup olmadığını belirlemek için Çoklu Nominal Naive Bayes sınıflandırıcısını kullanarak bir makine öğrenimi modeli geliştirmeyi amaçlar. Bu kod, bir diyabet veri seti üzerinde çalışır ve veriyi eğitim ve test setlerine böler, ardından Grid Search yöntemini kullanarak en iyi hiperparametreleri bulur. Son olarak, en iyi modeli kullanarak eğitim ve test setlerinde tahminler yapar ve doğruluk sonuçlarını hesaplar.
2. Metot
Veri Yükleme ve Ön İşleme: İlk olarak, diyabet veri seti pandas kütüphanesi kullanılarak yüklenir ve özellikler (X) ile hedef değişken (y) ayrılır.
Veri Seti Bölme: Veri seti, eğitim ve test setlerine train_test_split fonksiyonu kullanılarak ayrılır. Bu durumda, veri setinin %20'si test seti olarak ayrılır.
Model Oluşturma ve Grid Search: Çoklu Nominal Naive Bayes sınıflandırıcısı başlatılır. Ardından, Grid Search yöntemi ile belirli bir hiperparametre alanında en iyi modeli bulmak için çapraz doğrulama yapılır. Grid Search, belirli bir hiperparametre grid'i üzerinde çalışır ve belirli bir metrik (bu durumda doğruluk) için en iyi hiperparametreleri seçer.
En İyi Modelin Seçilmesi ve Test Setinde Tahminler: Grid Search sonucunda elde edilen en iyi model seçilir ve bu model kullanılarak eğitim ve test setlerinde tahminler yapılır.
Doğruluk Sonuçlarının Hesaplanması: Tahminler ile gerçek etiketler arasındaki doğruluk sonuçları hesaplanır.
Performans Metriklerinin Yazdırılması: Eğitim ve test setleri için doğruluk sonuçları ve sınıflandırma raporları yazdırılır.
3. Sonuçlar
En iyi alpha değeri: 0.1
Eğitim doğruluğu: 0.6042345276872965
Test doğruluğu: 0.6623376623376623
Eğitim seti performansı:
               precision    recall  f1-score   support

           0       0.70      0.69      0.69       401
           1       0.43      0.45      0.44       213

    accuracy                           0.60       614
   macro avg       0.57      0.57      0.57       614
weighted avg       0.61      0.60      0.61       614

Test seti performansı:
               precision    recall  f1-score   support

           0       0.73      0.75      0.74        99
           1       0.53      0.51      0.52        55

    accuracy                           0.66       154
   macro avg       0.63      0.63      0.63       154
weighted avg       0.66      0.66      0.66       154


RANDOM SEARCH
1. Giriş (Özet)
Bu Python kodu, diyabet hastalığının olup olmadığını belirlemek için Gaussian Naive Bayes sınıflandırıcısını kullanarak bir makine öğrenimi modeli geliştirmeyi amaçlar. Ancak, bu kod, standart Gaussian Naive Bayes yerine Randomized Search yöntemini kullanarak hiperparametrelerin optimizasyonunu gerçekleştirir ve min-max normalizasyonu ile önceden işlenmiş bir veri seti üzerinde çalışır.
2. Metot
Veri Yükleme ve Ön İşleme: Diyabet veri seti pandas kütüphanesi kullanılarak yüklenir. Eksik veriler kontrol edilir ve 0 değerleri eksik veri olarak kabul edilir, daha sonra ortalama değerlerle doldurulur.
Veri Normalizasyonu ve Ayırma: Veri seti min-max normalizasyonu ile normalize edilir ve özellikler (X) ile hedef değişken (y) ayrılır.
Eğitim ve Test Setlerine Ayırma: Veri seti, eğitim ve test setlerine train_test_split fonksiyonu kullanılarak ayrılır.
Gaussian Naive Bayes Modelinin Eğitimi: Gaussian Naive Bayes sınıflandırıcısı başlatılır ve eğitim seti üzerinde eğitilir.
Randomized Search ile Hiperparametre Ayarlama: Randomized Search, Gaussian Naive Bayes sınıflandırıcısının hiperparametrelerini optimize etmek için kullanılır. Bu, daha iyi bir model performansı sağlayabilir.
Model Eğitimi ve Tahminler: En iyi model seçilir ve test seti üzerinde tahminler yapılır.
Doğruluk ve Performans Metriklerinin Yazdırılması: Test seti üzerinde yapılan tahminlerin doğruluğu hesaplanır ve sınıflandırma raporu yazdırılır.
Confusion Matrix Görselleştirme: Son olarak, confusion matrix görselleştirilir.
3. Sonuçlar
Accuracy: 0.7012987012987013
              precision    recall  f1-score   support

           0       0.73      0.88      0.80        51
           1       0.60      0.35      0.44        26

    accuracy                           0.70        77
   macro avg       0.66      0.61      0.62        77
weighted avg       0.68      0.70      0.68        77



BAYESİAN OPTİMİZATİON
1. Giriş (Özet)
Bu Python kodu, diyabet hastalığının olup olmadığını belirlemek için Naive Bayes sınıflandırıcısını kullanarak bir makine öğrenimi modeli geliştirmeyi amaçlar. Ancak, bu kod, standart Naive Bayes yerine Bayesian Optimization yöntemini kullanarak hiperparametrelerin optimizasyonunu gerçekleştirir ve min-max normalizasyonu ile önceden işlenmiş bir veri seti üzerinde çalışır.

2. Metot
Veri Yükleme ve Ön İşleme: Diyabet veri seti pandas kütüphanesi kullanılarak yüklenir. Eksik veriler kontrol edilir ve 0 değerleri eksik veri olarak kabul edilir, daha sonra ortalama değerlerle doldurulur.
Veri Normalizasyonu ve Ayırma: Veri seti min-max normalizasyonu ile normalize edilir ve özellikler (X) ile hedef değişken (y) ayrılır.
Eğitim ve Test Setlerine Ayırma: Veri seti, eğitim ve test setlerine train_test_split fonksiyonu kullanılarak ayrılır.
Bayesian Optimization ile Hiperparametre Ayarlama: Bayesian Optimization, Naive Bayes sınıflandırıcısının hiperparametrelerini optimize etmek için kullanılır. Bu, daha iyi bir model performansı sağlayabilir.
Model Eğitimi ve Tahminler: En iyi model seçilir ve test seti üzerinde tahminler yapılır.
Doğruluk ve Performans Metriklerinin Yazdırılması: Test seti üzerinde yapılan tahminlerin doğruluğu hesaplanır ve sınıflandırma raporu yazdırılır.
Confusion Matrix Görselleştirme: Son olarak, confusion matrix görselleştirilir.
3. Sonuçlar
Accuracy: 0.7922077922077922
              precision    recall  f1-score   support

           0       0.80      0.90      0.85        50
           1       0.76      0.59      0.67        27

    accuracy                           0.79        77
   macro avg       0.78      0.75      0.76        77
weighted avg       0.79      0.79      0.79        77

SONUÇ:
3 farklı modelde de doğruluk oranları farklı çıkmıştır. En iyi doğruluk oranına sahip model min-max normalizasyonu ve bayesian optimization kulandığımız modeldir.Bayesian Optimization, hiperparametreleri en iyi performansı elde etmek için otomatik olarak ayarlar.Min-max normalizasyonu da veri setinin özelliklerini belirli bir aralığa ölçeklendirir. Bu, daha iyi performans sağlamamızın sebeplerinden biri de olabilir.















