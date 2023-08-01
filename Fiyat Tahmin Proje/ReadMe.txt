Bu projede kullandığımız veri seti arabaların bazı özelliklerini tutuyor. Ben bu projede model yazıp veri setiyle modeli eğitmeyi hedefledim.

Amacımız bu veri setindeki arabaların özelliklerine göre fiyat tahmin yapan bir regresyon modeli oluşturmaktı. 

Bu projede uyguladığım adımları tek tek incelersek ;
1-) Veri setini anlayıp modelde kullanacağım kısmı belirledikten sonra kullanmayacağım kolonları düşürmek oldu.
2-) Kullanmayacağım kolonları düşürdükten sonra isnull() fonksiyonuyla boş olan hücreleri kontrol ettim veri setim düzenli ve eksiksiz olduğu için herhangi bir boş hücreye rastlamadım eğer rastlasaydım veri setinin büyüklüğüne göre ve boş yerlerin sayısına göre ortalama değerlerle doldurmak veya boş hücreleri silmek gibi işlemler yapabilirdim.
3-) Veri setimi modelde kullanılacak hale getirdikten sonra benim girdilerimi ve çıktılarımı belirlemem gerekiyor benim bu veri setinde modelden alacağım çıktı fiyat(price) girdilerim ise fiyat dışında kalan düzenlediğim veri setindeki diğer kolonlar oluyor.
4-) Girdi ve çıktılarımı belirledikten sonra x değişkenine girdilerimi y değişkenine ise çıktılarımı atadım bunun sebebi eğer çıktılarımı girdilerle birlikte verirsem modelim ezber yapıp tam doğru sonuç vericektir bunu istemiyoruz.
5-) Veri setim artık tamammen hazır ve modelde kullanılacak halde şimdi veri setimi eğitim ve test için bölmem gerekiyor bunu sklearn kütüphanesinin train_test_split fonksiyonuyla yaptım ve verilerimin %20 lik kadarını test için ayırdım.
6-) Veri setimi ayırdıktan sonra label encoding ve 0 ile 1 arasında olan bir normalizasyon yapmam gerekiyor. Bunun için sklearn kütüphanesinin MinMaxScaler ve StandartScaler fonksiyonlarını kullandım eğitimde kullanılacak veriyi fit_transform test verisini transform ettim çünkü test verisini de fitlersek model için ipucu koymuş oluruz buda bir nevi modelin öğrenmekten ziyade ezber yaptığını görürüz.
7-) Verilerle işim biti şimdi modeli oluşturmam gerektiği için keras kütüphanesini import ettim Sequentail fonksiyonu ile model nesnesi oluşturduktan sonra add foksiyonu sayesinde katman ve nöronlarımı ekledim.
8-) Modelde arada Dropout kullanmamın sebebi modelin öğrendiği bilgileri unutturup ezber yapmasının önüne geçmek.
9-) Son katmanda 1 nöron koymamın sebebi tek çıktı almak istediğimden eğer iki çıktı almak isteseydik 2 yazmamız gerekti.
10-) Katman ve nöronlarımı ekledikten sonra modeli derlememiz gerekiyor ve bunun için compile fonksiyonunu kullandık ben optimizer için "adam" kullandım çünkü adam optimizasyon için iyi bir algoritmadır.
11-) Yeni veriler için tahmin yapmam gerektiği için predict() fonksiyonunu kullandım.
12-) Modelin tahmin ettiği değerler ile gerçek değerleri kıyaslamak için matplotlib kütüphanesiyle gerçek test verileri ile tahmin verileri arasında grafik çizdirerek modelin doğruluk tablosunu çıkardım main1.py dosyasında.
13-) Modelin r^2 skorunu hem eğitim hem test için hesaplanmış ve loss değerinin grafiğinin çizilmiş hali main.py dosyasında bulunmaktadır.

TEŞEKKÜRLER

