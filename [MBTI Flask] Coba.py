from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=16))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_model_weights():
    model = create_model()
    model.load_weights('D:\Bangkit 2023\Capstone\model_mbti.h5')
    return model

model = load_model_weights()

questions = [
    "Apakah Anda lebih suka belajar melalui interaksi dengan orang lain? (Y/T)",
    "Apakah Anda lebih suka belajar melalui pengamatan dan pengalaman langsung? (Y/T)",
    "Apakah Anda lebih suka belajar melalui membaca dan riset mandiri? (Y/T)",
    "Apakah Anda lebih suka belajar dengan mendengarkan penjelasan dari orang lain? (Y/T)",
    "Apakah Anda lebih suka belajar dengan mencoba dan berlatih secara langsung? (Y/T)",
    "Apakah Anda lebih suka belajar dengan berdiskusi dan berbagi ide dengan orang lain? (Y/T)",
    "Apakah Anda lebih suka belajar dengan membuat catatan dan merencanakan langkah-langkah? (Y/T)",
    "Apakah Anda lebih suka belajar dengan mengikuti instruksi dan panduan yang jelas? (Y/T)",
    "Apakah Anda lebih suka menghabiskan waktu luang dengan aktivitas luar ruangan? (Y/T)",
    "Apakah Anda lebih suka menghabiskan waktu luang dengan membaca buku atau menonton film? (Y/T)",
    "Apakah Anda lebih suka menghabiskan waktu luang dengan bermain olahraga atau aktif fisik? (Y/T)",
    "Apakah Anda lebih suka menghabiskan waktu luang dengan berkreasi atau menggambar? (Y/T)",
    "Apakah Anda lebih suka menghabiskan waktu luang dengan berinteraksi sosial bersama teman? (Y/T)",
    "Apakah Anda lebih suka menghabiskan waktu luang dengan merenung atau berpikir secara pribadi? (Y/T)",
    "Apakah Anda lebih suka menghabiskan waktu luang dengan mengeksplorasi tempat baru? (Y/T)",
    "Apakah Anda lebih suka menghabiskan waktu luang dengan menulis atau menyusun ide? (Y/T)"
]

mbti_description = {
        "ISTJ": "Individu yang praktis dan berpikiran fakta yang kendalanya tidak diragukan lagi. Mereka cenderung terorganisir, bertanggung jawab, dan dapat diandalkan. ISTJ memiliki ketelitian tinggi dalam menjalankan tugas-tugas yang terstruktur.",
        "ISFJ": "Pelindung yang sangat berdedikasi dan hangat, selalu siap membela orang yang mereka cintai. Mereka cenderung mengutamakan kebutuhan orang lain dan memiliki kepekaan sosial yang tinggi. ISFJ sering kali menjadi pendengar yang baik dan penolong setia.",
        "INFJ": "Pendiam dan mistis, namun idealis yang sangat inspiratif dan tak kenal lelah. Mereka sangat peduli dengan kesejahteraan orang lain dan memiliki intuisi yang kuat. INFJ cenderung menjadi pembela nilai-nilai moral dan memiliki kemampuan untuk memahami perasaan dan motivasi orang lain.",
        "INTJ": "Pemikir imajinatif dan strategis dengan rencana untuk segalanya. Mereka memiliki wawasan yang mendalam dan berorientasi pada solusi. INTJ cenderung independen, analitis, dan memiliki kemampuan pemecahan masalah yang kuat.",
        "ISTP": "Eksperimen yang berani dan praktis, master dari semua jenis alat. Mereka memiliki keterampilan teknis yang kuat dan cenderung berpikir logis. ISTP sering kali menikmati tantangan fisik dan memiliki pemahaman yang mendalam tentang bagaimana sesuatu bekerja.",
        "ISFP": "Seniman yang fleksibel dan menawan, selalu siap untuk mengeksplorasi dan merasakan sesuatu yang baru. Mereka sangat menghargai keindahan dan ekspresi diri. ISFP cenderung mengutamakan nilai-nilai pribadi dan memiliki kemampuan untuk menciptakan karya seni yang bermakna.",
        "INFP": "Orang yang puitis, baik hati, dan altruistik, selalu bersemangat untuk membantu tujuan yang baik. Mereka memiliki nilai-nilai yang kuat, sensitif terhadap perasaan orang lain, dan cenderung berpikir secara abstrak. INFP sering kali menjadi sumber inspirasi dan pendorong perubahan positif.",
        "INTP": "Penemu inovatif dengan rasa haus yang tak terpadamkan akan pengetahuan. Mereka adalah pemikir yang kritis, analitis, dan memiliki pemikiran yang logis. INTP cenderung tertarik pada konsep-konsep abstrak dan memiliki keingintahuan yang besar terhadap dunia.",
        "ESTP": "Orang yang cerdas, energik, dan sangat perseptif yang benar-benar menikmati hidup di ujung tanduk. Mereka cenderung berani mengambil risiko, antusias, dan berorientasi pada hasil. ESTP sering kali memiliki kemampuan komunikasi yang kuat dan suka menjadi pusat perhatian.",
        "ESFP": "Orang-orang yang spontan, energik, dan antusias - hidup tidak pernah membosankan disekitar mereka. Mereka cenderung penuh semangat, suka berinteraksi dengan orang lain, dan pandai bergaul. ESFP sering kali menjadi sumber kegembiraan dan keceriaan dalam kelompok sosial.",
        "ENFP": "Semangat bebas yang antusias, kreatif, dan mudah bergaul yang selalu dapat menemukan alasan untuk tersenyum. Mereka cenderung optimis, bersemangat, dan memiliki kemampuan memotivasi orang lain. ENFP sering kali menjadi penggerak sosial dan pemimpin yang inspiratif.",
        "ENTP": "Pemikir cerdas dan ingin tahu yang tidak bisa menolak tantangan intelektual. Mereka cenderung kreatif, berpikir lateral, dan memiliki kemampuan berbicara yang baik. ENTP sering kali menjadi inovator yang berani dan memiliki keingintahuan yang luas dalam berbagai topik.",
        "ESTJ": "Administrator yang luar biasa, tak tertandingi dalam mengelola berbagai hal atau orang. Mereka cenderung berorientasi pada tugas, berdisiplin, dan memiliki pemikiran yang rasional. ESTJ memiliki kemampuan organisasi yang kuat dan cenderung menjadi pemimpin yang efisien.",
        "ESFJ": "Orang-orang yang luar biasa peduli, pandai bergaul dan populer, selalu ingin membantu. Mereka cenderung menjaga hubungan yang erat dengan orang lain, peka terhadap kebutuhan mereka, dan berkomitmen untuk menciptakan harmoni. ESFJ sering kali menjadi pendukung setia dalam kelompok sosial.",
        "ENFJ": "Pemimpin yang karismatik dan inspiratif, mampu memikat pendengarnya. Mereka sangat peduli dengan kesejahteraan orang lain dan memiliki kemampuan untuk membentuk hubungan yang kuat. ENFJ sering kali menjadi motivator dan penggerak sosial yang mampu mempengaruhi perubahan positif.",
        "ENTJ": "Pemimpin yang berani, imajinatif, dan berkemauan keras, selalu menemukan jalan - atau membuatnya. Mereka memiliki pandangan jauh ke depan, berorientasi pada tujuan, dan mampu menggerakkan orang lain untuk mencapai keberhasilan. ENTJ sering kali menjadi pemimpin yang karismatik dan strategis." 
    }

mbti_recommend = {
        "ISTJ": "Untuk tipe MBTI ISTJ, ada beberapa rekomendasi belajar yang dapat membantu Anda memperoleh pemahaman yang efektif. Pertama, buatlah jadwal belajar yang terstruktur sesuai dengan kecenderungan ISTJ yang menghargai keteraturan dan struktur. Patuhi jadwal tersebut dengan disiplin.\n \
             Selanjutnya, fokuslah pada detail dan fakta. ISTJ cenderung memperhatikan detail dan fakta secara mendalam. Usahakan untuk memahami konsep secara menyeluruh dan kuasai detail-detail penting yang relevan.\n \
             Metode belajar tradisional seperti membaca teks, membuat catatan, dan melakukan latihan sistematis cocok bagi ISTJ. Manfaatkan sumber daya yang terpercaya, seperti buku, jurnal akademik, atau sumber daya resmi lainnya, untuk mendalami pemahaman Anda.\n \
             Penggunaan daftar tugas dan checklist sangat berguna dalam mengatur pekerjaan yang harus diselesaikan. Pastikan Anda membuat daftar tugas yang jelas dan gunakan checklist untuk memastikan tidak ada yang terlewat.\n \
             Pendekatan step-by-step sesuai dengan kecenderungan ISTJ yang menghargai langkah-demi-langkah. Pecahkan materi kompleks menjadi bagian-bagian yang lebih kecil untuk memudahkan pemahaman dan pengolahan informasi.\n \
             Lakukan evaluasi diri secara objektif untuk mengidentifikasi kekuatan dan kelemahan Anda. Fokuslah pada perbaikan dan perkuat kelemahan Anda melalui usaha yang terstruktur.\n \
             Terakhir, gunakan ringkasan dan diagram untuk mengorganisir informasi dan memvisualisasikan hubungan antara konsep. Hal ini akan membantu Anda mengingat dan memahami materi dengan lebih baik.",
        "ISFJ": "Untuk tipe MBTI ISFJ, ada beberapa rekomendasi belajar yang dapat membantu Anda memperoleh pemahaman yang efektif. Pertama, penting untuk menciptakan lingkungan belajar yang nyaman. ISFJ cenderung belajar dengan baik dalam lingkungan yang tenang dan bebas dari gangguan. Buatlah tempat belajar yang sesuai dan nyaman bagi Anda.\n \
             Selanjutnya, andalkan metode belajar tradisional yang terstruktur, seperti membaca buku teks, membuat catatan, dan melibatkan diri dalam latihan. Metode ini cocok dengan kecenderungan ISFJ dalam memproses informasi.\n \
             Gunakan pengulangan dan praktikkan materi secara teratur. Pengulangan membantu memperkuat pemahaman, sementara praktik dengan menjawab soal latihan atau mengulangi konsep akan membantu memperkuat keterampilan Anda.\n \
             Buat catatan yang rapi dan teratur saat belajar, dan gunakan daftar tugas untuk mengatur pekerjaan yang harus diselesaikan. Ini membantu menjaga keteraturan dan memberikan panduan yang jelas dalam proses belajar.\n \
             Cari kejelasan dan petunjuk yang jelas dalam materi yang dipelajari. ISFJ menghargai kejelasan, jadi pastikan Anda memahami instruksi dengan baik dan jika perlu, tanyakan jika ada hal yang belum jelas.\n \
             Terlibatlah dalam pembelajaran praktis dengan mengaplikasikan materi yang dipelajari dalam situasi nyata atau contoh-contoh sehari-hari. Hal ini akan membantu memperkuat pemahaman dan menghubungkan konsep-konsep tersebut dengan kehidupan Anda.\n \
             Kolaborasi dalam kelompok belajar juga bermanfaat. Diskusikan materi dengan orang lain dalam kelompok belajar untuk mendapatkan wawasan baru dan saling mendukung dalam pemahaman.\n \
             Terakhir, penting untuk menghargai kejujuran dan integritas dalam belajar. ISFJ cenderung menghargai nilai-nilai moral dan etika. Jadi, tetaplah jujur dalam belajar dan hindari tindakan plagiat atau kecurangan.",
        "INFJ": "Untuk tipe MBTI INFJ, ada beberapa rekomendasi belajar yang dapat membantu Anda memperoleh pemahaman yang efektif. Pertama, penting untuk menciptakan waktu dan ruang yang tenang. Ciptakan lingkungan belajar yang minim gangguan agar dapat fokus dan melakukan refleksi secara optimal.\n \
             Selanjutnya, manfaatkan pendekatan holistik dalam belajar. INFJ cenderung melihat hubungan dan pola dalam informasi. Cari kaitan antara konsep-konsep yang dipelajari untuk memperdalam pemahaman Anda.\n \
             Anda juga dapat memanfaatkan kemampuan empati yang kuat sebagai INFJ. Cobalah untuk memahami perspektif orang lain dalam materi yang dipelajari. Hal ini akan membantu memperluas pemahaman Anda.\n \
             Gunakan teknik visual seperti diagram, grafik, atau mind map untuk membantu memvisualisasikan dan memahami konsep-konsep yang kompleks. Teknik ini akan membantu Anda mengorganisir informasi dengan lebih baik.\n \
             Sesuai dengan kecenderungan INFJ, berikan waktu untuk refleksi. Merenung dan mengaitkan pemahaman Anda dalam catatan atau jurnal dapat membantu Anda memproses informasi secara mendalam.\n \
             Diskusikan materi dengan kelompok kecil teman sekelas atau belajar secara satu-satu untuk mendapatkan perspektif baru dan memperdalam pemahaman Anda.\n \
             Terapkan nilai-nilai pribadi Anda dalam belajar. INFJ cenderung memiliki sistem nilai yang kuat. Pilih topik studi atau proyek yang mencerminkan minat dan tujuan hidup Anda untuk meningkatkan motivasi dan keterlibatan dalam pembelajaran.\n \
             Jaga keseimbangan antara teori dan praktik. Sebagai pemikir intuitif, INFJ perlu menghubungkan konsep-konsep teoritis dengan aplikasi praktis dalam dunia nyata. Temukan cara untuk mengaitkan apa yang dipelajari dengan contoh atau situasi praktis.\n \
             Dengan mengikuti rekomendasi ini, Anda dapat memaksimalkan potensi belajar Anda sebagai seorang INFJ dan mencapai pemahaman yang mendalam dalam bidang yang diminati.",
        "INTJ": "Untuk tipe MBTI INTJ, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, buat rencana belajar terstruktur dengan membuat jadwal belajar yang terperinci dan mengikuti rencana yang telah ditentukan. Disiplin dalam mengikuti jadwal akan membantu Anda mengatur waktu dengan efisien. \n \
             Selanjutnya, fokus pada pemahaman konsep-konsep secara mendalam dengan melakukan analisis kritis dan menemukan hubungan antara konsep-konsep tersebut. Berusaha untuk memahami inti dari konsep-konsep yang dipelajari akan membantu Anda memperoleh pemahaman yang lebih komprehensif.\n \
             Pilih sumber belajar yang menantang dengan menggunakan sumber belajar yang memerlukan pemikiran kritis dan menawarkan perspektif baru atau pemecahan masalah kompleks. Berinteraksi dengan sumber-sumber belajar yang menantang akan mendorong Anda untuk berpikir secara lebih mendalam dan melihat hal-hal dari sudut pandang yang berbeda.\n \
             Gunakan metode visualisasi seperti diagram, grafik, atau mind map untuk memvisualisasikan hubungan antara konsep-konsep yang dipelajari. Visualisasi dapat membantu Anda mengorganisir informasi dan memperkuat pemahaman melalui penggambaran visual.\n \
             Berdiskusi dan berkolaborasi dengan orang lain seperti teman sekelas, rekan kerja, atau forum online untuk mendapatkan wawasan baru. Diskusi dengan orang lain akan memperluas pemahaman Anda melalui pertukaran ide dan perspektif yang berbeda.\n \
             Luangkan waktu untuk refleksi setelah belajar dengan merenungkan materi yang dipelajari dan mencoba menerapkannya dalam konteks yang berbeda. Merenungkan materi akan membantu Anda menginternalisasikan konsep-konsep tersebut dan melihat bagaimana konsep-konsep tersebut dapat diterapkan dalam situasi nyata.\n \
             Berikan diri tantangan dengan memberikan tugas atau proyek yang menantang untuk mengembangkan keterampilan dan pengetahuan. Menghadapi tantangan akan mendorong Anda untuk terus berkembang dan meningkatkan kemampuan Anda.Terakhir, berikan waktu istirahat yang cukup untuk merenung dan mengisi ulang energi. Istirahat yang cukup akan membantu menjaga keseimbangan dan memperoleh efektivitas belajar yang optimal.",
        "ISTP": "Untuk tipe MBTI ISTP, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, belajarlah melalui pengalaman langsung dengan terlibat dalam aktivitas praktis, eksperimen, atau proyek yang memungkinkan Anda menerapkan konsep dalam situasi nyata. Ini membantu Anda memperkuat pemahaman melalui pengalaman langsung.\n \
             Selanjutnya, gunakan metode belajar visual dan praktis dengan menggunakan visualisasi, diagram, atau model fisik untuk membantu memahami konsep. Latihan praktis dan permainan juga dapat digunakan untuk memperkuat pemahaman Anda secara interaktif.\n \
             Tantanglah diri Anda dengan menghadapi masalah dan tantangan nyata. Carilah proyek atau masalah yang menarik bagi Anda, dan temukan solusinya melalui eksperimen dan analisis. Ini akan merangsang pemikiran kritis dan kreativitas Anda.\n \
             Fleksibel dan beradaptasilah dengan metode belajar yang sesuai dengan kebutuhan Anda. Jelajahi berbagai sumber daya yang berbeda dan gunakan teknologi serta sumber daya online, seperti video tutorial, forum diskusi, atau aplikasi pembelajaran interaktif.\n \
             Berikan diri Anda waktu jeda dan waktu luang untuk beristirahat dan merefresh pikiran sebelum kembali belajar. Istirahat yang cukup membantu meningkatkan konsentrasi dan daya serap Anda.\n \
             Manfaatkan kemampuan Anda dalam mencari solusi secara mandiri. Gunakan waktu untuk berpikir secara kritis dan eksplorasi berbagai pendekatan dalam memecahkan masalah.\n \
             Terakhir, jadilah toleran terhadap risiko dan kegagalan. Jangan takut untuk mencoba pendekatan baru atau bereksperimen dalam belajar. Melalui keberanian ini, Anda dapat mengembangkan keterampilan dan pemahaman yang lebih mendalam.",
        "ISFP": "Untuk tipe MBTI ISFP, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, kaitkan pembelajaran dengan perasaan Anda. Sambungkan materi pembelajaran dengan pengalaman pribadi, kisah, atau seni untuk memperkuat pemahaman Anda secara emosional.\n \
             Selanjutnya, gunakan metode belajar yang kreatif. Anda dapat menggunakan seni visual, musik, drama, atau menulis untuk mempelajari konsep. Gunakan pengalaman seni untuk menggambarkan konsep dan memperluas pemahaman Anda.\n \
             Pilihlah lingkungan belajar yang nyaman dan menenangkan. Ciptakan tempat belajar yang sesuai dengan preferensi sensorik Anda, di mana Anda dapat fokus dan merasa terinspirasi.\n \
             Belajarlah melalui pengamatan dan pengalaman langsung. Amati dunia sekitar Anda dan ambil pelajaran dari pengalaman langsung. Libatkan diri dalam kegiatan lapangan, eksplorasi alam, atau perjalanan untuk memperkaya pemahaman Anda.\n \
             Berkolaborasilah dengan teman sekelas atau minta bantuan dari mentor. Diskusikan materi dengan teman sekelas Anda atau temui mentor untuk mendapatkan wawasan baru dan memperdalam pemahaman Anda.\n \
             Gunakan metode belajar reflektif. Sisihkan waktu untuk merenungkan materi yang dipelajari. Tulis jurnal, buat catatan reflektif, atau bermeditasi untuk menggali pemahaman yang lebih dalam.\n \
             Terimalah variasi dan kebebasan dalam belajar. Gunakan metode yang beragam, coba pendekatan baru, dan pilih mata pelajaran yang menarik minat Anda. Hal ini akan membuat proses belajar lebih menarik dan menyenangkan.\n \
             Akhirnya, kenali emosi dan kebutuhan diri sendiri saat belajar. Berikan waktu untuk self-care dan perhatikan keseimbangan antara belajar dan kegiatan lain yang memberi Anda kebahagiaan.",
        "INFP": "Untuk tipe MBTI INFP, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, ciptakan lingkungan belajar yang nyaman, tenang, dan inspiratif agar Anda dapat bekerja lebih baik. Pilihlah tempat belajar yang memenuhi kriteria ini untuk mendukung fokus dan kreativitas Anda.\n \
             Selanjutnya, selaraskan belajar Anda dengan nilai-nilai pribadi yang kuat. Pilih topik atau materi yang sesuai dengan minat dan nilai-nilai Anda, sehingga pembelajaran menjadi lebih bermakna dan relevan bagi Anda.\n \
             Manfaatkan kecenderungan kreatif dan artistik yang sering dimiliki oleh INFP. Gunakan metode belajar yang melibatkan aspek artistik, seperti membuat ilustrasi, menulis puisi, atau membuat catatan yang indah. Hal ini akan membantu meningkatkan pemahaman dan keterlibatan Anda dalam materi yang dipelajari.\n \
             Selain itu, manfaatkan pemikiran reflektif yang dalam yang dimiliki oleh INFP. Sisihkan waktu untuk merenung, memproses informasi, dan membuat koneksi pribadi dengan apa yang dipelajari. Dengan melakukan ini, Anda dapat memperdalam pemahaman Anda dan mengaitkan materi dengan pengalaman atau pemikiran pribadi.\n \
             Diskusikan materi dengan kelompok kecil atau mitra belajar yang sejalan dengan minat dan pandangan Anda. Ini akan membantu Anda memperluas wawasan, mendapatkan perspektif baru, dan meningkatkan pemahaman melalui diskusi dan kolaborasi.\n \
             Tuliskan dan susun catatan pribadi yang menggambarkan pemahaman dan refleksi pribadi Anda terhadap materi yang dipelajari. INFP sering menyukai tulisan dan proses pemikiran melalui kata-kata. Dengan menulis, Anda dapat mengorganisir pemahaman Anda secara lebih jelas dan mengembangkan gagasan-gagasan baru.\n \
             Berikan waktu untuk eksplorasi mandiri. INFP cenderung suka menggali topik secara mendalam. Gunakan waktu untuk belajar secara mandiri, mengeksplorasi sumber daya yang berbeda, dan mengikuti minat pribadi Anda. Ini akan memungkinkan Anda untuk mengejar minat khusus dan memperdalam pengetahuan Anda dalam bidang yang diminati.\n \
             Terakhir, jaga keseimbangan antara kebebasan dan struktur dalam belajar. INFP menghargai kreativitas dan kebebasan, tetapi juga bisa membutuhkan sedikit struktur. Temukan keseimbangan antara eksplorasi bebas dan kebutuhan untuk mengikuti jadwal atau rencana yang terstruktur, sehingga Anda dapat belajar dengan efektif dan tetap terorganisir.",
        "INTP": "Untuk tipe MBTI INTP, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, biarkan diri Anda mengeksplorasi topik dengan bebas tanpa terlalu terikat pada struktur yang ketat. Hal ini akan memungkinkan Anda untuk mengikuti minat dan penasaran Anda dalam pemahaman yang lebih mendalam.\n \
             Selanjutnya, gunakan pendekatan logika dan analisis dalam memahami konsep-konsep secara mendalam. INTP cenderung memiliki kecenderungan untuk berpikir secara logis, sehingga memanfaatkan kekuatan ini akan membantu Anda memahami dan menghubungkan konsep-konsep dengan lebih baik.\n \
             Manfaatkan teknologi dan sumber daya online untuk mendukung pembelajaran Anda. Gunakan video tutorial, forum diskusi, dan sumber daya online lainnya untuk mendapatkan informasi tambahan, belajar dari orang lain, dan berinteraksi dengan komunitas yang memiliki minat yang sama.\n \
             Diskusikan ide-ide dengan orang lain untuk mendapatkan sudut pandang baru dan melatih kemampuan verbal Anda. Berdiskusi dengan orang lain akan membantu Anda menguji dan mengartikulasikan pemahaman Anda, serta memperluas wawasan dengan mempertimbangkan sudut pandang yang berbeda.\n \
             Berikan waktu untuk merenungkan konsep-konsep secara mendalam. Sisihkan waktu untuk merenung dan memikirkan konsep-konsep dengan mendalam. Tuliskan pemikiran Anda dalam bentuk catatan atau jurnal untuk membantu mengorganisir dan mengklarifikasi pemahaman Anda.\n \
             Selain itu, cari tugas atau proyek yang menantang di luar kurikulum standar untuk mengembangkan keterampilan dan pengetahuan Anda. Beri diri tantangan intelektual yang mendorong Anda keluar dari zona nyaman dan mendorong pertumbuhan intelektual.\n \
             Jangan takut untuk mencoba pendekatan belajar yang berbeda dan jangan takut membuat kesalahan. Eksperimen dengan berbagai metode belajar dan terima kesalahan sebagai bagian dari proses belajar. Lihatlah kesalahan sebagai kesempatan untuk belajar dan berkembang.\n \
             Terakhir, tetap terbuka terhadap perubahan dan pemikiran baru. Jika ada bukti atau sudut pandang yang berbeda yang muncul, jadilah fleksibel dalam merevisi pemahaman Anda. Teruslah membuka pikiran Anda dan teruslah belajar dari berbagai perspektif yang berbeda.",
        "ESTP": "Untuk tipe MBTI ESTP, berikut adalah rekomendasi cara belajar yang dapat membantu Anda memperoleh pemahaman yang efektif. Pertama, pilih metode pembelajaran yang aktif dan praktis, seperti terlibat dalam diskusi, eksperimen, simulasi, atau proyek nyata. Dengan melakukan kegiatan praktis dan interaktif, Anda dapat memperkuat pemahaman Anda.\n \
             Selanjutnya, gunakan metode belajar yang beragam, seperti video, presentasi, diskusi kelompok, atau aplikasi pembelajaran interaktif. Dengan variasi dalam metode pembelajaran, Anda dapat mempertahankan minat dan keterlibatan dalam materi yang dipelajari.\n \
             Tetapkan tantangan dan tujuan yang jelas dalam belajar. ESTP cenderung termotivasi oleh tantangan, jadi tetapkan tujuan belajar yang spesifik dan tantang diri Anda sendiri untuk mencapainya.\n \
             Manfaatkan pengalaman langsung dalam pembelajaran, seperti magang, kerja praktek, atau proyek lapangan yang relevan dengan materi pelajaran. Melalui pengalaman praktis, Anda dapat memahami dan mengaitkan konsep dengan dunia nyata.\n \
             Cari kesempatan untuk berdiskusi dan berkolaborasi dengan orang lain, baik dengan teman sekelas maupun dengan bergabung dalam kelompok belajar. Diskusi dan kolaborasi akan membantu Anda memperoleh sudut pandang yang berbeda dan memperdalam pemahaman Anda.\n \
             Manfaatkan teknologi dan sumber daya digital dalam pembelajaran. Gunakan aplikasi, video tutorial, atau platform pembelajaran online untuk mendapatkan informasi dan sumber daya tambahan yang dapat meningkatkan pemahaman Anda.\n \
             Aturlah jadwal belajar yang fleksibel sesuai dengan gaya hidup Anda. ESTP cenderung menyukai fleksibilitas dan kebebasan dalam mengatur waktu. Dengan mengatur jadwal belajar yang sesuai dengan gaya hidup Anda, Anda dapat memanfaatkan waktu luang Anda secara efektif.\n \
             Terakhir, luangkan waktu untuk berpartisipasi dalam kegiatan fisik atau olahraga. Melibatkan diri dalam kegiatan fisik dapat membantu mengurangi kegelisahan dan meningkatkan fokus saat belajar.",
        "ESFP": "Untuk tipe MBTI ESFP, berikut adalah rekomendasi cara belajar yang dapat membantu Anda memperoleh pemahaman yang efektif. Pertama, pilih metode pembelajaran yang melibatkan emosi dan perasaan. Sambungkan materi pembelajaran dengan pengalaman pribadi, kisah, atau seni untuk memperkuat pemahaman Anda.\n \
             Selanjutnya, manfaatkan pendekatan kreatif dalam belajar. Gunakan seni visual, musik, tarian, atau permainan peran untuk menggambarkan konsep dan memperluas pemahaman Anda.\n \
             Ciptakan lingkungan belajar yang menenangkan dan nyaman. Pilih tempat yang ramai, penuh warna, dan sesuai dengan preferensi sensorik Anda.\n \
             ESFP cenderung belajar melalui interaksi sosial. Diskusikan materi dengan teman sekelas, bergabung dengan kelompok belajar, atau gunakan metode kolaboratif untuk memperdalam pemahaman Anda.\n \
             Ambil bagian dalam kegiatan praktis dan pengalaman langsung yang relevan dengan materi. Terlibat dalam eksperimen, proyek lapangan, atau magang akan memperkaya pemahaman Anda.\n \
             Manfaatkan teknologi dan sumber daya multimedia dalam pembelajaran. Gunakan video, audio, atau aplikasi pembelajaran interaktif untuk memperkaya pengalaman belajar Anda.\n \
             Aturlah jadwal belajar yang fleksibel sesuai dengan gaya hidup Anda. Manfaatkan fleksibilitas dan spontanitas yang Anda sukai untuk mengatur waktu belajar Anda.\n \
             Berikan penghargaan pada diri sendiri saat mencapai tujuan belajar. Selain itu, eksplorasi keberagaman topik dan metode belajar untuk menjaga minat dan motivasi Anda.",
        "ENFP": "Untuk tipe MBTI ENFP, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, eksplorasi berbagai topik yang menarik bagi Anda. Dengan melakukan eksplorasi, Anda dapat memperluas pengetahuan Anda dan menjaga minat Anda dalam belajar.\n \
             Selanjutnya, lakukan diskusi dengan orang lain untuk mendapatkan wawasan baru, melibatkan diri dalam perdebatan ide, dan melihat perspektif yang berbeda. Diskusi dengan orang lain akan membantu memperkaya pemahaman Anda dan mendorong pemikiran kritis.\n \
             Manfaatkan kreativitas dalam pembelajaran. Gunakan metode kreatif seperti visualisasi, mind map, atau bahkan membuat lagu atau puisi untuk membantu Anda mengingat informasi dengan cara yang menarik dan unik.\n \
             Manfaatkan teknologi dan sumber daya online untuk mendapatkan akses ke berbagai informasi yang beragam dan terkini. Teknologi dapat menjadi alat yang sangat berguna dalam memperluas pengetahuan Anda.\n \
             Terlibatlah dalam pembelajaran praktis dengan mencoba menerapkan materi yang dipelajari dalam proyek praktis, simulasi, atau situasi nyata. Ini akan membantu Anda memperkuat pemahaman dan mendapatkan pengalaman praktis.\n \
             Tantang diri Anda dengan tugas yang menantang di luar kurikulum. Hal ini akan membantu Anda menjaga minat dan mendorong pertumbuhan intelektual Anda.\n \
             Bergabunglah dengan kelompok belajar atau studi kelompok dengan orang-orang yang memiliki minat serupa. Melalui kolaborasi dalam kelompok, Anda dapat saling mendukung dan memperdalam pemahaman Anda.\n \
             Lakukan refleksi dan evaluasi diri secara teratur untuk melihat kemajuan belajar Anda, mengidentifikasi kekuatan dan kelemahan, serta membuat perbaikan yang diperlukan. Dengan refleksi yang baik, Anda dapat terus meningkatkan kualitas belajar Anda.",
        "ENTP": "Untuk tipe MBTI ENTP, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, jangan takut untuk mengeksplorasi dan mempelajari berbagai topik yang menarik bagi Anda. Hal ini akan membantu Anda memperluas pengetahuan dan wawasan.\n \
             Selanjutnya, aktif terlibat dalam diskusi dan berdebat dengan orang lain. Diskusi dengan orang lain akan membantu Anda mendapatkan perspektif baru, melatih pemikiran kritis, dan memperdalam pemahaman Anda.\n \
             Manfaatkan teknologi dan sumber daya online yang tersedia. Dengan menggunakan teknologi, Anda dapat mengakses beragam informasi dan sumber daya yang dapat memperkaya pemahaman Anda.\n \
             Gunakan pendekatan kreatif dan inovatif dalam belajar. Gunakan metode seperti mind map atau gambar untuk membantu Anda memahami konsep secara visual dan menggali pemikiran kreatif.\n \
             Terlibatlah dalam diskusi kelompok atau proyek kolaboratif. Melalui diskusi dan proyek kolaboratif, Anda dapat memperdalam pemahaman dan memperluas perspektif Anda dengan memanfaatkan kecerdasan kolektif.\n \
             Berikan diri tantangan intelektual di luar kurikulum. Cari tugas atau proyek yang menantang untuk mengembangkan pemikiran kritis dan menguji keterampilan Anda di luar batasan kurikulum.\n \
             Jaga fleksibilitas dan adaptabilitas dalam belajar. Bersikaplah fleksibel dan adaptif terhadap perubahan yang mungkin terjadi dalam pembelajaran. Hal ini akan membantu Anda menghadapi tantangan dengan lebih baik.\n \
             Lakukan evaluasi diri secara teratur untuk melihat kemajuan belajar Anda dan identifikasi area yang perlu ditingkatkan. Dengan melakukan evaluasi dan refleksi, Anda dapat terus meningkatkan kualitas belajar Anda.",
        "ESTJ": "Untuk tipe MBTI ESTJ, berikut adalah rekomendasi cara belajar yang dapat membantu Anda memperoleh pemahaman yang efektif. Pertama, buat jadwal belajar yang terstruktur dan patuhi dengan disiplin. ESTJ cenderung menghargai keteraturan dan struktur dalam pembelajaran.\n \
             Selanjutnya, manfaatkan metode belajar yang praktis. Terlibatlah dalam latihan, proyek-proyek, atau simulasi yang memungkinkan Anda menerapkan konsep dalam situasi nyata.\n \
             Gunakan sumber daya resmi dan teks bahan ajar. Gunakan buku teks, materi ajar yang terpercaya, dan sumber daya online yang terverifikasi untuk mendapatkan pemahaman yang mendalam.\n \
             Melibatkan diri dalam diskusi dan perdebatan. Diskusikan materi dengan orang lain, ikuti perdebatan, atau bergabung dalam kelompok studi untuk mendapatkan wawasan baru dan mempertajam pemahaman.\n \
             Gunakan metode visual dan diagram. Visualisasi, diagram, atau peta konsep dapat membantu Anda memahami hubungan antara konsep-konsep yang kompleks dengan lebih baik.\n \
             Pastikan Anda memahami dan mengikuti instruksi dengan seksama. ESTJ menghargai aturan dan petunjuk yang jelas, jadi pastikan Anda memahami dan mengikuti instruksi dengan teliti.\n \
             Buat catatan yang rapi dan terstruktur saat belajar. Membuat catatan yang sistematis membantu Anda memperjelas dan mengingat informasi dengan lebih baik.\n \
             Lakukan evaluasi diri secara objektif untuk melihat kemajuan belajar Anda dan mengidentifikasi area yang perlu diperbaiki. Tetapkan tujuan yang jelas dan ukur kemajuan Anda secara teratur.",
        "ESFJ": "Untuk tipe MBTI ESFJ, berikut adalah rekomendasi cara belajar yang dapat membantu Anda memperoleh pemahaman yang efektif. Pertama, manfaatkan pembelajaran melalui interaksi sosial. Bergabunglah dengan kelompok belajar, diskusikan materi dengan teman sekelas, atau minta bantuan dari mentor untuk memperdalam pemahaman Anda.\n \
             Selanjutnya, gunakan metode belajar yang terstruktur. Buat jadwal belajar, mengorganisasi catatan, atau gunakan sistem catatan yang teratur. Rencanakan langkah-langkah yang jelas dalam proses pembelajaran Anda.\n \
             Ciptakan lingkungan belajar yang nyaman dan ramah. Pastikan tempat belajar Anda bersih, nyaman, dan bebas dari gangguan agar Anda dapat fokus pada pembelajaran. Sesuaikan lingkungan belajar dengan preferensi sensorik Anda.\n \
             Gunakan pendekatan belajar yang praktis. Terlibatlah dalam kegiatan praktis yang memungkinkan Anda menerapkan konsep dalam situasi nyata. Gunakan studi kasus, simulasi, atau proyek praktis untuk meningkatkan pemahaman Anda.\n \
             Manfaatkan sumber daya dan materi yang terstruktur. Gunakan buku teks, panduan, atau sumber daya terstruktur lainnya untuk mempelajari materi secara sistematis dan terarah.\n \
             Rencanakan waktu belajar dengan baik. Buat jadwal belajar yang teratur dan konsisten. Aturlah waktu yang cukup untuk setiap topik atau tugas yang perlu dipelajari.\n \
             Gunakan catatan dan pemetaan konsep. Buat catatan yang rapi dan pemetaan konsep yang membantu Anda mengorganisasi informasi dan menghubungkan konsep yang berbeda.\n \
             Terima umpan balik dan lakukan evaluasi secara berkala. Mintalah umpan balik dari guru atau teman sekelas untuk memperbaiki pemahaman Anda. Lakukan evaluasi secara berkala untuk mengukur kemajuan Anda dalam pembelajaran.",
        "ENFJ": "Untuk tipe MBTI ENFJ, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, manfaatkan kolaborasi dan interaksi sosial dalam pembelajaran. Carilah kesempatan untuk bekerja dalam kelompok atau berdiskusi dengan orang lain. Hal ini akan membantu Anda mendapatkan wawasan dan perspektif baru.\n \
             Selanjutnya, manfaatkan kemampuan komunikasi yang baik sebagai ENFJ. Gunakan kemampuan ini untuk menjelaskan konsep kepada orang lain atau membantu teman sekelas yang membutuhkan bantuan. Mengajar orang lain juga dapat memperkuat pemahaman Anda.\n \
             Coba kaitkan materi yang dipelajari dengan situasi atau contoh dalam kehidupan nyata. Dengan menghubungkan materi dengan realitas, Anda dapat memahami dan mengingat informasi dengan lebih baik.\n \
             Bangun hubungan dengan dosen atau mentor yang dapat memberikan panduan dan saran dalam proses belajar. Mereka dapat memberikan wawasan tambahan dan membantu memperkaya pemahaman Anda.\n \
             Manfaatkan metode visual dan audio dalam pembelajaran. Gunakan diagram, peta konsep, atau rekaman audio untuk membantu memvisualisasikan dan memahami konsep yang kompleks.\n \
             Terlibatlah dalam proyek-proyek sosial yang memiliki dampak sosial atau membantu orang lain. Menggabungkan pembelajaran dengan tujuan sosial akan memberikan motivasi tambahan dan memperkuat pemahaman Anda.\n \
             Buat jadwal belajar yang teratur dan disiplin untuk mengelola waktu dengan efisien dan menghindari penundaan. Disiplin dalam mengatur jadwal akan membantu Anda tetap fokus dan efektif dalam belajar.\n \
             Lakukan evaluasi diri secara teratur untuk melihat kemajuan belajar Anda. Refleksikan apa yang telah Anda pelajari dan identifikasi area yang perlu diperbaiki. Buat rencana tindakan untuk meningkatkan pemahaman Anda secara terus-menerus.",
        "ENTJ": "Untuk tipe MBTI ENTJ, berikut adalah rekomendasi cara belajar yang dapat membantu Anda dalam memperoleh pemahaman yang efektif. Pertama, rencanakan belajar Anda dengan tujuan yang jelas. Buatlah rencana belajar yang terstruktur dengan tujuan spesifik dan langkah-langkah terperinci untuk mencapainya.\n \
             Selanjutnya, fokus pada aplikasi praktis dari konsep-konsep yang dipelajari. Temukan cara untuk menghubungkan konsep dengan situasi dunia nyata atau contoh-contoh yang relevan. Hal ini akan membantu Anda memahami dan mengingat informasi dengan lebih baik.\n \
             Cari sumber belajar yang efisien yang ringkas, langsung ke inti, dan memadukan informasi secara efektif. Pilihlah sumber belajar yang memberikan pemahaman yang komprehensif dalam waktu yang efisien.\n \
             Diskusikan ide-ide dengan orang lain untuk mendapatkan perspektif baru dan melatih kemampuan verbal Anda. Diskusi dengan orang lain akan membantu Anda melihat sudut pandang yang berbeda dan memperkaya pemahaman Anda.\n \
             Manfaatkan metode pembelajaran aktif dengan terlibat dalam kegiatan praktis. Terapkan konsep-konsep yang dipelajari dalam situasi nyata atau latihan-latihan praktis untuk memperkuat pemahaman Anda.\n \
             Ciptakan lingkungan belajar yang produktif dengan membuat area belajar yang terorganisir dan minim gangguan. Dengan lingkungan yang kondusif, Anda dapat fokus sepenuhnya pada pembelajaran.\n \
             Ambil inisiatif dalam kelompok belajar dengan berperan aktif, mengorganisir diskusi, dan membantu mengarahkan kelompok menuju tujuan belajar. Ini akan memperkuat pemahaman Anda dan membangun kemampuan kepemimpinan.\n \
             Lakukan evaluasi dan refleksi secara teratur untuk melihat kemajuan belajar Anda dan mengidentifikasi area yang perlu diperbaiki. Evaluasi diri akan membantu Anda terus meningkatkan kualitas belajar Anda."
    }


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    scores = {
        'E': 0,
        'I': 0,
        'S': 0,
        'N': 0,
        'T': 0,
        'F': 0,
        'J': 0,
        'P': 0,
        'E2': 0,
        'I2': 0,
        'S2': 0,
        'N2': 0,
        'T2': 0,
        'F2': 0,
        'J2': 0,
        'P2': 0
    }

    for key in data.keys():
        scores[key] = int(data[key])

    input_data = []
    for key in scores.keys():
        input_data.append(scores[key])

    total = sum(input_data)
    input_data = [score / total for score in input_data]

    prediction = model.predict([input_data])
    mbti_types = ['ESTJ', 'ISTJ', 'ENTJ', 'INTJ', 'ESTP', 'ISTP', 'ENTP', 'INTP', 'ESFJ', 'ISFJ', 'ENFJ', 'INFJ', 'ESFP', 'ISFP', 'ENFP', 'INFP']
    mbti_index = prediction.argmax()
    mbti_type = mbti_types[mbti_index]
    mbti_descriptions = mbti_description[mbti_index]
    mbti_recommendations = mbti_recommend[mbti_index]

    result = {
        'MBTI Anda': mbti_type,
        'Deskripsi MBTI Anda': mbti_descriptions,
        'Rekomendasi Belajar': mbti_recommendations
    }

    return jsonify(result)

@app.route('/questions', methods=['GET'])
def get_questions():
    return jsonify(questions)

if __name__ == '__main__':
    app.run()
