# GitHub'a Yükleme Talimatları

## 1. GitHub'da Yeni Repo Oluştur

1. https://github.com/new adresine git
2. Repository name: `voice-of-customer-nlp`
3. Description: `NLP pipeline that transforms 5,000+ app reviews into actionable product insights using topic modeling and interactive dashboard`
4. Public seç
5. **Initialize this repository with a README:** SEÇME (zaten var)
6. **Create repository** tıkla

## 2. Local'den GitHub'a Yükle

Bu klasörde terminal aç ve şu komutları çalıştır:

```bash
# Git başlat
cd "c:\Users\Emine\OneDrive\Masaüstü\Google Project\voice_of_customer"
git init

# Dosyaları ekle
git add .

# İlk commit
git commit -m "Initial commit: Voice of Customer NLP pipeline with Streamlit dashboard"

# GitHub repo'nuza bağla (YOURUSERNAME yerine kendi kullanıcı adını yaz)
git remote add origin https://github.com/YOURUSERNAME/voice-of-customer-nlp.git

# GitHub'a yükle
git branch -M main
git push -u origin main
```

## 3. README'ye Screenshot/GIF Ekle

### Screenshot almak için:
1. Dashboard'u çalıştır: `streamlit run app.py`
2. Windows Snipping Tool (Win+Shift+S) ile screenshot al
3. `screenshots/` klasörüne kaydet

### GIF oluşturmak için (opsiyonel):
- ScreenToGif indir: https://www.screentogif.com/
- Dashboard'u kullanırken kayıt yap
- `screenshots/dashboard-demo.gif` olarak kaydet

### README'ye ekle:
```markdown
## 📊 Demo

![Dashboard Demo](screenshots/dashboard-demo.gif)

*Interactive dashboard showing topic distribution and keyword search*
```

## 4. README'yi Güncelle

Son kontroller:
- [ ] Screenshots/GIF eklendi mi?
- [ ] Requirements.txt güncel mi?
- [ ] .gitignore doğru çalışıyor mu? (venv, data dosyaları yüklenmesin)
- [ ] GitHub repo linki doğru mu?

## 5. Son İyileştirmeler

GitHub repo'da:
- About kısmına description ekle
- Topics ekle: `nlp`, `topic-modeling`, `streamlit`, `data-science`, `python`
- Website kısmına Streamlit share linki ekle (deploy edersen)

## 6. LinkedIn'de Paylaş

Örnek post:
```
🎯 New Project: Voice of Customer Analysis

Built an NLP pipeline that transforms 5,000+ user reviews into actionable insights:
• Topic modeling to categorize feedback (Bugs, Features, UI/UX)
• Interactive dashboard for trend analysis
• Real-time keyword search

Tech: Python, BERTopic, Streamlit

Check it out: [GitHub link]

#DataScience #NLP #MachineLearning
```
