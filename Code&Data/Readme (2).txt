
2) INSTRUCTIUNI DE UTILIZARE A CODULUI

Resurse necesare:
- Google Colab (recomandat)
- Setul de date: NoiseLibriSpeech/test-clean
- Librării Python:
    - transformers
    - torchaudio
    - datasets
    - jiwer
    - noisereduce
    - scipy
    - matplotlib
    - tqdm
    - pandas
    - numpy

Instalare:
- În Google Colab,se rulează următoarele comenzi:
  !pip install transformers torchaudio datasets jiwer noisereduce

Etape de lucru:
1. Se montează Google Drive pentru a accesa setul de date:

   from google.colab import drive
   drive.mount('/content/drive')

2. Se încarcă modelul Wav2Vec2.0 pre-antrenat de la Facebook pentru recunoaștere vocală.
3. Se definesc trei metode de reducere a zgomotului:
Noisereduce (bazat pe estimarea zgomotului)
Filtru trece-jos (low-pass)
Spectral gate (cu estimare a zgomotului din primele 0.5 secunde)

4.Se aplică aceste metode asupra fișierelor .wav din NoiseLibriSpeech/test-clean.

Pentru fiecare fișier:

Se calculează SNR înainte și după reducerea zgomotului
Se face inferență ASR cu Wav2Vec2
Se compară transcrierea cu eticheta de referință (.trans.txt) folosind metricile WER și CER
Se salvează rezultatele:
Transcrierile generate: transcrieri_<metoda>.csv
Metricile WER/CER și SNR: wer_results_<metoda>.csv
Se generează grafice comparative pentru:
SNR înainte vs. după
WER per metodă
CER per metodă

5. Rezultate:
Fișierele CSV și graficele sunt salvate în directorul de lucru
Se poate evalua eficiența fiecărei metode de reducere a zgomotului pe performanța ASR
