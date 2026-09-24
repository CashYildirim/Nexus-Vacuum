"""
NEXUS-VAKUM | Adim 3: Tohum taramasi ve kayan pencere analizi
=============================================================
IKI SORUNU COZER

(1) TEK TOHUM SORUNU
    Eski calismada tek bir random_state=42 kosusu raporlanmisti.
    Burada 10 farkli tohumla veri uretilip model egitiliyor;
    ortalama +/- %95 guven araligi veriliyor. Makalede sayilar
    boyle raporlanmalidir.

(2) "CANLI DOGRULUK ARALIGI" YANILGISI
    Eski makale "%88.5 - %93.2" araligini dinamik uzay kosullarinda
    kararlilik kaniti olarak sunuyordu. Bu betik gosteriyor ki:
    veri akisi bagimsiz ve ayni dagilimdan geldigi icin, N ornekli
    bir pencerenin dogrulugu p basarili binom dagilimindan gelir ve
    beklenen dalgalanma +/- 2*sqrt(p(1-p)/N)'dir.
    Yani o aralik bir BULGU degil, ISTATISTIKSEL KACINILMAZLIKTIR.

    Ayrica eski app.py min/max degerlerini KUMULATIF gecmis uzerinden
    aliyordu; paneli ne kadar uzun calistirirsaniz min o kadar duser.
    Yani raporlanan aralik, uygulamanin kac dakika acik kaldiginin
    fonksiyonuydu -- tekrar uretilemez. Burasi duzeltiliyor.

Calistirma : python 03_tohum_ve_pencere.py
Cikti      : 03_sonuclar.json
"""

import json
import importlib.util
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- Ureteç dosyasini bu betikle AYNI klasorde ara. ---
# Boylece hangi klasorden calistirdiginiz ve dosyayi nasil
# adlandirdiginiz onemli olmaz.
BURASI = Path(__file__).resolve().parent
_ADAYLAR = ["01_veri_uretimi.py", "veriuretimi.py", "veri_uretimi.py"]
URETEC = next((BURASI / a for a in _ADAYLAR if (BURASI / a).exists()), None)

if URETEC is None:
    raise FileNotFoundError(
        f"Ureteç dosyasi bulunamadi. Su klasorde arandi: {BURASI}\n"
        f"Aranan adlar: {_ADAYLAR}\n"
        "Ureteç dosyasinin adini degistirdiyseniz, yukaridaki "
        "_ADAYLAR listesine ekleyin ya da bu betikle AYNI klasore "
        "kopyalayin."
    )

spec = importlib.util.spec_from_file_location("gen", URETEC)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def tohum_taramasi(tohumlar=range(10)):
    acc, mf1, bayes = [], [], []
    for s in tohumlar:
        cfg = dict(gen.CFG)
        cfg["seed"] = int(s)
        df, rapor = gen.uret(cfg, kaydet=False)
        X = df.drop(columns=["Ariza_Durumu"])
        y = df["Ariza_Durumu"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        m = RandomForestClassifier(n_estimators=100, max_depth=12,
                                   class_weight="balanced",
                                   random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        acc.append(accuracy_score(yte, p))
        mf1.append(f1_score(yte, p, average="macro", zero_division=0))
        bayes.append(rapor["bayes_tavani_etiket_gurultulu"])
    return np.array(acc), np.array(mf1), np.array(bayes)


def pencere_analizi(N_pencere=500, n_pencere=400, p=None, seed=0):
    """Kayan pencere dogrulugunun beklenen dalgalanmasi."""
    rng = np.random.default_rng(seed)
    ornek = rng.binomial(N_pencere, p, n_pencere) / N_pencere
    sigma_teorik = np.sqrt(p * (1 - p) / N_pencere)
    return {
        "p": round(float(p), 4),
        "N_pencere": N_pencere,
        "sigma_teorik": round(float(sigma_teorik), 5),
        "teorik_2sigma_band": [round(float(p - 2 * sigma_teorik), 4),
                               round(float(p + 2 * sigma_teorik), 4)],
        "simulasyon_min": round(float(ornek.min()), 4),
        "simulasyon_max": round(float(ornek.max()), 4),
        "simulasyon_std": round(float(ornek.std()), 5),
    }


def main():
    print("=" * 74)
    print("TOHUM TARAMASI  (10 bagimsiz veri uretimi + egitim)")
    print("=" * 74)
    acc, mf1, bayes = tohum_taramasi()

    def ozet(v, ad):
        m, s = v.mean(), v.std(ddof=1)
        ga = 1.96 * s / np.sqrt(len(v))
        print(f"  {ad:<22} {m:.4f}  +/- {ga:.4f}  "
              f"(std {s:.4f}, min {v.min():.4f}, max {v.max():.4f})")
        return {"ortalama": round(float(m), 4), "std": round(float(s), 4),
                "ga95_yari_genislik": round(float(ga), 4),
                "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    r_acc = ozet(acc, "Test dogrulugu")
    r_f1 = ozet(mf1, "Makro F1")
    r_by = ozet(bayes, "Bayes tavani")

    print(f"\n  Tavana uzaklik: {(bayes.mean()-acc.mean())*100:+.2f} puan")
    print("  Model Bayes tavanina yapisiksa, ek model karmasikligi")
    print("  performans kazandirmaz. Bunu makalede soyleyin.")

    print("\n" + "=" * 74)
    print("KAYAN PENCERE ANALIZI")
    print("=" * 74)
    pen = {}
    for N in [30, 100, 500]:
        r = pencere_analizi(N_pencere=N, p=float(acc.mean()))
        pen[f"N={N}"] = r
        lo, hi = r["teorik_2sigma_band"]
        print(f"  N={N:<5} teorik +/-2sigma band: "
              f"[%{lo*100:.1f}, %{hi*100:.1f}]   "
              f"simulasyon: [%{r['simulasyon_min']*100:.1f}, "
              f"%{r['simulasyon_max']*100:.1f}]")

    print("\n  YORUM (makaleye yazilacak):")
    print("  Kayan pencere dogrulugundaki dalgalanma, IID bir akista")
    print("  binom ornekleme gurultusuyle aciklanir. Bu aralik model")
    print("  kararliliginin kaniti DEGILDIR. Eski calismadaki N=30")
    print("  penceresi bu yuzden cok genis bir aralik uretiyordu.")
    print("  Cozum: araligi bir GUVEN ARALIGI olarak raporlayin ve")
    print("  pencere boyutunu acikca belirtin.")

    with open("03_sonuclar.json", "w") as f:
        json.dump({"tohum_taramasi": {"dogruluk": r_acc, "makro_f1": r_f1,
                                      "bayes_tavani": r_by},
                   "pencere_analizi": pen}, f, indent=2, ensure_ascii=False)
    print("\nKaydedildi -> 03_sonuclar.json")


if __name__ == "__main__":
    main()