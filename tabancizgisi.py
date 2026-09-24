"""
NEXUS-VAKUM | Adim 2: Taban cizgisi karsilastirmasi ve sizinti analizi
======================================================================
Makalenin MERKEZI IDDIASI su: "olasiliksal ML, esik-tabanli deterministik
mantiktan ustundur". Eski calismada bu iddiayi destekleyen HICBIR sayi
yoktu -- deterministik kural hic kurulup calistirilmamisti.

Bu betik sunlari uretir:
  1. Deterministik esik kurali (makalede elestirilen yontem)
  2. Lojistik regresyon
  3. Karar agaci (derinlik 3, yorumlanabilir)
  4. Random Forest
  -> ayni test kumesinde, ayni metriklerle

  5. McNemar testi: RF ile esik kurali arasindaki fark istatistiksel
     olarak anlamli mi?
  6. Bootstrap %95 guven araliklari
  7. Karisiklik matrisi (uzay guvenligi acisindan kritik: hangi ariza
     hangi sinifa kariyor?)
  8. SIZINTI ABLASYONU: turetilmis oznitelikler (Hesaplanan_Sicaklik_C,
     Tork_Stres_Katsayisi) cikarildiginda ne oluyor?
  9. Belirsizlik kapisi analizi: guven < 0.80 ise otonom karari askiya
     alma politikasinin gercek maliyeti/faydasi.

Calistirma : python 02_taban_cizgisi.py   (once 01'i calistirin)
Cikti      : 02_sonuclar.json , konsol tablolari
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from scipy.stats import chi2

SINIFLAR = ["Saglikli", "Termal", "Adim Kacirma", "Voltaj"]
TUREV_OZNITELIK = ["Hesaplanan_Sicaklik_C", "Tork_Stres_Katsayisi"]

# Deterministik esikler (eski app.py'deki mantikla ayni ruhta)
ESIK = {"T_C": 100.0, "S": 0.95, "V": 5.8, "I": 0.35}


def esik_kurali(X, esik=ESIK):
    """Klasik uzay sistemlerinde kullanilan deterministik mantik.
    Oncelik sirasi: termal > mekanik > elektriksel."""
    T = X["Hesaplanan_Sicaklik_C"].values
    S = X["Tork_Stres_Katsayisi"].values
    V = X["Calisma_Voltaji_V"].values
    I = X["Faz_Akimi_A"].values
    y = np.zeros(len(X), dtype=int)
    y[(V < esik["V"]) & (I > esik["I"])] = 3
    y[S > esik["S"]] = 2
    y[T > esik["T_C"]] = 1
    return y


def bootstrap_ci(y_true, y_pred, metrik=accuracy_score, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    skor = [metrik(y_true[i], y_pred[i]) for i in
            (rng.integers(0, N, N) for _ in range(n))]
    return np.percentile(skor, [2.5, 97.5])


def mcnemar(y_true, pred_a, pred_b):
    """A ve B modellerinin hata desenlerini karsilastirir.
    b = A dogru & B yanlis, c = A yanlis & B dogru."""
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return b, c, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)      # sureklilik duzeltmeli
    return b, c, float(1 - chi2.cdf(stat, 1))


def yanlis_alarm_orani(y_true, y_pred):
    """Gercekte saglikli iken ariza ilan etme orani.
    Uzay gorevlerinde gereksiz kapatmanin dogrudan maliyeti budur."""
    saglikli = y_true == 0
    return float((y_pred[saglikli] != 0).mean())


def kacirilan_ariza_orani(y_true, y_pred):
    """Gercekte ariza varken 'saglikli' deme orani. Guvenlik kritigi."""
    arizali = y_true != 0
    return float((y_pred[arizali] == 0).mean())


def main():
    df = pd.read_csv("nexus_vakum_veri.csv")
    X = df.drop(columns=["Ariza_Durumu"])
    y = df["Ariza_Durumu"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    modeller = {
        "Lojistik Regresyon": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=42)),
        "Karar Agaci (d=3)": DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1),
    }

    tahminler = {"Esik Kurali (deterministik)": esik_kurali(Xte)}
    for ad, m in modeller.items():
        m.fit(Xtr, ytr)
        tahminler[ad] = m.predict(Xte)

    # =================================================================
    # TABLO: TABAN CIZGISI KARSILASTIRMASI
    # =================================================================
    print("=" * 84)
    print("TABLO A  -  TABAN CIZGISI KARSILASTIRMASI (test n=%d)" % len(yte))
    print("=" * 84)
    print(f"{'Yontem':<30}{'Dogruluk':>12}{'%95 GA':>18}"
          f"{'Makro F1':>11}{'Yanlis Alarm':>14}{'Kacirilan':>11}")
    print("-" * 84)

    sonuc = {}
    for ad, p in tahminler.items():
        acc = accuracy_score(yte, p)
        lo, hi = bootstrap_ci(yte, p)
        mf1 = f1_score(yte, p, average="macro", zero_division=0)
        fa = yanlis_alarm_orani(yte, p)
        ka = kacirilan_ariza_orani(yte, p)
        sonuc[ad] = {"dogruluk": round(acc, 4),
                     "ga95": [round(lo, 4), round(hi, 4)],
                     "makro_f1": round(mf1, 4),
                     "yanlis_alarm": round(fa, 4),
                     "kacirilan_ariza": round(ka, 4)}
        print(f"{ad:<30}{acc*100:>11.2f}%"
              f"{f'[{lo*100:.1f}, {hi*100:.1f}]':>18}"
              f"{mf1:>11.3f}{fa*100:>13.2f}%{ka*100:>10.2f}%")
    print("=" * 84)

    # =================================================================
    # McNEMAR
    # =================================================================
    print("\nTABLO B  -  McNEMAR TESTI (Random Forest'a karsi)")
    print("-" * 70)
    rf = tahminler["Random Forest"]
    mcn = {}
    for ad, p in tahminler.items():
        if ad == "Random Forest":
            continue
        b, c, pv = mcnemar(yte, p, rf)
        mcn[ad] = {"sadece_A_dogru": b, "sadece_RF_dogru": c,
                   "p_degeri": round(pv, 6)}
        anl = "ANLAMLI" if pv < 0.05 else "anlamli degil"
        print(f"  {ad:<30} b={b:4d}  c={c:4d}  p={pv:.2e}  -> {anl}")

    # =================================================================
    # KARISIKLIK MATRISI
    # =================================================================
    print("\nTABLO C  -  KARISIKLIK MATRISI (Random Forest)")
    print("-" * 70)
    cm = confusion_matrix(yte, rf, labels=[0, 1, 2, 3])
    print(f"{'gercek \\ tahmin':<18}" + "".join(f"{s:>15}" for s in SINIFLAR))
    for i, s in enumerate(SINIFLAR):
        print(f"{s:<18}" + "".join(f"{v:>15d}" for v in cm[i]))

    print("\n" + classification_report(yte, rf, labels=[0, 1, 2, 3],
                                       target_names=SINIFLAR, zero_division=0,
                                       digits=3))

    # =================================================================
    # SIZINTI ABLASYONU
    # =================================================================
    print("=" * 84)
    print("TABLO D  -  SIZINTI ABLASYONU")
    print("  Turetilmis oznitelikler etiket ureten fonksiyonun dogrudan")
    print("  girdileridir. Cikarildiklarinda performans ne kadar duser?")
    print("=" * 84)
    ablasyon = {}
    for ad, sut in [("Tum oznitelikler", list(X.columns)),
                    ("Turevler CIKARILDI",
                     [c for c in X.columns if c not in TUREV_OZNITELIK])]:
        m = RandomForestClassifier(n_estimators=100, max_depth=12,
                                   class_weight="balanced",
                                   random_state=42, n_jobs=-1)
        m.fit(Xtr[sut], ytr)
        acc = m.score(Xte[sut], yte)
        mf1 = f1_score(yte, m.predict(Xte[sut]), average="macro",
                       zero_division=0)
        ablasyon[ad] = {"dogruluk": round(acc, 4), "makro_f1": round(mf1, 4),
                        "oznitelik_sayisi": len(sut)}
        print(f"  {ad:<26} n_oz={len(sut)}  dogruluk=%{acc*100:.2f}  "
              f"makroF1={mf1:.3f}")
    fark = (ablasyon["Tum oznitelikler"]["dogruluk"]
            - ablasyon["Turevler CIKARILDI"]["dogruluk"])
    print(f"\n  -> Turetilmis ozniteliklerin katkisi: {fark*100:.2f} puan")
    print("     Bu farki makalede 'Sinirliliklar' bolumunde tartisin.")

    # =================================================================
    # BELIRSIZLIK KAPISI
    # =================================================================
    print("\n" + "=" * 84)
    print("TABLO E  -  BELIRSIZLIK KAPISI (makalenin en ozgun fikri)")
    print("  Politika: max(predict_proba) < esik ise otonom karar askiya alinir")
    print("=" * 84)
    rf_model = modeller["Random Forest"]
    prob = rf_model.predict_proba(Xte).max(axis=1)
    print(f"{'Esik':>8}{'Askiya alinan':>16}{'Kalan dogruluk':>18}"
          f"{'Askidakilerde dogruluk':>25}")
    print("-" * 70)
    kapi = {}
    for t in [0.0, 0.60, 0.70, 0.80, 0.90, 0.95]:
        gecen = prob >= t
        if gecen.sum() == 0:
            continue
        acc_g = accuracy_score(yte[gecen], rf[gecen])
        acc_a = (accuracy_score(yte[~gecen], rf[~gecen])
                 if (~gecen).sum() > 0 else float("nan"))
        kapi[str(t)] = {"askiya_alinan_oran": round(1 - gecen.mean(), 4),
                        "kalan_dogruluk": round(acc_g, 4),
                        "askidaki_dogruluk": None if np.isnan(acc_a)
                        else round(acc_a, 4)}
        print(f"{t:>8.2f}{(1-gecen.mean())*100:>15.2f}%{acc_g*100:>17.2f}%"
              f"{('  -' if np.isnan(acc_a) else f'{acc_a*100:.2f}%'):>25}")
    print("\n  Bu tablo makalenizin en degerli katkisidir: belirsizligi")
    print("  filtrelemek dogrulugu ne kadar artiriyor, kac ornek pahasina?")

    # =================================================================
    # CAPRAZ DOGRULAMA
    # =================================================================
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_depth=12,
                               class_weight="balanced", random_state=42,
                               n_jobs=-1),
        Xtr, ytr, cv=StratifiedKFold(10, shuffle=True, random_state=42))
    print("\n" + "=" * 84)
    print(f"10-kat capraz dogrulama: %{cv.mean()*100:.2f} (+/- %{cv.std()*100:.2f})")
    rf_model.fit(Xtr, ytr)
    print(f"Egitim dogrulugu       : %{rf_model.score(Xtr, ytr)*100:.2f}")
    print(f"Test dogrulugu         : %{rf_model.score(Xte, yte)*100:.2f}")
    print(f"Egitim-test farki      : "
          f"{(rf_model.score(Xtr, ytr)-rf_model.score(Xte, yte))*100:.2f} puan")
    print("=" * 84)

    with open("02_sonuclar.json", "w") as f:
        json.dump({"taban_cizgisi": sonuc, "mcnemar": mcn,
                   "karisiklik_matrisi": cm.tolist(),
                   "ablasyon": ablasyon, "belirsizlik_kapisi": kapi,
                   "cv_ortalama": round(float(cv.mean()), 4),
                   "cv_std": round(float(cv.std()), 4)},
                  f, indent=2, ensure_ascii=False)
    print("\nSonuclar kaydedildi -> 02_sonuclar.json")

    # =================================================================
    # MODELI KAYDET  (04_app.py bunlari yukler)
    # =================================================================
    joblib.dump(rf_model, "nexus_model.pkl")
    joblib.dump(list(X.columns), "model_sutunlari.pkl")
    print("Model kaydedildi      -> nexus_model.pkl")
    print("Sutun duzeni kaydedildi-> model_sutunlari.pkl")
    print(f"  Oznitelik sirasi: {list(X.columns)}")


if __name__ == "__main__":
    main()