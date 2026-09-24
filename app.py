"""
NEXUS-VAKUM | Adim 4: Duzeltilmis canli telemetri paneli
=========================================================
ESKI app.py'DEKI KRITIK TUTARSIZLIKLAR

  Konu              | Eski egitim ureteci        | Eski app.py
  ------------------|----------------------------|---------------------------
  Sicaklik modeli   | (P_kayip*yogunluk)         | P_kayip * 11.2
                    |   *(1.5-P*100)*12.5        | yogunluk ve vakum YOK
  Etiketleme        | stokastik sigmoid          | SERT ESIK (100 / .75 / 5.5)
  Ortam sicakligi   | -50 .. 80 C                | -10 .. 60 C
  Voltaj            | 5.0 .. 9.5 V               | 5.2 .. 9.5 V
  Pencere           | makalede N=500             | kodda [-30:]
  Min/Max           | -                          | KUMULATIF gecmis

Yani model, egitildiginden BASKA bir etiketleme kuralina karsi
olculuyordu. Makaledeki "%88.5 - %93.2 canli dogruluk" bu yuzden
gecersizdi. Ustelik min/max kumulatif oldugu icin raporlanan aralik,
panelin kac dakika acik kaldiginin fonksiyonuydu.

BU SURUMDE
  * Fizik ve etiketleme 01_veri_uretimi.py'den AYNEN ithal ediliyor.
    Tek kaynak ilkesi: simulasyon mantigi tek yerde tanimli.
  * GERCEK kayan pencere (deque, maxsize=N).
  * Binom guven araligi bandi da cizdiriliyor; boylece dalgalanmanin
    ornekleme gurultusu oldugu gorsel olarak belli oluyor.
  * Belirsizlik kapisi (guven < esik -> otonom karar askiya alinir)
    ayri bir metrik olarak izleniyor.
  * Gecmis sinirli; bellek sizintisi yok.

Calistirma : streamlit run 04_app.py
Gereksinim : 01_veri_uretimi.py ayni klasorde, nexus_model.pkl egitilmis
"""

from collections import deque

import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib

import importlib.util
from pathlib import Path

# --- Tum yollar bu dosyanin bulundugu klasore gore cozulur. ---
# Boylece streamlit'i hangi dizinden calistirdiginiz onemli olmaz.
BURASI = Path(__file__).resolve().parent
_ADAYLAR = ["01_veri_uretimi.py", "veriuretimi.py", "veri_uretimi.py"]
URETEC = next((BURASI / a for a in _ADAYLAR if (BURASI / a).exists()), None)
MODEL_YOLU = BURASI / "nexus_model.pkl"
SUTUN_YOLU = BURASI / "model_sutunlari.pkl"

if URETEC is None:
    st.error(
        f"Ureteç dosyasi bulunamadi. Su klasorde arandi: {BURASI}\n\n"
        f"Aranan adlar: {_ADAYLAR}\n\n"
        "Ureteç dosyasinin adini degistirdiyseniz, bu dosyadaki "
        "_ADAYLAR listesine ekleyin ya da bu betikle AYNI klasore "
        "kopyalayin."
    )
    st.stop()

spec = importlib.util.spec_from_file_location("gen", URETEC)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

SIGMA_SB = gen.SIGMA_SB
CFG = gen.CFG
SINIFLAR = ["NORMAL", "THERMAL OVERHEATING", "STEP LOSS", "VOLTAGE DROP"]

PENCERE = 500          # makalede raporlanan deger ile AYNI olmali
GUVEN_ESIGI = 0.80
YENILEME_SANIYE = 1.0  # ekrani okunabilir/ekran-goruntusu-alinabilir yapar

st.set_page_config(page_title="NEXUS-VAKUM", page_icon="*", layout="wide")
st.title("NEXUS-VAKUM: Autonomous Space Manufacturing Robot")
st.caption("Visual telemetry and predictive maintenance panel "
           f"| sliding window N={PENCERE} | confidence gate {GUVEN_ESIGI}")


# Oznitelik sirasi. model_sutunlari.pkl okunamazsa bu kullanilir,
# boylece 'sutunlar' HER DURUMDA tanimli olur (NameError imkansiz).
VARSAYILAN_SUTUNLAR = [
    "Ortam_Sicakligi_C", "Vakum_Basinci_Torr", "Calisma_Voltaji_V",
    "Faz_Akimi_A", "Faz_Direnci_Ohm", "Motor_Hizi_PPS",
    "Hesaplanan_Sicaklik_C", "Tork_Stres_Katsayisi",
]


@st.cache_resource
def model_yukle():
    if not MODEL_YOLU.exists():
        raise FileNotFoundError(
            f"{MODEL_YOLU} yok. Once su sirayla calistirin:\n"
            "  python 01_veri_uretimi.py\n"
            "  python 02_taban_cizgisi.py"
        )
    m = joblib.load(MODEL_YOLU)
    sut = joblib.load(SUTUN_YOLU) if SUTUN_YOLU.exists() \
        else list(VARSAYILAN_SUTUNLAR)
    return m, list(sut)


# Hata YUTULMUYOR: sorun varsa mesaji gosterip calismayi durduruyoruz,
# ama asla tanimsiz degiskenle devam etmiyoruz.
model, sutunlar = None, list(VARSAYILAN_SUTUNLAR)
try:
    model, sutunlar = model_yukle()
except Exception as hata:
    st.error(f"Model yuklenemedi.\n\n{type(hata).__name__}: {hata}")
    st.stop()

if model is None:
    st.error("Model yuklenemedi.")
    st.stop()

# Egitimde kullanilan sutunlarla uretilen telemetri uyusuyor mu?
_eksik = set(sutunlar) - set(VARSAYILAN_SUTUNLAR)
if _eksik:
    st.error(
        "Kaydedilmis model, bu ureteçin uretmedigi ozniteliklere ihtiyac "
        f"duyuyor: {sorted(_eksik)}\n\n"
        "Muhtemel sebep: elinizdeki nexus_model.pkl ESKI kodla egitilmis. "
        "Silip 02_taban_cizgisi.py'yi tekrar calistirin."
    )
    st.stop()


def tek_ornek(rng):
    """01_veri_uretimi.py ile AYNI fizik ve AYNI etiketleme.
    Tek fark: N=1 ve gorev suresi kisaltilmis entegrasyon."""
    cfg = CFG
    prof = rng.choice(4, p=cfg["profil_agirliklari"])
    U = lambda a, b: rng.uniform(a, b)

    par = [
        # ortam,      voltaj,      akim,        direnc,     pps
        ((-50, 30), (6.8, 9.5), (.20, .34), (9, 16), (100, 1400),
         (.10, .45), (2, 6), (80, 220), (600, 3600)),
        ((20, 80), (6.5, 9.5), (.34, .45), (15, 20), (300, 1800),
         (.70, 1.0), (3, 4), (200, 450), (2400, 10800)),
        ((-20, 50), (6.0, 9.0), (.28, .42), (11, 19), (2000, 3200),
         (.45, .95), (6, 2), (100, 300), (1200, 7200)),
        ((-40, 40), (5.0, 6.3), (.34, .45), (9, 20), (200, 2000),
         (.20, .70), (2, 5), (80, 300), (600, 5400)),
    ][prof]

    ortam = U(*par[0]); voltaj = U(*par[1]); akim = U(*par[2])
    direnc = U(*par[3]); pps = U(*par[4])
    yogunluk = U(*par[5])
    yaglayici = rng.beta(*par[6])
    R_il = U(*par[7]); sure = U(*par[8])
    basinc = 10 ** rng.uniform(*np.log10(cfg["basinc_Torr"]))

    # --- gecici rejim termal entegrasyon ---
    mc = cfg["kutle_kg"] * cfg["ozgul_isi_J_kgK"]
    A = cfg["yuzey_alani_m2"]
    epsF = cfg["yayinim"] * cfg["gorus_faktoru"]
    dt = cfg["zaman_adimi_s"]
    h = cfg["h_referans_W_m2K"] * basinc / (basinc + cfg["basinc_yarim_Torr"])
    Q = (akim ** 2) * direnc * yogunluk
    T0 = ortam + 273.15
    T = T0
    for k in range(int(sure / dt)):
        T += dt * (Q - epsF * SIGMA_SB * A * (T ** 4 - cfg["uzay_sicakligi_K"] ** 4)
                   - h * A * (T - T0) - (T - T0) / R_il) / mc
    C = T - 273.15

    # --- tork ---
    der = np.clip(1 - cfg["tork_derating_1_K"] * max(C - 25, 0), 0.25, 1.0)
    vf = np.clip(voltaj / cfg["nominal_voltaj_V"], 0.4, 1.0)
    pullout = (cfg["tutma_torku_mNm"] * der * vf
               / np.sqrt(1 + (pps / cfg["kose_frekansi_PPS"]) ** 2))
    surt = (cfg["surtunme_taban_mNm"] + cfg["surtunme_kazanci_mNm"]
            * yaglayici * (1 + 0.004 * max(C - 25, 0)))
    ts = (cfg["sabit_yuk_mNm"] + surt + 5 * yogunluk) / max(pullout, 1e-3)

    # --- AYNI olasiliksal etiketleme ---
    p_t = (1 / (1 + np.exp(-(C - cfg["T_sigmoid_merkez_C"]) / cfg["T_sigmoid_egim"]))
           if C > cfg["T_kapi_C"] else 0.0)
    p_s = (1 / (1 + np.exp(-(ts - cfg["S_sigmoid_merkez"]) / cfg["S_sigmoid_egim"]))
           if ts > cfg["S_kapi"] else 0.0)
    p_v = (cfg["p_voltaj"] if (voltaj < cfg["V_kritik_V"]
                               and akim > cfg["I_kritik_A"]) else 0.0)
    P = np.array([max(1 - max(p_t, p_s, p_v), 0.0), p_t, p_s, p_v])
    P = P / P.sum()
    etiket = int(rng.choice(4, p=P))
    if rng.random() < cfg["etiket_gurultusu"]:
        etiket = int(rng.integers(0, 4))

    satir = {
        "Ortam_Sicakligi_C": ortam, "Vakum_Basinci_Torr": basinc,
        "Calisma_Voltaji_V": voltaj, "Faz_Akimi_A": akim,
        "Faz_Direnci_Ohm": direnc, "Motor_Hizi_PPS": pps,
        "Hesaplanan_Sicaklik_C": C + rng.normal(0, cfg["termistor_sigma_C"]),
        "Tork_Stres_Katsayisi": ts + rng.normal(0, cfg["tork_sigma"]),
    }
    return satir, etiket


# ---------------- oturum durumu: SINIRLI gecmis ----------------
if "pencere" not in st.session_state:
    st.session_state.pencere = deque(maxlen=PENCERE)   # (dogru_mu, guven)
    st.session_state.egri = deque(maxlen=200)
    st.session_state.rng = np.random.default_rng()
    st.session_state.sayac = 0
    st.session_state.askida = 0

canli = st.sidebar.checkbox("Live stream simulation", value=True)
st.sidebar.markdown("---")
kutu = st.sidebar.empty()

c1, c2, c3, c4 = st.columns(4)
g1, g2 = st.columns(2)

if canli:
    satir, gercek = tek_ornek(st.session_state.rng)
    X = pd.DataFrame([satir])[sutunlar]
    tahmin = int(model.predict(X)[0])
    guven = float(model.predict_proba(X).max())

    st.session_state.pencere.append((tahmin == gercek, guven))
    st.session_state.sayac += 1
    if guven < GUVEN_ESIGI:
        st.session_state.askida += 1

    c1.metric("Temperature", f"{satir['Hesaplanan_Sicaklik_C']:.1f} C")
    c2.metric("Voltage", f"{satir['Calisma_Voltaji_V']:.2f} V")
    c3.metric("Speed", f"{int(satir['Motor_Hizi_PPS'])} PPS")
    c4.metric("Torque stress", f"{satir['Tork_Stres_Katsayisi']:.2f}")

    if guven < GUVEN_ESIGI:
        st.warning(f"HOLD - autonomous action suspended "
                   f"(confidence {guven*100:.1f}% < {GUVEN_ESIGI*100:.0f}%) "
                   f"| provisional: {SINIFLAR[tahmin]}")
    else:
        st.success(f"{SINIFLAR[tahmin]}  |  confidence {guven*100:.1f}%")

    # ---- GERCEK kayan pencere ----
    dogru = np.array([d for d, _ in st.session_state.pencere])
    n = len(dogru)
    if n >= 30:
        acc = dogru.mean()
        sigma = np.sqrt(acc * (1 - acc) / n)
        st.session_state.egri.append(acc)
        kutu.info(
            f"**Sliding-window accuracy (N={n})**\n\n"
            f"* Point estimate: %{acc*100:.2f}\n"
            f"* 95% CI: [%{(acc-1.96*sigma)*100:.2f}, "
            f"%{(acc+1.96*sigma)*100:.2f}]\n"
            f"* Expected binomial sigma: %{sigma*100:.2f}\n\n"
            f"**Uncertainty gate**\n\n"
            f"* Suspended: {st.session_state.askida}/"
            f"{st.session_state.sayac} "
            f"(%{100*st.session_state.askida/st.session_state.sayac:.1f})"
        )
        with g1:
            st.write("### Sliding-window accuracy")
            st.line_chart(pd.DataFrame({"accuracy": list(st.session_state.egri)}))
    else:
        kutu.info(f"Filling window... {n}/30")

    with g2:
        st.write("### Confidence distribution")
        st.line_chart(pd.DataFrame(
            {"confidence": [g for _, g in st.session_state.pencere]}))

    time.sleep(YENILEME_SANIYE)
    st.rerun()