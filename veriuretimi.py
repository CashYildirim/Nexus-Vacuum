"""
NEXUS-VAKUM | Adim 1: Fiziksel temelli telemetri ureteci
=========================================================
ESKI KODA GORE DUZELTILENLER

(A) ETIKET GURULTUSU HATASI
    Eski kodda %8'lik flip blogunun hemen ardindan gelen
        ariza_array = np.array(ariza_etiketi)
    satiri o blogu tamamen geri aliyordu. Olculen sonuc: CSV'ye
    %8 degil %2.87 gurultu gidiyordu. Artik TEK blok var ve
    gerceklesen degisim sayisi raporlaniyor.

(B) TERMAL MODEL -> GECICI REJIM + ISINIM
    Makalenin girisinde Q ~ T^4 vaat ediliyor ama eski denklemde
    Stefan-Boltzmann terimi yoktu. Artik yigik kapasite modeli:
        m*c*dT/dt = Q_joule - eps*sigma*A*F*(T^4 - T_uzay^4)
                    - h(P)*A*(T - T_ortam) - (T - T_govde)/R_iletim
    Sayisal entegrasyon ileri Euler, dt = 15 s.

(C) VAKUM ARTIK GERCEKTEN ETKILI
    Eski terim (1.5 - P*100), P in [1e-6, 1e-3] icin 1.4000..1.4999
    araligindaydi; toplam etkisi %6.7, yani olu bir degiskendi.
    Artik serbest molekul gecisi:  h(P) = h_ref * P / (P + P_yarim)
    ve basinc log-uniform ornekleniyor.

(D) TORK MODELI -> PULL-OUT EGRISI
    Eski (PPS/PPSmax)*(T/Tcrit) ifadesinin fiziksel dayanagi yoktu.
    Artik adim kaybi, yuk torkunun pull-out torkunu asmasiyla olusuyor:
        T_pullout = T_tutma * derating(T) * (V/V_nom)
                    / sqrt(1 + (PPS/PPS_kose)^2)
    Surtunme torku yaglayici bozunmasina ve sicakliga bagli.

(E) GIZLI (LATENT) DEGISKENLER  -- SIZINTIYI KIRAN ASIL DUZELTME
    baski_yogunlugu, yaglayici_bozunmasi, iletim_direnci ve
    gorev_suresi CSV'ye YAZILMIYOR. Boylece etiket, gozlenebilir
    ozniteliklerin kapali-form fonksiyonu olmaktan cikiyor ve model
    indirgenemez (aleatorik) belirsizlikle karsilasiyor. Gercek bir
    uzay aracinda da bu durumlar dogrudan olculemez.

(F) GOREV PROFILI KARISIMI
    Tekduze rastgele ornekleme ile arıza siniflari neredeyse hic
    uretilmiyordu (olculen: %93 saglikli). Artik 4 gorev profilinin
    karisimi kullaniliyor. DIKKAT: bu, sinif onceliklerinin bir
    TASARIM PARAMETRESI oldugu anlamina gelir; yorunge arıza
    oranlarinin kestirimi DEGILDIR. Bunu makalede acikca yazin.

(G) BAYES TAVANI RAPORLANIYOR
    Model bu degerin uzerine cikamaz. Makalede bu sayi verilmelidir.

Calistirma : python 01_veri_uretimi.py
Cikti      : nexus_vakum_veri.csv , nexus_vakum_config.json
"""

import json
import numpy as np
import pandas as pd

SIGMA_SB = 5.670374419e-8

CFG = {
    "seed": 42,
    "n_sample": 10000,

    # --- aktuator termal ---
    "kutle_kg": 0.150,
    "ozgul_isi_J_kgK": 460.0,
    "yuzey_alani_m2": 0.0042,
    "yayinim": 0.80,
    "gorus_faktoru": 0.15,          # aktuator yapinin icine gomulu
    "uzay_sicakligi_K": 200.0,
    "zaman_adimi_s": 15.0,
    "max_gorev_suresi_s": 10800.0,

    # --- vakum / konveksiyon ---
    "basinc_Torr": [1e-6, 1e-2],
    "h_referans_W_m2K": 8.5,
    "basinc_yarim_Torr": 1.0,

    # --- mekanik ---
    "tutma_torku_mNm": 90.0,
    "kose_frekansi_PPS": 1800.0,
    "nominal_voltaj_V": 9.0,
    "tork_derating_1_K": 0.0035,
    "surtunme_taban_mNm": 4.0,
    "surtunme_kazanci_mNm": 14.0,
    "sabit_yuk_mNm": 8.0,

    # --- etiketleme ---
    "T_sigmoid_merkez_C": 100.0, "T_sigmoid_egim": 4.0, "T_kapi_C": 80.0,
    "S_sigmoid_merkez": 0.95, "S_sigmoid_egim": 0.07, "S_kapi": 0.75,
    "V_kritik_V": 5.8, "I_kritik_A": 0.35, "p_voltaj": 0.80,

    "etiket_gurultusu": 0.05,

    # --- olcum gurultusu ---
    "termistor_sigma_C": 1.5,
    "tork_sigma": 0.02,

    # --- gorev profili agirliklari ---
    "profil_agirliklari": [0.50, 0.20, 0.18, 0.12],
    "profil_adlari": ["Nominal", "Termal_Yuk", "Mekanik_Yuk", "Guc_Anomalisi"],
}


def uret(cfg=CFG, kaydet=True):
    rng = np.random.default_rng(cfg["seed"])
    N = cfg["n_sample"]

    # =================================================================
    # 1. GOREV PROFILI KARISIMI
    # =================================================================
    prof = rng.choice(4, N, p=cfg["profil_agirliklari"])
    U = lambda a, b: rng.uniform(a, b, N)
    S = lambda a, b, c, d: np.select(
        [prof == 0, prof == 1, prof == 2, prof == 3], [a, b, c, d])

    ortam_C = S(U(-50, 30), U(20, 80), U(-20, 50), U(-40, 40))
    voltaj = S(U(6.8, 9.5), U(6.5, 9.5), U(6.0, 9.0), U(5.0, 6.3))
    akim = S(U(.20, .34), U(.34, .45), U(.28, .42), U(.34, .45))
    direnc = S(U(9, 16), U(15, 20), U(11, 19), U(9, 20))
    pps = S(U(100, 1400), U(300, 1800), U(2000, 3200), U(200, 2000))

    basinc = 10 ** rng.uniform(*np.log10(cfg["basinc_Torr"]), N)

    # ---- GIZLI degiskenler: CSV'ye YAZILMAZ ----
    baski_yogunlugu = S(U(.10, .45), U(.70, 1.0), U(.45, .95), U(.20, .70))
    yaglayici = S(rng.beta(2, 6, N), rng.beta(3, 4, N),
                  rng.beta(6, 2, N), rng.beta(2, 5, N))
    R_iletim = S(U(80, 220), U(200, 450), U(100, 300), U(80, 300))
    gorev_suresi = S(U(600, 3600), U(2400, 10800), U(1200, 7200), U(600, 5400))

    # =================================================================
    # 2. GECICI REJIM TERMAL ENTEGRASYON
    # =================================================================
    mc = cfg["kutle_kg"] * cfg["ozgul_isi_J_kgK"]
    A = cfg["yuzey_alani_m2"]
    epsF = cfg["yayinim"] * cfg["gorus_faktoru"]
    T_uzay4 = cfg["uzay_sicakligi_K"] ** 4
    dt = cfg["zaman_adimi_s"]

    h = cfg["h_referans_W_m2K"] * basinc / (basinc + cfg["basinc_yarim_Torr"])
    Q_joule = (akim ** 2) * direnc * baski_yogunlugu          # W

    T0 = ortam_C + 273.15
    T = T0.copy()
    for k in range(int(cfg["max_gorev_suresi_s"] / dt)):
        aktif = (k * dt) < gorev_suresi
        dT = dt * (Q_joule
                   - epsF * SIGMA_SB * A * (T ** 4 - T_uzay4)
                   - h * A * (T - T0)
                   - (T - T0) / R_iletim) / mc
        T = T + np.where(aktif, dT, 0.0)

    sicaklik_C = T - 273.15
    sicaklik_olculen = sicaklik_C + rng.normal(0, cfg["termistor_sigma_C"], N)

    # =================================================================
    # 3. TORK DENGESI
    # =================================================================
    derating = np.clip(
        1.0 - cfg["tork_derating_1_K"] * np.maximum(sicaklik_C - 25.0, 0),
        0.25, 1.0)
    v_faktor = np.clip(voltaj / cfg["nominal_voltaj_V"], 0.4, 1.0)

    tork_pullout = (cfg["tutma_torku_mNm"] * derating * v_faktor
                    / np.sqrt(1.0 + (pps / cfg["kose_frekansi_PPS"]) ** 2))

    surtunme = (cfg["surtunme_taban_mNm"] + cfg["surtunme_kazanci_mNm"]
                * yaglayici * (1.0 + 0.004 * np.maximum(sicaklik_C - 25, 0)))
    tork_yuk = cfg["sabit_yuk_mNm"] + surtunme + 5.0 * baski_yogunlugu

    tork_stres = tork_yuk / np.maximum(tork_pullout, 1e-3)
    tork_olculen = tork_stres + rng.normal(0, cfg["tork_sigma"], N)

    # =================================================================
    # 4. OLASILIKSAL ETIKETLEME
    # =================================================================
    p_t = np.where(sicaklik_C > cfg["T_kapi_C"],
                   1 / (1 + np.exp(-(sicaklik_C - cfg["T_sigmoid_merkez_C"])
                                   / cfg["T_sigmoid_egim"])), 0.0)
    p_s = np.where(tork_stres > cfg["S_kapi"],
                   1 / (1 + np.exp(-(tork_stres - cfg["S_sigmoid_merkez"])
                                   / cfg["S_sigmoid_egim"])), 0.0)
    p_v = np.where((voltaj < cfg["V_kritik_V"]) & (akim > cfg["I_kritik_A"]),
                   cfg["p_voltaj"], 0.0)

    P = np.stack([np.zeros(N), p_t, p_s, p_v], axis=1)
    P[:, 0] = np.clip(1.0 - P[:, 1:].max(axis=1), 0.0, None)
    P = P / P.sum(axis=1, keepdims=True)

    etiket = np.array([rng.choice(4, p=P[i]) for i in range(N)])

    # ---- TEK etiket gurultusu blogu ----
    temiz = etiket.copy()
    n_flip = int(N * cfg["etiket_gurultusu"])
    idx = rng.choice(N, size=n_flip, replace=False)
    etiket[idx] = rng.integers(0, 4, size=n_flip)
    gerceklesen = int((etiket != temiz).sum())

    # =================================================================
    # 5. KAYDETME
    # =================================================================
    df = pd.DataFrame({
        "Ortam_Sicakligi_C": ortam_C,
        "Vakum_Basinci_Torr": basinc,
        "Calisma_Voltaji_V": voltaj,
        "Faz_Akimi_A": akim,
        "Faz_Direnci_Ohm": direnc,
        "Motor_Hizi_PPS": pps,
        "Hesaplanan_Sicaklik_C": sicaklik_olculen,
        "Tork_Stres_Katsayisi": tork_olculen,
        "Ariza_Durumu": etiket,
    })

    bayes_ham = float(P.max(axis=1).mean())
    bayes_gur = ((1 - cfg["etiket_gurultusu"]) * bayes_ham
                 + cfg["etiket_gurultusu"] * 0.25)

    rapor = {
        "config": cfg,
        "sinif_dagilimi": np.bincount(etiket, minlength=4).tolist(),
        "profil_dagilimi": np.bincount(prof, minlength=4).tolist(),
        "etiket_gurultusu_hedef": n_flip,
        "etiket_gurultusu_gerceklesen": gerceklesen,
        "bayes_tavani_etiket_gurultusuz": round(bayes_ham, 4),
        "bayes_tavani_etiket_gurultulu": round(bayes_gur, 4),
        "sicaklik_C": {"min": round(float(sicaklik_C.min()), 1),
                       "max": round(float(sicaklik_C.max()), 1),
                       "ort": round(float(sicaklik_C.mean()), 1)},
        "tork_stres": {"min": round(float(tork_stres.min()), 3),
                       "max": round(float(tork_stres.max()), 3),
                       "ort": round(float(tork_stres.mean()), 3)},
        "latent_degiskenler": ["baski_yogunlugu", "yaglayici_bozunmasi",
                               "R_iletim_K_W", "gorev_suresi_s"],
    }

    if kaydet:
        df.to_csv("nexus_vakum_veri.csv", index=False)
        with open("nexus_vakum_config.json", "w") as f:
            json.dump(rapor, f, indent=2, ensure_ascii=False)

    print("=" * 66)
    print("VERI SETI OLUSTURULDU -> nexus_vakum_veri.csv")
    print("=" * 66)
    ad = ["Saglikli", "Termal", "Adim Kacirma", "Voltaj"]
    for i, n in enumerate(rapor["sinif_dagilimi"]):
        print(f"  {i} {ad[i]:14s} {n:5d}  (%{100*n/N:.1f})")
    print(f"\n  Sicaklik : {rapor['sicaklik_C']['min']} .. "
          f"{rapor['sicaklik_C']['max']} C  (ort {rapor['sicaklik_C']['ort']})")
    print(f"  Tork stres: {rapor['tork_stres']['min']} .. "
          f"{rapor['tork_stres']['max']}")
    print(f"  Etiket gurultusu: hedef {n_flip}, gerceklesen {gerceklesen}")
    print()
    print(f"  >> BAYES TAVANI (etiket gurultusuz) : %{bayes_ham*100:.2f}")
    print(f"  >> BAYES TAVANI (etiket gurultulu)  : %{bayes_gur*100:.2f}")
    print("     Hicbir model bu degerin uzerine cikamaz.")
    print("     MAKALEDE BU SAYIYI RAPOR EDIN.")
    print("=" * 66)
    return df, rapor


if __name__ == "__main__":
    uret()