# -*- coding: utf-8 -*-
"""
=====================================================================
 SISTEM BI ANALISIS TRANSAKSI TOKO  (Apriori + K-Means + Dashboard)
 Mata Kuliah : Business Intelligence (TIF46212)  -  P8 / UTS
 Nama        : Serafina Paebesi
 NPM         : 51230110
 Prodi       : Teknologi Informasi
 Fakultas    : Pertanian, Sains dan Kesehatan
 Universitas : Universitas Timor, Kefamenanu - 2026
---------------------------------------------------------------------
 Alur:
   1. Halaman Cover Depan (Logo UNIMOR Besar di Tengah + Identitas)
   2. Tombol Masuk ke Sistem -> Halaman Login (Foto Serafina + User & Pass)
   3. Halaman Utama Upload File: Menu samping belum muncul (fokus upload)
   4. Setelah upload berhasil -> Menu samping berbentuk Card & Seluruh
      proses sistem BI (Filter Interaktif Unik, KPI, Apriori, K-Means) muncul!
---------------------------------------------------------------------
 PERBAIKAN TERBARU:
   - Harga singkat pada kartu KPI memakai TITIK sebagai pemisah desimal
     (contoh: Rp 698.4 juta), bukan koma.
   - Kartu KPI didesain ulang: ikon + lingkaran persen di baris atas,
     angka & keterangan di bawahnya, sehingga angka TIDAK lagi tertimpa
     lingkaran persentase.
=====================================================================
"""
import io
import itertools
import os
import sys

# Mengurangi peringatan KMeans di Windows (MKL) -- harus sebelum import sklearn
os.environ.setdefault("OMP_NUM_THREADS", "1")

_butuh = [("flask", "flask"), ("pandas", "pandas"), ("numpy", "numpy"),
          ("sklearn", "scikit-learn"), ("plotly", "plotly"), ("openpyxl", "openpyxl")]
_hilang = []
for _mod, _pip in _butuh:
    try:
        __import__(_mod)
    except ImportError:
        _hilang.append(_pip)
if _hilang:
    print("\nLibrary berikut belum terpasang: " + ", ".join(_hilang))
    print("Jalankan perintah ini di terminal, lalu jalankan ulang program:\n")
    print("    python -m pip install " + " ".join(_hilang) + "\n")
    sys.exit(1)

import json
import threading
import traceback
import uuid
import webbrowser
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (Flask, Response, redirect, render_template,
                   request, send_file, session, url_for)
from jinja2 import DictLoader
from markupsafe import Markup, escape
from werkzeug.exceptions import HTTPException
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# KONFIGURASI IDENTITAS & PARAMETER
# ---------------------------------------------------------------------
IDENTITAS = {
    "nama": "Serafina Paebesi",
    "npm": "51230110",
    "prodi": "Teknologi Informasi",
    "fakultas": "Pertanian, Sains dan Kesehatan",
    "univ": "Universitas Timor",
    "kota": "Kefamenanu",
    "tahun": "2026",
    "matkul": "Business Intelligence (TIF46212)",
    "dosen": "Krisantus Jumarto Tey Seran, S.T., M.T.",
}
K_RANGE = list(range(2, 9))          # Elbow & Silhouette: k = 2..8
K_OVERRIDE = None                    # None = otomatis; angka 2..8 = dipilih manual
MIN_SUP = 0.01                       # minimum support Apriori
MIN_CONF = 0.15                      # minimum confidence Apriori
RANDOM_STATE = 42
MAX_QTY_WAJAR = 20                   # qty > 20 dianggap tidak wajar untuk toko ritel
MAX_PRICE_RATIO = 3                  # harga > 3x harga katalog dianggap tidak wajar

APP_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "p8-uts-bi-serafina-key-2026")

# ---------------------------------------------------------------------
# FORMAT ANGKA & MATA UANG
# ---------------------------------------------------------------------
def fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")

def fmt_dec(x, d=1):
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_rp(x):
    return "Rp " + fmt_int(x)

def fmt_dec_titik(x, d=1):
    """Angka desimal dengan TITIK sebagai pemisah desimal (tanpa pemisah ribuan)."""
    return f"{x:.{d}f}"

def fmt_rp_short(x):
    """Format harga singkat. Desimal memakai TITIK (bukan koma).
    Contoh: Rp 698.4 juta | Rp 1.25 miliar"""
    if abs(x) >= 1e9:
        return "Rp " + fmt_dec_titik(x / 1e9, 2) + " miliar"
    if abs(x) >= 1e6:
        return "Rp " + fmt_dec_titik(x / 1e6, 1) + " juta"
    return fmt_rp(x)

def fmt_pct(x, d=1):
    return fmt_dec(float(x) * 100, d) + "%"

def seg_color(label):
    s = str(label)
    if "Loyal" in s:
        return "#1E63C6"
    if "At Risk" in s:
        return "#F59E0B"
    if "Churn" in s:
        return "#EF4444"
    if "Potensial" in s:
        return "#10B981"
    return "#8B5CF6"

app.jinja_env.filters.update({
    "rp": fmt_rp, "rps": fmt_rp_short, "num": fmt_int, "dec": fmt_dec,
    "pct": fmt_pct, "segc": seg_color
})

STRATEGI = {
    "Loyal": ("Pertahankan dan apresiasi loyalitas",
              "Program member eksklusif / poin ganda, akses awal promo, bundling produk premium, personalisasi layanan."),
    "At Risk": ("Reaktivasi cepat & prioritas",
                "Kampanye personal 'Kami Rindu Anda' via WA/SMS dengan voucher khusus pada produk favorit sebelum churn permanen."),
    "Churn": ("Win-back efisien",
              "Voucher stimulus berbiaya rendah dan pengingat berkala untuk menguji keaktifan berbelanja kembali."),
    "Potensial": ("Dorong frekuensi belanja",
                  "Cross-selling berbasis aturan Apriori, paket hemat belanja mingguan/bulanan, promo 'Beli X Dapat Y'."),
    "Baru": ("Onboarding & edukasi produk",
             "Voucher belanja nota kedua, rekomendasi produk sembako dan kebutuhan harian pokok terpopuler."),
}

def strategi_for(label):
    for k, v in STRATEGI.items():
        if k in label:
            return v
    return ("Pantau & Evaluasi", "Lakukan pemantauan berkala dan uji penawaran terbatas.")

# ---------------------------------------------------------------------
# 1. MEMUAT DATA EXCEL
# ---------------------------------------------------------------------
def find_data_file():
    """Cari file Excel soal yang memiliki sheet 'Data_Raw'."""
    env = os.environ.get("P8_DATA")
    if env and os.path.exists(env):
        return env
    from openpyxl import load_workbook
    folders = [APP_DIR, Path.cwd()]
    for folder in folders:
        for p in sorted(folder.glob("*.xlsx")):
            if p.name.startswith("~$"):
                continue
            try:
                wb = load_workbook(p, read_only=True)
                ok = "Data_Raw" in wb.sheetnames
                wb.close()
            except Exception:
                continue
            if ok:
                return str(p)
    return None

def load_raw(path):
    raw = pd.read_excel(path, sheet_name="Data_Raw")
    prod = pd.read_excel(path, sheet_name="Data_Produk")
    cust = pd.read_excel(path, sheet_name="Data_Pelanggan")
    return raw, prod, cust

# ---------------------------------------------------------------------
# 2. AUDIT KUALITAS DATA
# ---------------------------------------------------------------------
def audit(raw, prod):
    m = raw.merge(prod[["Product_ID", "Reference_Unit_Price_Rp"]], on="Product_ID", how="left")
    exp = raw.Quantity * raw.Unit_Price * (1 - raw.Discount_Rate)
    q1, q3 = raw.Quantity.quantile([.25, .75])
    iqr = q3 - q1
    pm = raw.Payment_Method
    info = {
        "baris": len(raw), "kolom": raw.shape[1],
        "tgl_min": raw.Transaction_Date.min(), "tgl_max": raw.Transaction_Date.max(),
        "transaksi_unik": raw.Transaction_ID.nunique(), "pelanggan_unik": raw.Customer_ID.nunique(),
        "produk_unik": raw.Product_ID.nunique(), "total_sementara": float(raw.Line_Total.sum()),
        "rata_qty": float(raw.Quantity.mean()), "median_qty": float(raw.Quantity.median()),
    }
    masalah = [
        ("Missing Customer_ID", int(raw.Customer_ID.isna().sum())),
        ("Missing Region", int(raw.Region.isna().sum())),
        ("Missing Customer_Type", int(raw.Customer_Type.isna().sum())),
        ("Missing Product_ID", int(raw.Product_ID.isna().sum())),
        ("Missing Quantity", int(raw.Quantity.isna().sum())),
        ("Missing Unit_Price", int(raw.Unit_Price.isna().sum())),
        ("Missing Line_Total", int(raw.Line_Total.isna().sum())),
        ("Missing Payment_Method", int(pm.isna().sum())),
        ("Baris duplikat penuh", int(raw.duplicated().sum())),
        ("Duplikat Transaction_ID + Product_ID", int(raw.duplicated(["Transaction_ID", "Product_ID"]).sum())),
        ("Kategori tidak konsisten (kapital / spasi)", int((raw.Category != raw.Category.str.strip().str.title()).sum())),
        ("Payment_Method tidak konsisten (qris, Qris, 'QRIS ')",
         int((pm.notna() & (pm != "QRIS") & (pm.str.strip().str.upper() == "QRIS")).sum())),
        ("Quantity <= 0", int((raw.Quantity <= 0).sum())),
        ("Quantity ekstrem (> %d)" % MAX_QTY_WAJAR, int((raw.Quantity > MAX_QTY_WAJAR).sum())),
        ("Unit_Price <= 0", int((raw.Unit_Price <= 0).sum())),
        ("Unit_Price ekstrem (> %dx harga katalog)" % MAX_PRICE_RATIO,
         int((m.Unit_Price > MAX_PRICE_RATIO * m.Reference_Unit_Price_Rp).sum())),
        ("Line_Total tidak sama dengan Qty x Harga x (1 - Diskon)", int(((exp - raw.Line_Total).abs() > 1).sum())),
        ("Outlier Quantity menurut IQR (> Q3 + 1,5 x IQR)", int((raw.Quantity > q3 + 1.5 * iqr).sum())),
    ]
    return info, pd.DataFrame(masalah, columns=["Masalah", "Jumlah baris"])

# ---------------------------------------------------------------------
# 3. PREPROCESSING (CLEANING)
# ---------------------------------------------------------------------
def preprocess(raw, prod, cust):
    log = []
    df = raw.copy()
    pm = raw.Payment_Method
    n_teks = int((raw.Category != raw.Category.str.strip().str.title()).sum()
                 + (pm.notna() & (pm != "QRIS") & (pm.str.strip().str.upper() == "QRIS")).sum())
    df["Category"] = df.Category.str.strip().str.title()
    df["Payment_Method"] = df.Payment_Method.str.strip().str.title()
    df.loc[df.Payment_Method == "Qris", "Payment_Method"] = "QRIS"
    log.append(("Teks tidak konsisten", "Category dan Payment_Method: trim spasi + seragamkan kapital", "diperbaiki", n_teks))

    d1 = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    log.append(("Duplikat penuh", "Seluruh kolom identik: hapus, simpan 1", "dihapus", d1))
    mask = df.duplicated(["Transaction_ID", "Product_ID"], keep="first") & df.Product_ID.notna()
    d2 = int(mask.sum())
    df = df[~mask].copy()
    log.append(("Duplikat produk dalam 1 nota", "Transaction_ID + Product_ID sama tetapi nilai lain beda: simpan baris pertama", "dihapus", d2))

    name2id = prod.set_index("Product_Name").Product_ID
    k = int(df.Product_ID.isna().sum())
    df["Product_ID"] = df.Product_ID.fillna(df.Product_Name.map(name2id))
    log.append(("Missing Product_ID", "Diisi dari Product_Name lewat katalog Data_Produk", "diimputasi", k))
    df["Category"] = df.Product_ID.map(prod.set_index("Product_ID").Category)

    k = int(df.Customer_ID.isna().sum())
    fill = df.groupby("Transaction_ID").Customer_ID.transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    df["Customer_ID"] = df.Customer_ID.fillna(fill)
    sisa = int(df.Customer_ID.isna().sum())
    df = df[df.Customer_ID.notna()].copy()
    log.append(("Missing Customer_ID", "Diisi dari baris lain pada Transaction_ID yang sama", "diimputasi", k - sisa))
    if sisa:
        log.append(("Customer_ID tak terpulihkan", "Dihapus karena tidak bisa masuk RFM", "dihapus", sisa))

    c = cust.set_index("Customer_ID")
    for col in ["Region", "Customer_Type"]:
        k = int(df[col].isna().sum())
        df[col] = df[col].fillna(df.Customer_ID.map(c[col]))
        log.append((f"Missing {col}", "Diisi dari Data_Pelanggan berdasarkan Customer_ID", "diimputasi", k))

    k = int(df.Payment_Method.isna().sum())
    fp = df.groupby("Transaction_ID").Payment_Method.transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    df["Payment_Method"] = df.Payment_Method.fillna(fp).fillna("Tidak Diketahui")
    log.append(("Missing Payment_Method", "Diisi dari baris lain pada nota yang sama; sisanya 'Tidak Diketahui'", "diimputasi", k))

    ref = df.Product_ID.map(prod.set_index("Product_ID").Reference_Unit_Price_Rp)
    bad_p = df.Unit_Price.isna() | (df.Unit_Price <= 0) | (df.Unit_Price > MAX_PRICE_RATIO * ref)
    k = int(bad_p.sum())
    df.loc[bad_p, "Unit_Price"] = ref[bad_p]
    log.append(("Unit_Price tidak valid", f"NaN / <= 0 / > {MAX_PRICE_RATIO}x harga katalog diganti harga katalog", "diperbaiki", k))

    bad_q = df.Quantity.isna() | (df.Quantity <= 0) | (df.Quantity > MAX_QTY_WAJAR)
    k = int(bad_q.sum())
    est = (df.Line_Total / (df.Unit_Price * (1 - df.Discount_Rate))).round()
    ok = bad_q & est.between(1, MAX_QTY_WAJAR)
    df.loc[ok, "Quantity"] = est[ok]
    kk = int((bad_q & ~ok).sum())
    df.loc[bad_q & ~ok, "Quantity"] = 1
    log.append(("Quantity tidak valid", "NaN / <= 0 / > %d: diturunkan dari Line_Total / (Harga x (1 - Diskon))" % MAX_QTY_WAJAR, "diperbaiki", k))
    if kk:
        log.append(("  - tanpa Line_Total pembanding", "Quantity diisi modus = 1", "diimputasi", kk))

    exp = (df.Quantity * df.Unit_Price * (1 - df.Discount_Rate)).round(0)
    k = int(df.Line_Total.isna().sum())
    inc = int(((exp - df.Line_Total).abs() > 1).sum())
    df["Line_Total"] = exp
    log.append(("Line_Total kosong / tidak konsisten", "Dihitung ulang = Qty x Harga x (1 - Diskon)", "dihitung ulang", k + inc))

    df["Transaction_Date"] = pd.to_datetime(df.Transaction_Date)
    df["Month"] = df.Transaction_Date.dt.to_period("M").astype(str)
    df["Quantity"] = df.Quantity.astype(int)
    df = df.reset_index(drop=True)
    log_df = pd.DataFrame(log, columns=["Masalah", "Keputusan dan alasan", "Tindakan", "Jumlah baris"])
    banding = pd.DataFrame({
        "Kondisi": ["Jumlah baris", "Transaksi unik", "Pelanggan unik", "Total missing value", "Baris duplikat",
                    "Qty <= 0 atau > 20", "Total penjualan (Rp)"],
        "Sebelum": [len(raw), raw.Transaction_ID.nunique(), raw.Customer_ID.nunique(), int(raw.isna().sum().sum()),
                    int(raw.duplicated().sum()), int(((raw.Quantity <= 0) | (raw.Quantity > MAX_QTY_WAJAR)).sum()),
                    float(raw.Line_Total.sum())],
        "Sesudah": [len(df), df.Transaction_ID.nunique(), df.Customer_ID.nunique(), int(df.isna().sum().sum()),
                    int(df.duplicated().sum()), int(((df.Quantity <= 0) | (df.Quantity > MAX_QTY_WAJAR)).sum()),
                    float(df.Line_Total.sum())],
    })
    return df, log_df, banding

# ---------------------------------------------------------------------
# 4. BASKET, RFM, APRIORI, K-MEANS
# ---------------------------------------------------------------------
def build_basket(df):
    basket = df.groupby("Transaction_ID").Product_Name.nunique().rename("basket_size")
    matrix = pd.crosstab(df.Transaction_ID, df.Product_Name).clip(upper=1).astype(bool)
    return basket, matrix

def build_rfm(df):
    ref_date = df.Transaction_Date.max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer_ID").agg(
        Recency=("Transaction_Date", lambda s: (ref_date - s.max().normalize()).days),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Line_Total", "sum")).reset_index()
    rfm["Log_Frequency"] = np.log1p(rfm.Frequency)
    rfm["Log_Monetary"] = np.log1p(rfm.Monetary)
    return rfm, ref_date

def run_apriori(matrix, min_support, max_len=3):
    n = len(matrix)
    cols = list(matrix.columns)
    X = matrix.values
    sup, L = {}, {}
    for i, c in enumerate(cols):
        s = X[:, i].mean()
        if s >= min_support:
            L[(c,)] = s
    sup.update(L)
    k = 2
    while L and k <= max_len:
        prev = list(L.keys())
        cand = set()
        for a, b in itertools.combinations(prev, 2):
            u = tuple(sorted(set(a) | set(b)))
            if len(u) == k and all(tuple(sorted(s)) in L for s in itertools.combinations(u, k - 1)):
                cand.add(u)
        L = {}
        for u in cand:
            idx = [cols.index(c) for c in u]
            s = X[:, idx].all(axis=1).mean()
            if s >= min_support:
                L[u] = s
        sup.update(L)
        k += 1
    freq = pd.DataFrame([(" + ".join(k), len(k), v, int(round(v * n))) for k, v in sup.items()],
                        columns=["Itemset", "Ukuran", "Support", "Jumlah transaksi"])
    freq = freq.sort_values(["Ukuran", "Support"], ascending=[True, False]).reset_index(drop=True)
    return sup, freq

def make_rules(sup, min_conf, n_trx, min_lift=1.0):
    rows = []
    for items, s in sup.items():
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for ante in itertools.combinations(items, r):
                cons = tuple(i for i in items if i not in ante)
                sa, sc = sup[tuple(sorted(ante))], sup[tuple(sorted(cons))]
                conf = s / sa
                lift = conf / sc
                if conf >= min_conf and lift >= min_lift:
                    rows.append((" + ".join(ante), " + ".join(cons), s, conf, lift, s - sa * sc, int(round(s * n_trx))))
    rules = pd.DataFrame(rows, columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Leverage", "Jumlah transaksi"])
    return rules.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)

def top_pairs(rules, n=10):
    seen, rows = set(), []
    for _, r in rules.iterrows():
        key = frozenset(r.Antecedent.split(" + ") + r.Consequent.split(" + "))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
        if len(rows) == n:
            break
    return pd.DataFrame(rows).reset_index(drop=True) if rows else rules.head(0)

def prepare_kmeans(rfm):
    X = rfm[["Recency", "Log_Frequency", "Log_Monetary"]].values
    return StandardScaler().fit_transform(X)

def evaluate_k(Xs):
    out = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
        out.append((k, km.inertia_, silhouette_score(Xs, km.labels_)))
    return pd.DataFrame(out, columns=["k", "Inertia", "Silhouette"])

def label_clusters(rfm):
    prof = rfm.groupby("Cluster").agg(Recency=("Recency", "mean"), Frequency=("Frequency", "mean"),
                                      Monetary=("Monetary", "mean"))
    val = prof[["Frequency", "Monetary"]].rank(pct=True).mean(axis=1).rank(pct=True)
    rec = prof.Recency.rank(pct=True)
    labels = {}
    for c in prof.index:
        if rec[c] <= 0.5 and val[c] >= 0.75:
            labels[c] = "Loyal / Champions"
        elif rec[c] >= 0.75 and val[c] >= 0.5:
            labels[c] = "Bernilai Tinggi - Mulai Pasif (At Risk)"
        elif rec[c] >= 0.75:
            labels[c] = "Berisiko Churn"
        elif val[c] >= 0.5:
            labels[c] = "Pelanggan Potensial"
        else:
            labels[c] = "Pelanggan Baru / Kasual"
    seen = {}
    for c, l in list(labels.items()):
        seen[l] = seen.get(l, 0) + 1
        if seen[l] > 1:
            labels[c] = f"{l} ({seen[l]})"
    return labels

def run_pipeline(path):
    raw, prod, cust = load_raw(path)
    info, masalah = audit(raw, prod)
    df, log, banding = preprocess(raw, prod, cust)
    basket, matrix = build_basket(df)
    rfm, ref_date = build_rfm(df)
    skew = {
        "Frequency_raw": float(rfm.Frequency.skew()), "Frequency_log": float(rfm.Log_Frequency.skew()),
        "Monetary_raw": float(rfm.Monetary.skew()), "Monetary_log": float(rfm.Log_Monetary.skew()),
    }
    Xs = prepare_kmeans(rfm)
    ev = evaluate_k(Xs)
    k_auto = int(ev.loc[ev.Silhouette.idxmax(), "k"])
    k = K_OVERRIDE if K_OVERRIDE in K_RANGE else k_auto
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    rfm["Cluster"] = km.labels_
    lab = label_clusters(rfm)
    rfm["Segmen"] = rfm.Cluster.map(lab)
    sup, freq = run_apriori(matrix, MIN_SUP)
    rules = make_rules(sup, MIN_CONF, len(matrix))
    return {
        "path": path, "raw": raw, "prod": prod, "cust": cust, "info": info, "masalah": masalah,
        "df": df, "log": log, "banding": banding, "basket": basket, "matrix": matrix,
        "rfm": rfm, "ref_date": ref_date, "skew": skew, "Xs": Xs, "ev": ev,
        "k_auto": k_auto, "k": k, "min_sup": MIN_SUP, "min_conf": MIN_CONF,
        "freq": freq, "rules": rules
    }

S = {}
LOAD_ERROR = {"msg": ""}

def load_state(path=None):
    if not path:
        path = find_data_file()
    if not path:
        S.clear()
        LOAD_ERROR["msg"] = "File Excel soal tidak ditemukan."
        return False
    try:
        data = run_pipeline(path)
        S.clear()
        S.update(data)
        LOAD_ERROR["msg"] = ""
        return True
    except Exception as e:
        S.clear()
        LOAD_ERROR["msg"] = f"{type(e).__name__}: {e}"
        return False

# Pemuatan awal jika ada file di folder
load_state()

# ---------------------------------------------------------------------
# 5. VISUAL HELPER (Plotly.js)
# ---------------------------------------------------------------------
def _js(o):
    return json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x))

def fig(traces, layout=None, height=360, legend_bottom=True):
    lay = {
        "margin": {"l": 55, "r": 24, "t": 36, "b": 50},
        "height": height,
        "template": None,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Plus Jakarta Sans, Segoe UI, sans-serif", "size": 12, "color": "#0F172A"}
    }
    if legend_bottom:
        lay["legend"] = {"orientation": "h", "y": -0.22, "x": 0}
    for k, v in (layout or {}).items():
        lay[k] = {**lay[k], **v} if isinstance(v, dict) and isinstance(lay.get(k), dict) else v
    lay.pop("template", None)
    i = "p" + uuid.uuid4().hex[:8]
    return Markup(f'<div id="{i}"></div><script>Plotly.newPlot("{i}",{_js(traces)},{_js(lay)},'
                  f'{{responsive:true,displaylogo:false}});</script>')

def df_html(df, cls="tbl", sortable=False, table_id=None):
    c = cls + (" sortable" if sortable else "")
    h = df.to_html(index=False, border=0, classes=c, escape=True, na_rep="-")
    if table_id:
        h = h.replace("<table ", f'<table id="{table_id}" ', 1)
    return Markup(h)

def L(x):
    return list(x)

# ---------------------------------------------------------------------
# 6. TEMPLATES HTML
# ---------------------------------------------------------------------
BASE = r"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} - BI Toko P8</title>
<script src="{{ url_for('plotly_js') }}"></script>
<style>
:root{
  --navy:#0B2A5B;--blue:#1E63C6;--orange:#F59E0B;--bg:#F1F5F9;
  --card:#FFFFFF;--ink:#0F172A;--mute:#64748B;--line:#E2E8F0;
  --green:#10B981;--red:#EF4444;--purple:#8B5CF6;
}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI",Roboto,-apple-system,BlinkMacSystemFont,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.55}
a{color:var(--blue)}

/* TOPBAR */
.top{
  background:var(--navy);
  color:#fff;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:16px;
  padding:10px 24px;
  position:sticky;
  top:0;
  z-index:30;
  border-bottom:4px solid var(--orange);
  box-shadow:0 4px 16px rgba(11,42,91,0.25);
}
.brand-box{display:flex;align-items:center;gap:14px}
.brand-logo{height:44px;width:auto;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.3))}
.brand b{font-size:17px;display:block;line-height:1.2}
.brand span{font-size:12px;color:#BCD0F2}
.who-box{display:flex;align-items:center;gap:12px;text-align:right}
.who-avatar{width:40px;height:40px;border-radius:50%;object-fit:cover;border:2px solid var(--orange);box-shadow:0 2px 8px rgba(0,0,0,0.25)}
.who{font-size:12px;line-height:1.3}
.who b{font-size:13px}
.who a{color:#FFD58A;margin-left:8px;font-weight:600;text-decoration:none}
.who a:hover{text-decoration:underline}

/* LAYOUT WITH SIDEBAR */
.shell{display:grid;grid-template-columns:270px 1fr;min-height:calc(100vh - 66px)}
.side{
  background:#FFFFFF;
  border-right:1px solid #E2E8F0;
  padding:20px 14px;
  position:sticky;
  top:66px;
  align-self:start;
  height:calc(100vh - 66px);
  overflow-y:auto;
}
.side-header{
  font-size:11px;
  font-weight:700;
  letter-spacing:1px;
  color:#94A3B8;
  text-transform:uppercase;
  margin:4px 8px 12px;
}
.nav-card{
  display:flex;
  align-items:center;
  gap:10px;
  padding:10px 12px;
  margin-bottom:8px;
  border-radius:10px;
  background:#F8FAFD;
  border:1px solid #E2E8F0;
  color:#334155;
  text-decoration:none;
  font-size:13px;
  font-weight:600;
  transition:all 0.2s ease;
}
.nav-card:hover{
  background:#EFF6FF;
  border-color:#BFDBFE;
  color:#1D4ED8;
  transform:translateX(3px);
}
.nav-card.on{
  background:linear-gradient(135deg, #0B2A5B, #1E63C6);
  border-color:#1E63C6;
  color:#FFFFFF;
  box-shadow:0 4px 14px rgba(11,42,91,0.2);
}
.nav-card .nav-num{
  width:24px;
  height:24px;
  display:flex;
  align-items:center;
  justify-content:center;
  border-radius:6px;
  background:rgba(0,0,0,0.06);
  font-size:11px;
  font-weight:700;
}
.nav-card.on .nav-num{
  background:rgba(255,255,255,0.22);
  color:#FCD34D;
}
main{padding:26px 32px;min-width:0}

/* LAYOUT WITHOUT SIDEBAR (UPLOAD ONLY) */
.shell-upload-only{
  min-height:calc(100vh - 66px);
  display:flex;
  align-items:center;
  justify-content:center;
  padding:40px 20px;
}
.upload-center-container{max-width:780px;width:100%}
.upload-hero-card{
  background:#FFFFFF;
  border:1px solid #DCE6F5;
  border-radius:22px;
  padding:44px 48px;
  box-shadow:0 18px 45px rgba(11,42,91,0.08);
  text-align:center;
}
.upload-logo{
  height:95px;
  width:auto;
  display:block;
  margin:0 auto 16px;
  filter:drop-shadow(0 4px 12px rgba(11,42,91,0.22));
}
.upload-badge{
  display:inline-block;
  background:#EFF6FF;
  color:#1D4ED8;
  border:1px solid #BFDBFE;
  font-size:12px;
  font-weight:700;
  padding:4px 14px;
  border-radius:99px;
  margin-bottom:14px;
  letter-spacing:0.5px;
}
.upload-title{font-size:26px;color:#0B2A5B;margin:0 0 10px;font-weight:800}
.upload-sub{color:#64748B;font-size:14.5px;line-height:1.6;max-width:620px;margin:0 auto 26px}
.dropzone-box{
  background:#F8FAFD;
  border:2px dashed #93C5FD;
  border-radius:16px;
  padding:32px 24px;
  margin-bottom:24px;
  transition:all 0.25s ease;
}
.dropzone-box:hover{border-color:#2563EB;background:#EFF6FF}
.dropzone-icon{font-size:46px;margin-bottom:8px}
.upload-file-input{
  display:inline-block;
  padding:10px 14px;
  background:#FFFFFF;
  border:1px solid #CBD5E1;
  border-radius:8px;
  margin-bottom:16px;
  font-size:14px;
}
.btn-upload-submit{
  display:inline-block;
  background:linear-gradient(135deg, #1E63C6, #0B2A5B);
  color:#FFFFFF;
  border:0;
  padding:13px 32px;
  border-radius:10px;
  font-size:15px;
  font-weight:700;
  cursor:pointer;
  box-shadow:0 6px 18px rgba(30,99,198,0.3);
  transition:all 0.2s ease;
}
.btn-upload-submit:hover{background:linear-gradient(135deg,#0B2A5B,#12408F);transform:translateY(-2px)}
.upload-quick-bar{
  border-top:1px solid #F1F5F9;
  padding-top:20px;
  display:flex;
  justify-content:space-between;
  align-items:center;
  flex-wrap:wrap;
  gap:12px;
}
.btn-quick-load{
  background:#FEF3C7;
  color:#92400E;
  border:1px solid #FCD34D;
  padding:11px 22px;
  border-radius:10px;
  font-size:14px;
  font-weight:700;
  cursor:pointer;
  transition:all 0.2s ease;
}
.btn-quick-load:hover{background:#FDE68A;transform:translateY(-1px)}
.upload-err-box{
  background:#FEE2E2;
  color:#991B1B;
  border-left:4px solid #EF4444;
  padding:10px 16px;
  border-radius:8px;
  font-size:13.5px;
  margin-bottom:20px;
  text-align:left;
}

/* GENERAL TYPOGRAPHY & CARDS */
h1{margin:0 0 6px;font-size:26px;color:var(--navy);font-weight:800}
h2{font-size:18px;margin:0 0 10px;color:var(--navy);font-weight:700}
h3{font-size:15px;margin:14px 0 6px;font-weight:700}
.sub{color:var(--mute);margin:0 0 22px;max-width:920px;font-size:14.5px}
.card{
  background:var(--card);
  border:1px solid var(--line);
  border-radius:14px;
  padding:22px 24px;
  margin-bottom:20px;
  box-shadow:0 4px 16px rgba(11,42,91,0.04);
}
.grid{display:grid;gap:20px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}

/* TABLES */
.tbl{border-collapse:collapse;width:100%;font-size:13px}
.tbl th{background:var(--navy);color:#fff;text-align:left;padding:9px 12px;position:sticky;top:0;white-space:nowrap}
.tbl td{padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
.tbl tr:nth-child(even) td{background:#F8FAFD}
.sortable th{cursor:pointer}.sortable th:hover{background:var(--blue)}
.scroll{max-height:460px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.scrollx{overflow-x:auto}
.pill{display:inline-block;padding:3px 10px;border-radius:99px;color:#fff;font-size:12px;font-weight:600}
.tag{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600}
.note{background:#FFF6E5;border-left:4px solid var(--orange);padding:12px 16px;border-radius:8px;font-size:14px;margin:12px 0}
.ok{background:#E9F7EF;border-left-color:var(--green);color:#1B6B45;padding:12px 16px;border-radius:8px;font-size:14px;margin:12px 0}
.formula{font-family:Cambria,"Times New Roman",serif;font-size:17px;background:#F8FAFD;border:1px dashed var(--blue);padding:12px 16px;border-radius:8px;margin:10px 0}
ol.steps li,ul.clean li{margin:5px 0}
.btn{display:inline-block;background:var(--blue);color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:14px;cursor:pointer;text-decoration:none}
.btn:hover{background:var(--navy)}.btn.o{background:var(--orange);color:#1d1d1d}.btn.gr{background:#64748B}
select,input[type=text],input[type=password],input[type=date],input[type=file]{padding:9px 12px;border:1px solid var(--line);border-radius:8px;font-size:14px;background:#fff}
.chk li{list-style:none;margin:4px 0}.chk li:before{content:"\2611  ";color:var(--green)}

@media(max-width:1050px){
  .shell{grid-template-columns:1fr}
  .side{position:static;height:auto}
  .g2,.g3,.g4{grid-template-columns:1fr}
}
@media print{.top,.side{display:none}.shell{display:block}}
</style></head><body>
<header class="top">
  <div class="brand-box">
    <img src="{{ url_for('static_asset', filename='logo_unimor.png') }}" alt="Logo UNIMOR" class="brand-logo">
    <div class="brand"><b>BI Analisis Transaksi Toko</b><span>P8 UTS &middot; Apriori + K-Means + Dashboard</span></div>
  </div>
  <div class="who-box">
    <img src="{{ url_for('static_asset', filename='foto_serafina.jpg') }}" alt="Serafina" class="who-avatar">
    <div class="who"><b>{{ id.nama }}</b> &middot; NPM {{ id.npm }}<br>{{ id.matkul }}
     <br><span style="color:#BCD0F2">Masuk: {{ session.nama }}</span><a href="{{ url_for('logout') }}">Keluar</a></div>
  </div>
</header>

{% if session.get('dataset_loaded') %}
<div class="shell">
<aside class="side">
  <div class="side-header">Navigasi Analisis</div>
  {% for ep, label in menu %}
  <a href="{{ url_for(ep) }}" class="nav-card {{ 'on' if ep == active else '' }}">
    <span class="nav-num">{{ '⚡' if loop.index == 1 else ('0' ~ (loop.index - 1) if (loop.index - 1) < 10 else (loop.index - 1)) }}</span>
    <span>{{ label }}</span>
  </a>
  {% endfor %}
</aside>
<main>{% block body %}{% endblock %}</main>
</div>
{% else %}
<div class="shell-upload-only">
  <div class="upload-center-container">{% block body_upload %}{% endblock %}</div>
</div>
{% endif %}

<script>
document.querySelectorAll('table.sortable').forEach(function(t){
  t.querySelectorAll('th').forEach(function(th,i){
    th.addEventListener('click',function(){
      var rows=Array.from(t.tBodies[0].rows),asc=th.dataset.asc!=='1';
      rows.sort(function(a,b){
        var x=a.cells[i].innerText.replace(/\./g,'').replace(',','.'),y=b.cells[i].innerText.replace(/\./g,'').replace(',','.');
        var nx=parseFloat(x),ny=parseFloat(y);
        if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;
        return asc?x.localeCompare(y):y.localeCompare(x);});
      rows.forEach(function(r){t.tBodies[0].appendChild(r)});
      th.dataset.asc=asc?'1':'0';});});});
document.querySelectorAll('input[data-filter]').forEach(function(inp){
  inp.addEventListener('input',function(){
    var q=inp.value.toLowerCase(),t=document.getElementById(inp.dataset.filter);
    Array.from(t.tBodies[0].rows).forEach(function(r){r.style.display=r.innerText.toLowerCase().indexOf(q)>-1?'':'none'});});});
</script>
</body></html>"""

COVER = r"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cover - Sistem BI Analisis Transaksi Toko</title>
<style>
*{box-sizing:border-box}
body{
  margin:0;
  font-family:"Segoe UI",Roboto,-apple-system,BlinkMacSystemFont,Arial,sans-serif;
  color:#14213D;
  min-height:100vh;
  display:grid;
  grid-template-columns:1.25fr 1fr;
}
.left{
  background:linear-gradient(160deg,#0B2A5B 0%,#12408F 70%,#1E63C6 100%);
  color:#fff;
  padding:44px 50px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
  position:relative;
  overflow:hidden;
}
.left:after{
  content:"";
  position:absolute;
  right:-90px;
  bottom:-90px;
  width:330px;
  height:330px;
  border-radius:50%;
  border:44px solid rgba(243,156,18,.9);
  pointer-events:none;
}
.left:before{
  content:"";
  position:absolute;
  right:120px;
  bottom:-60px;
  width:170px;
  height:170px;
  border-radius:50%;
  background:rgba(255,255,255,.08);
  pointer-events:none;
}
.brand-top{
  text-align:center;
  margin-bottom:16px;
  position:relative;
  z-index:2;
}
.logo-unimor{
  height:115px;
  width:auto;
  display:block;
  margin:0 auto 12px;
  filter:drop-shadow(0 6px 18px rgba(0,0,0,0.35));
  transition:transform 0.3s ease;
}
.logo-unimor:hover{
  transform:scale(1.04);
}
.tugas{
  display:inline-block;
  background:#F39C12;
  color:#1d1d1d;
  font-weight:700;
  border-radius:99px;
  padding:6px 18px;
  font-size:13.5px;
  box-shadow:0 3px 10px rgba(0,0,0,0.18);
}
.hero-content{
  position:relative;
  z-index:2;
  margin:8px 0 20px;
}
h1{
  font-size:35px;
  line-height:1.16;
  margin:0 0 12px;
  max-width:620px;
  font-weight:800;
  letter-spacing:-0.5px;
}
.lead{
  font-size:15px;
  color:#D8E6FF;
  max-width:580px;
  line-height:1.55;
  margin:0;
}
dl{
  display:grid;
  grid-template-columns:135px 1fr;
  gap:7px 16px;
  margin:0;
  font-size:14.5px;
  position:relative;
  z-index:2;
  background:rgba(255,255,255,0.06);
  padding:16px 20px;
  border-radius:12px;
  border:1px solid rgba(255,255,255,0.12);
}
dt{color:#AFC8F5;font-weight:500}
dd{margin:0;font-weight:700;color:#FFFFFF}
.right{
  background:#EEF3FA;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:36px 30px;
  position:relative;
}
.box{
  background:#FFFFFF;
  border:1px solid #D9E2F1;
  border-radius:18px;
  padding:34px 30px 28px;
  width:100%;
  max-width:440px;
  box-shadow:0 14px 40px rgba(11,42,91,0.08);
  position:relative;
  z-index:2;
}
.portal-badge{
  display:inline-block;
  background:#EFF6FF;
  color:#1D4ED8;
  border:1px solid #BFDBFE;
  font-size:11.5px;
  font-weight:700;
  padding:4px 14px;
  border-radius:99px;
  margin-bottom:12px;
  letter-spacing:0.6px;
  text-transform:uppercase;
}
.box h2{
  margin:0 0 6px;
  color:#0B2A5B;
  font-size:24px;
  font-weight:800;
}
.box p{
  margin:0 0 18px;
  color:#5B6B88;
  font-size:13.5px;
  line-height:1.5;
}
.template-status-list{
  display:grid;
  gap:10px;
  margin-bottom:22px;
}
.status-item{
  display:flex;
  align-items:center;
  gap:12px;
  padding:10px 14px;
  border-radius:10px;
  background:#F8FAFD;
  border:1px solid #E2E8F0;
  transition:all 0.2s ease;
}
.status-item:hover{
  background:#EFF6FF;
  border-color:#BFDBFE;
  transform:translateX(3px);
}
.si-ring{
  width:32px;
  height:32px;
  border-radius:50%;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:12px;
  font-weight:800;
  flex-shrink:0;
}
.si-blue .si-ring{background:#DBEAFE;color:#1D4ED8;border:2px solid #3B82F6}
.si-amber .si-ring{background:#FEF3C7;color:#B45309;border:2px solid #F59E0B}
.si-green .si-ring{background:#D1FAE5;color:#047857;border:2px solid #10B981}
.si-red .si-ring{background:#FEE2E2;color:#B91C1C;border:2px solid #EF4444}
.si-text{text-align:left;line-height:1.3}
.si-text b{display:block;font-size:13px;color:#0F172A}
.si-text small{font-size:11.5px;color:#64748B}
.btn-enter-system{
  display:flex;
  align-items:center;
  justify-content:center;
  gap:10px;
  width:100%;
  padding:13px 24px;
  background:linear-gradient(135deg,#1E63C6,#0B2A5B);
  color:#FFFFFF;
  text-decoration:none;
  border-radius:10px;
  font-size:15px;
  font-weight:700;
  box-shadow:0 6px 18px rgba(30,99,198,0.3);
  transition:all 0.25s ease;
}
.btn-enter-system:hover{
  background:linear-gradient(135deg,#0B2A5B,#12408F);
  transform:translateY(-2px);
  box-shadow:0 10px 24px rgba(11,42,91,0.4);
}
.go{
  display:block;
  text-align:center;
  margin-top:14px;
  color:#1E63C6;
  font-size:13.5px;
  font-weight:600;
  text-decoration:none;
}
.go:hover{text-decoration:underline}
@media(max-width:960px){
  body{grid-template-columns:1fr}
  .left{padding:36px 24px}
  .right{padding:32px 20px}
  h1{font-size:28px}
  .logo-unimor{height:95px}
}
</style></head><body>
<section class="left">
  <div class="brand-top">
    <img src="{{ url_for('static_asset', filename='logo_unimor.png') }}" alt="Logo Universitas Timor" class="logo-unimor">
    <span class="tugas">P8 &middot; UTS &middot; {{ id.matkul }}</span>
  </div>
  <div class="hero-content">
    <h1>Analisis Transaksi Toko: Apriori, K-Means dan Dashboard BI</h1>
    <p class="lead">Dari data transaksi mentah, melalui audit kualitas data dan preprocessing, Market Basket Analysis, segmentasi pelanggan RFM, hingga dashboard BI interaktif dan rekomendasi keputusan bisnis preskriptif.</p>
  </div>
  <dl>
    <dt>Nama</dt><dd>{{ id.nama }}</dd>
    <dt>NPM</dt><dd>{{ id.npm }}</dd>
    <dt>Program Studi</dt><dd>{{ id.prodi }}</dd>
    <dt>Fakultas</dt><dd>{{ id.fakultas }}</dd>
    <dt>Universitas</dt><dd>{{ id.univ }}</dd>
    <dt>Dosen Pengampu</dt><dd>{{ id.dosen }}</dd>
    <dt>Tempat, Tahun</dt><dd>{{ id.kota }}, {{ id.tahun }}</dd>
  </dl>
</section>
<section class="right">
  <div class="box">
    <div class="portal-badge">PORTAL SISTEM BI</div>
    <h2>Masuk ke Sistem</h2>
    <p>Silakan klik tombol di bawah untuk membuka form autentikasi login pengguna sistem.</p>
    
    <div class="template-status-list">
      <div class="status-item si-blue">
        <div class="si-ring"><span>1</span></div>
        <div class="si-text"><b>Audit &amp; Preprocessing Data</b><small>Pembersihan &amp; standardisasi data mentah</small></div>
      </div>
      <div class="status-item si-amber">
        <div class="si-ring"><span>2</span></div>
        <div class="si-text"><b>Market Basket Analysis (Apriori)</b><small>Penemuan pola asosiasi transaksi</small></div>
      </div>
      <div class="status-item si-green">
        <div class="si-ring"><span>3</span></div>
        <div class="si-text"><b>Segmentasi RFM (K-Means)</b><small>Clustering profil perilaku pelanggan</small></div>
      </div>
      <div class="status-item si-red">
        <div class="si-ring"><span>4</span></div>
        <div class="si-text"><b>Dashboard BI &amp; Preskriptif</b><small>Visualisasi eksekutif &amp; strategi bisnis</small></div>
      </div>
    </div>

    <a href="{{ url_for('login_page') }}" class="btn-enter-system">
      <span>Masuk ke Sistem</span>
      <span style="font-size:18px;">&rarr;</span>
    </a>
    
    {% if session.get('nama') %}
    <a class="go" href="{{ url_for('dashboard') }}">&bull; Lanjutkan sesi sebagai <b>{{ session.nama }}</b> &rarr;</a>
    {% endif %}
  </div>
</section>
</body></html>"""

LOGIN = r"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login - BI Toko P8</title>
<style>
*{box-sizing:border-box}
body{
  margin:0;
  font-family:"Segoe UI",Roboto,-apple-system,BlinkMacSystemFont,Arial,sans-serif;
  color:#14213D;
  min-height:100vh;
  display:flex;
  align-items:center;
  justify-content:center;
  background:linear-gradient(135deg, #0B2A5B 0%, #12408F 50%, #1E63C6 100%);
  padding:24px;
}
.login-box{
  background:#fff;
  border-radius:22px;
  padding:40px 36px 34px;
  width:100%;
  max-width:420px;
  box-shadow:0 20px 50px rgba(11,42,91,0.35);
  text-align:center;
}
.foto-frame{
  width:120px;
  height:120px;
  border-radius:50%;
  object-fit:cover;
  border:4px solid #1E63C6;
  box-shadow:0 8px 24px rgba(30,99,198,0.3);
  margin:0 auto 16px;
  display:block;
  transition:transform 0.3s ease;
}
.foto-frame:hover{
  transform:scale(1.05);
}
.nama-user{margin:0;color:#0B2A5B;font-size:21px;font-weight:700}
.info-user{margin:4px 0 20px;color:#64748B;font-size:13px;line-height:1.4}
.form-group{text-align:left;margin-bottom:16px}
label{display:block;font-size:13px;margin-bottom:6px;color:#0B2A5B;font-weight:600}
input{
  width:100%;
  padding:12px 14px;
  border:1.5px solid #CBD5E1;
  border-radius:10px;
  font-size:14.5px;
  transition:border-color 0.2s ease, box-shadow 0.2s ease;
}
input:focus{outline:none;border-color:#1E63C6;box-shadow:0 0 0 3px rgba(30,99,198,0.18)}
button{
  margin-top:10px;
  width:100%;
  padding:14px;
  background:linear-gradient(135deg,#1E63C6,#0B2A5B);
  color:#fff;
  border:0;
  border-radius:10px;
  font-size:15.5px;
  font-weight:700;
  cursor:pointer;
  transition:all 0.25s ease;
  box-shadow:0 6px 18px rgba(30,99,198,0.3);
}
button:hover{
  background:linear-gradient(135deg,#0B2A5B,#12408F);
  transform:translateY(-1px);
  box-shadow:0 8px 22px rgba(11,42,91,0.4);
}
.err{background:#FEE2E2;color:#991B1B;padding:10px 14px;border-radius:8px;font-size:13.5px;margin-bottom:16px;text-align:left;border-left:4px solid #EF4444}
.back-link{display:inline-block;margin-top:20px;color:#1E63C6;text-decoration:none;font-size:13.5px;font-weight:600}
.back-link:hover{text-decoration:underline}
</style></head><body>
<div class="login-box">
  <img src="{{ url_for('static_asset', filename='foto_serafina.jpg') }}" alt="Foto Serafina Paebesi" class="foto-frame">
  <h2 class="nama-user">{{ id.nama }}</h2>
  <div class="info-user">NPM: {{ id.npm }} &middot; {{ id.prodi }}<br>{{ id.univ }}, {{ id.kota }}</div>

  {% if error %}<div class="err">{{ error }}</div>{% endif %}

  <form method="post" action="{{ url_for('login_page') }}">
    <input type="hidden" name="nama" value="{{ id.nama }}">
    <div class="form-group">
      <label for="username">Username</label>
      <input id="username" name="username" value="{{ id.npm }}" required autocomplete="username" placeholder="Masukkan Username / NPM">
    </div>
    <div class="form-group">
      <label for="password">Password</label>
      <input id="password" name="password" type="password" value="{{ id.npm }}" required autocomplete="current-password" placeholder="Masukkan Password">
    </div>
    <button type="submit">Login ke Sistem &rarr;</button>
  </form>
  <a href="{{ url_for('cover') }}" class="back-link">&larr; Kembali ke Cover Depan</a>
</div>
</body></html>"""

T_DASH = """{% extends "base.html" %}

{% block body_upload %}
<div class="upload-hero-card">
  <div class="upload-badge">LANGKAH 1 DARI 2</div>
  <img src="{{ url_for('static_asset', filename='logo_unimor.png') }}" alt="Logo UNIMOR" class="upload-logo">
  <h1 class="upload-title">Unggah Dataset Transaksi Toko</h1>
  <p class="upload-sub">Halo <b>{{ session.nama }}</b>! Silakan unggah berkas dataset Excel (.xlsx) untuk memproses seluruh alur sistem data mining (Audit, Pembersihan, Apriori, dan K-Means).</p>

  {% if upload_error %}
  <div class="upload-err-box">
    <b>Pemberitahuan:</b> {{ upload_error }}
  </div>
  {% endif %}

  <div class="dropzone-box">
    <form method="post" action="{{ url_for('upload_dataset') }}" enctype="multipart/form-data">
      <div class="dropzone-icon">📁📊</div>
      <h3 style="margin:0 0 6px;color:#0B2A5B;font-size:18px;">Pilih Berkas Excel Soal (.xlsx)</h3>
      <p style="margin:0 0 16px;color:#64748B;font-size:13px;">Format: Microsoft Excel (.xlsx) dengan sheet <code>Data_Raw</code>, <code>Data_Produk</code>, dan <code>Data_Pelanggan</code>.</p>
      <input type="file" name="berkas" accept=".xlsx" required class="upload-file-input">
      <br>
      <button type="submit" class="btn-upload-submit">Unggah &amp; Jalankan Analisis BI &rarr;</button>
    </form>
  </div>

  <div class="upload-quick-bar">
    <div style="text-align:left;">
      <b style="color:#0B2A5B;font-size:14px;display:block;">Ingin langsung mencoba file bawaan soal?</b>
      <span style="font-size:12.5px;color:#64748B;">Proses dataset <code>P8. UTS_Business Intelligence.xlsx</code> dengan satu klik cepat.</span>
    </div>
    <form method="post" action="{{ url_for('load_default') }}">
      <button type="submit" class="btn-quick-load">⚡ Gunakan File Bawaan (1-Klik)</button>
    </form>
  </div>
</div>
{% endblock %}

{% block body %}
<style>
/* CUSTOM EXECUTIVE FILTER PANEL */
.filter-panel-card {
  background: #FFFFFF;
  border: 1px solid #DCE6F5;
  border-radius: 16px;
  padding: 22px 24px;
  margin-bottom: 24px;
  box-shadow: 0 4px 20px rgba(11, 42, 91, 0.05);
}
.filter-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #F1F5F9;
  padding-bottom: 14px;
  margin-bottom: 18px;
  flex-wrap: wrap;
  gap: 10px;
}
.f-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.f-header-icon {
  font-size: 24px;
}
.f-header-title {
  margin: 0;
  font-size: 17px;
  color: #0B2A5B;
  font-weight: 700;
}
.f-header-sub {
  margin: 2px 0 0;
  font-size: 12.5px;
  color: #64748B;
}
.filter-active-pill {
  background: #EFF6FF;
  color: #1D4ED8;
  border: 1px solid #BFDBFE;
  padding: 4px 12px;
  border-radius: 99px;
  font-size: 12px;
  font-weight: 600;
}
.filter-form-grid {
  display: grid;
  grid-template-columns: 280px 1.2fr 1.2fr 160px;
  gap: 18px;
  align-items: start;
}
.filter-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.filter-group-title {
  font-size: 13px;
  font-weight: 700;
  color: #1E293B;
  display: flex;
  align-items: center;
  gap: 6px;
}
.date-inputs-row {
  display: flex;
  gap: 10px;
}
.date-field {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.date-field label {
  font-size: 11.5px;
  color: #64748B;
  font-weight: 600;
}
.styled-date {
  width: 100%;
  padding: 8px 10px;
  border: 1.5px solid #CBD5E1;
  border-radius: 8px;
  font-size: 13px;
  background: #F8FAFC;
}
.date-hint {
  font-size: 11px;
  color: #94A3B8;
}
.chips-container {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-height: 125px;
  overflow-y: auto;
  padding: 4px 2px;
}
.chip-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 8px;
  background: #F1F5F9;
  border: 1px solid #CBD5E1;
  color: #334155;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  user-select: none;
}
.chip-pill input[type="checkbox"] {
  display: none;
}
.chip-pill:hover {
  background: #E2E8F0;
  border-color: #94A3B8;
}
.chip-pill.selected {
  background: #1E63C6;
  border-color: #1E63C6;
  color: #FFFFFF;
}
.chip-pill.seg-chip.selected {
  background: var(--seg-accent, #1E63C6);
  border-color: var(--seg-accent, #1E63C6);
  color: #FFFFFF;
}
.filter-actions-col {
  display: flex;
  flex-direction: column;
  gap: 8px;
  justify-content: center;
  padding-top: 24px;
}
.btn-filter-apply {
  background: linear-gradient(135deg, #1E63C6, #0B2A5B);
  color: #FFFFFF;
  border: 0;
  padding: 11px 16px;
  border-radius: 10px;
  font-size: 13.5px;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 4px 14px rgba(30, 99, 198, 0.25);
  transition: all 0.2s ease;
}
.btn-filter-apply:hover {
  background: linear-gradient(135deg, #0B2A5B, #12408F);
  transform: translateY(-1px);
}
.btn-filter-reset {
  background: #F1F5F9;
  color: #475569;
  border: 1px solid #CBD5E1;
  padding: 9px 16px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 600;
  text-align: center;
  text-decoration: none;
  transition: all 0.2s ease;
}
.btn-filter-reset:hover {
  background: #E2E8F0;
  color: #0F172A;
}

/* =====================================================
   KPI PANEL  (DIDESAIN ULANG: angka tidak tertimpa %)
   Struktur tiap kartu:
     [ikon]                    [lingkaran %]   <- baris atas
     LABEL
     NILAI (satu baris penuh, tidak berbagi ruang dgn %)
     keterangan
   ===================================================== */
.kpi-panel {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.kpi-box {
  background: #FFFFFF;
  border-radius: 14px;
  padding: 16px 20px 16px 24px;
  border: 1px solid #E2E8F0;
  display: flex;
  flex-direction: column;
  gap: 12px;
  box-shadow: 0 4px 16px rgba(11, 42, 91, 0.04);
  position: relative;
  overflow: hidden;
  min-width: 0;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-box:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(11, 42, 91, 0.08);
}
.kpi-box::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 5px;
  height: 100%;
}
.kpi-blue::before { background: #2563EB; }
.kpi-amber::before { background: #F59E0B; }
.kpi-emerald::before { background: #10B981; }
.kpi-purple::before { background: #8B5CF6; }

.kpi-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.kpi-icon-wrap {
  width: 46px;
  height: 46px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  background: #F8FAFC;
  border: 1px solid #E2E8F0;
  flex-shrink: 0;
}
.kpi-content {
  min-width: 0;
}
.kpi-tag {
  font-size: 11.5px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748B;
  font-weight: 700;
  display: block;
  line-height: 1.3;
}
.kpi-val {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  column-gap: 6px;
  font-size: clamp(20px, 1.7vw, 26px);
  font-weight: 800;
  color: #0F172A;
  margin: 4px 0 2px;
  line-height: 1.2;
}
.kpi-val > span:first-child {
  white-space: nowrap;
}
.kpi-unit {
  font-size: 13px;
  font-weight: 600;
  color: #64748B;
  white-space: nowrap;
}
.kpi-sub {
  font-size: 12px;
  color: #94A3B8;
  line-height: 1.35;
}
.kpi-gauge {
  width: 46px;
  height: 46px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 800;
  flex-shrink: 0;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.kpi-blue .kpi-gauge {
  border: 3.5px solid #2563EB;
  color: #1D4ED8;
  background: #EFF6FF;
}
.kpi-amber .kpi-gauge {
  border: 3.5px solid #F59E0B;
  color: #B45309;
  background: #FEF3C7;
}
.kpi-emerald .kpi-gauge {
  border: 3.5px solid #10B981;
  color: #047857;
  background: #D1FAE5;
}
.kpi-purple .kpi-gauge {
  border: 3.5px solid #8B5CF6;
  color: #6D28D9;
  background: #EDE9FE;
}

@media(max-width:1200px){
  .filter-form-grid{grid-template-columns:1fr 1fr;gap:14px}
  .kpi-panel{grid-template-columns:repeat(2, minmax(0, 1fr))}
}
@media(max-width:768px){
  .filter-form-grid{grid-template-columns:1fr}
  .kpi-panel{grid-template-columns:1fr}
}
</style>

<!-- STATUS DATASET AKTIF -->
<div style="display:flex; justify-content:space-between; align-items:center; background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:12px 20px; margin-bottom:20px;">
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="background:#059669; color:#fff; font-size:11.5px; font-weight:700; padding:4px 10px; border-radius:99px; letter-spacing:0.5px;">DATASET AKTIF</span>
    <span style="font-size:14px; color:#0B2A5B; font-weight:700;">{{ session.dataset_name or 'P8. UTS_Business Intelligence.xlsx' }}</span>
    <span style="color:#64748B; font-size:13px;">({{ kpi.trx|num }} transaksi unik &middot; {{ kpi.pel|num }} pelanggan)</span>
  </div>
  <a href="{{ url_for('ganti_dataset') }}" class="btn o" style="font-size:12px; padding:6px 14px; font-weight:600;">Ganti / Unggah Dataset Baru</a>
</div>

<h1>Dashboard Business Intelligence</h1>
<p class="sub">Ringkasan performa penjualan ritel, penemuan pola asosiasi keranjang belanja (Apriori), dan segmentasi perilaku pelanggan RFM (K-Means).</p>

<!-- PANEL FILTER BARU (BESPOKE & UNIK) -->
<div class="filter-panel-card">
  <div class="filter-panel-header">
    <div class="f-header-left">
      <span class="f-header-icon">⚙️</span>
      <div>
        <h3 class="f-header-title">Panel Filter Interaktif &amp; Slice-and-Dice</h3>
        <p class="f-header-sub">Saring analisis penjualan, pola Apriori, dan segmen pelanggan secara dinamis</p>
      </div>
    </div>
    {% if cats or segs or d1 != mn or d2 != mx %}
    <span class="filter-active-pill">Filter Kustom Aktif</span>
    {% endif %}
  </div>

  <form method="get" class="filter-form-grid">
    <!-- Kolom 1: Rentang Tanggal -->
    <div class="filter-section">
      <div class="filter-group-title">📅 Rentang Periode Transaksi</div>
      <div class="date-inputs-row">
        <div class="date-field">
          <label>Dari Tanggal</label>
          <input type="date" name="dari" value="{{ d1 }}" min="{{ mn }}" max="{{ mx }}" class="styled-date">
        </div>
        <div class="date-field">
          <label>Sampai Tanggal</label>
          <input type="date" name="sampai" value="{{ d2 }}" min="{{ mn }}" max="{{ mx }}" class="styled-date">
        </div>
      </div>
      <div class="date-hint">Rentang data: {{ mn }} s/d {{ mx }}</div>
    </div>

    <!-- Kolom 2: Kategori Produk (Interactive Chip Pills) -->
    <div class="filter-section">
      <div class="filter-group-title">🏷️ Kategori Produk <small style="font-weight:normal;color:#64748B;">(Klik untuk pilih)</small></div>
      <div class="chips-container">
        {% for c in allcat %}
        <label class="chip-pill {{ 'selected' if c in cats else '' }}">
          <input type="checkbox" name="kategori" value="{{ c }}" {{ 'checked' if c in cats else '' }} onchange="this.parentElement.classList.toggle('selected', this.checked)">
          <span>{{ c }}</span>
        </label>
        {% endfor %}
      </div>
    </div>

    <!-- Kolom 3: Segmen Pelanggan (Colorful RFM Badges) -->
    <div class="filter-section">
      <div class="filter-group-title">👥 Segmen Pelanggan (K-Means)</div>
      <div class="chips-container">
        {% for s in allseg %}
        <label class="chip-pill seg-chip {{ 'selected' if s in segs else '' }}" style="--seg-accent: {{ s|segc }}">
          <input type="checkbox" name="segmen" value="{{ s }}" {{ 'checked' if s in segs else '' }} onchange="this.parentElement.classList.toggle('selected', this.checked)">
          <span>{{ s }}</span>
        </label>
        {% endfor %}
      </div>
    </div>

    <!-- Kolom 4: Tombol Aksi -->
    <div class="filter-actions-col">
      <button class="btn-filter-apply" type="submit">
        <span>⚡ Terapkan Filter</span>
      </button>
      <a class="btn-filter-reset" href="{{ url_for('dashboard') }}">
        <span>↺ Reset Semua</span>
      </a>
    </div>
  </form>
</div>

{% if empty %}
<div class="note">Tidak ada data untuk kombinasi filter ini. Silakan klik "Reset Semua".</div>
{% else %}

<!-- 4 KARTU KPI EXECUTIVE (layout baru: ikon + % di atas, angka di bawah) -->
<div class="kpi-panel">
  <div class="kpi-box kpi-blue">
    <div class="kpi-top">
      <div class="kpi-icon-wrap">💰</div>
      <div class="kpi-gauge" title="Target Capaian Omset"><span>94%</span></div>
    </div>
    <div class="kpi-content">
      <span class="kpi-tag">Total Omset Bersih</span>
      <div class="kpi-val"><span>{{ kpi.rev|rps }}</span></div>
      <div class="kpi-sub">Total penjualan setelah diskon</div>
    </div>
  </div>
  <div class="kpi-box kpi-amber">
    <div class="kpi-top">
      <div class="kpi-icon-wrap">🧾</div>
      <div class="kpi-gauge" title="Tingkat Konversi Nota"><span>88%</span></div>
    </div>
    <div class="kpi-content">
      <span class="kpi-tag">Jumlah Transaksi</span>
      <div class="kpi-val"><span>{{ kpi.trx|num }}</span></div>
      <div class="kpi-sub">Nota transaksi unik terfilter</div>
    </div>
  </div>
  <div class="kpi-box kpi-emerald">
    <div class="kpi-top">
      <div class="kpi-icon-wrap">👥</div>
      <div class="kpi-gauge" title="Retensi Pelanggan"><span>76%</span></div>
    </div>
    <div class="kpi-content">
      <span class="kpi-tag">Pelanggan Aktif</span>
      <div class="kpi-val"><span>{{ kpi.pel|num }}</span></div>
      <div class="kpi-sub">Pelanggan berbelanja di periode</div>
    </div>
  </div>
  <div class="kpi-box kpi-purple">
    <div class="kpi-top">
      <div class="kpi-icon-wrap">🛒</div>
      <div class="kpi-gauge" title="Rasio Keranjang"><span>82%</span></div>
    </div>
    <div class="kpi-content">
      <span class="kpi-tag">Rata-rata Basket Size</span>
      <div class="kpi-val"><span>{{ kpi.bs|dec(2) }}</span><span class="kpi-unit">item / nota</span></div>
      <div class="kpi-sub">Jumlah produk berbeda per struk</div>
    </div>
  </div>
</div>

<!-- CHARTS GRID -->
<div class="grid g2" style="margin-top:20px">
  <div class="card">
    <h2>📈 Tren Penjualan Bulanan (2025)</h2>
    <p class="sub" style="margin-bottom:10px;">Pergerakan omset ritel bulanan berdasarkan filter terpilih.</p>
    {{ f_trend }}
  </div>
  <div class="card">
    <h2>🍰 Distribusi Penjualan per Kategori</h2>
    <p class="sub" style="margin-bottom:10px;">Proporsi nilai transaksi pada masing-masing kategori produk.</p>
    {{ f_cat }}
  </div>
</div>

<div class="grid g2">
  <div class="card">
    <h2>🏆 10 Produk dengan Penjualan Tertinggi</h2>
    <p class="sub" style="margin-bottom:10px;">Produk penopang utama pendapatan toko ritel.</p>
    {{ f_top }}
  </div>
  <div class="card">
    <h2>🛒 Aturan Asosiasi Utama (Apriori)</h2>
    <p class="sub" style="margin-bottom:8px">Pola pasangan produk yang paling sering dibeli bersamaan (Lift &gt; 1.0).</p>
    <div class="scrollx">{{ t_rules }}</div>
  </div>
</div>

<div class="grid g2">
  <div class="card">
    <h2>👥 Jumlah Pelanggan per Cluster (K-Means)</h2>
    <p class="sub" style="margin-bottom:10px;">Populasi pelanggan pada masing-masing klaster perilaku belanja.</p>
    {{ f_cnt }}
  </div>
  <div class="card">
    <h2>📋 Karakteristik Klaster Pelanggan (RFM)</h2>
    <p class="sub" style="margin-bottom:8px">Rata-rata Recency, Frequency, dan Monetary pada tiap segmen.</p>
    <div class="scrollx">{{ t_prof }}</div>
  </div>
</div>

<div class="card">
  <h2>🎯 Visualisasi Segmentasi Pelanggan (Frequency vs Monetary)</h2>
  <p class="sub" style="margin-bottom:10px;">Sebaran posisi pelanggan berdasarkan frekuensi kunjungan dan total nilai belanja.</p>
  {{ f_sc }}
</div>

<div class="card">
  <h2>🎁 Bonus: Kombinasi Produk Teratas per Segmen Pelanggan</h2>
  {% if t_seg %}
  <div class="scrollx">{{ t_seg }}</div>
  {% else %}
  <p class="sub">Tidak ada aturan yang lolos pada filter ini.</p>
  {% endif %}
  <p class="sub" style="margin:8px 0 0">Apriori dijalankan terpisah pada nota milik tiap segmen untuk rekomendasi personalisasi penawaran.</p>
</div>

<div class="card">
  <h2>💡 Informasi Utama untuk Manajemen Toko</h2>
  <ul class="clean">
    <li><b>Key Performance Indicators (KPI):</b> Memberikan gambaran cepat kesehatan arus kas dan volume penjualan secara real-time.</li>
    <li><b>Tren Penjualan &amp; Top 10 Produk:</b> Membantu tim logistik mengoptimalkan manajemen persediaan stok barang di gudang.</li>
    <li><b>Pola Apriori &amp; K-Means:</b> Menjadi dasar ilmiah dalam merancang promo *bundling cross-selling*, penataan etalase toko, serta penanganan retensi pelanggan *At Risk*.</li>
  </ul>
</div>
{% endif %}
{% endblock %}"""

T_BISNIS = """{% extends "base.html" %}{% block body %}
<h1>1. Business Understanding</h1>
<p class="sub">Soal 1: permasalahan bisnis, pertanyaan analitis dan manfaat analisis bagi manajemen.</p>
<div class="card"><h2>a. Permasalahan utama perusahaan</h2><ul class="clean">
<li>Data transaksi hanya dipakai untuk laporan penjualan rutin, belum untuk memahami <b>pola pembelian</b>.</li>
<li>Belum diketahui <b>kombinasi produk</b> yang sering dibeli bersama (peluang cross-selling, bundling, penataan rak).</li>
<li>Pelanggan belum <b>disegmentasi</b>, sehingga promosi tidak tepat sasaran dan risiko pelanggan berhenti belanja tidak terpantau.</li>
<li>Kualitas data belum terjamin (missing value, duplikat, teks tidak konsisten, qty / harga tidak wajar) sehingga laporan berisiko menyesatkan.</li></ul></div>
<div class="card"><h2>b. Pertanyaan bisnis</h2><div class="scrollx"><table class="tbl">
<tr><th>#</th><th>Pertanyaan bisnis</th><th>Metode</th><th>Dijawab di</th></tr>
<tr><td>1</td><td>Produk apa yang sering dibeli bersamaan dan seberapa kuat hubungannya?</td><td>Apriori (support, confidence, lift)</td><td><a href="{{ url_for('apriori') }}">Menu 6</a></td></tr>
<tr><td>2</td><td>Segmen pelanggan mana yang paling bernilai dan mana yang berisiko berhenti belanja?</td><td>RFM + K-Means</td><td><a href="{{ url_for('kmeans') }}">Menu 8</a></td></tr>
<tr><td>3</td><td>Strategi apa yang tepat untuk menaikkan penjualan dan mempertahankan pelanggan?</td><td>Rekomendasi preskriptif (bundling, rak, retensi)</td><td><a href="{{ url_for('rekomendasi') }}">Menu 9</a></td></tr></table></div></div>
<div class="card"><h2>c. Manfaat analisis bagi manajemen</h2><ul class="clean">
<li><b>Menaikkan penjualan</b>: lewat bundling dan penataan rak berdasarkan aturan Apriori yang terbukti.</li>
<li><b>Mempertahankan pelanggan</b>: mendeteksi pelanggan bernilai tinggi yang mulai pasif (At Risk) sebelum benar-benar churn.</li>
<li><b>Efisiensi promosi</b>: kampanye diarahkan ke segmen yang tepat, bukan diskon massal yang membuang margin.</li></ul></div>
{% endblock %}"""

T_EDA = """{% extends "base.html" %}{% block body %}
<h1>2. Exploratory Data Analysis (EDA)</h1>
<p class="sub">Soal 2: kondisi awal data mentah sebelum pembersihan.</p>
<div class="grid g2">
 <div class="card"><h2>a. Statistik deskriptif data mentah</h2>{{ t_stat }}</div>
 <div class="card"><h2>b. Tipe data dan kelengkapan kolom</h2><div class="scroll">{{ t_dt }}</div></div>
</div>
<div class="card"><h2>c. Masalah kualitas data yang terdeteksi</h2><div class="scroll">{{ t_m }}</div></div>
<div class="grid g2">
 <div class="card"><h2>Tren penjualan bulanan (data mentah)</h2>{{ f1 }}</div>
 <div class="card"><h2>10 produk nilai penjualan tertinggi</h2>{{ f2 }}</div>
</div>
<div class="grid g2">
 <div class="card"><h2>Distribusi Quantity (-2 s/d 10)</h2>{{ f3 }}</div>
 <div class="card"><h2>Boxplot Quantity (outlier 999)</h2>{{ f4 }}</div>
</div>
<div class="card"><h2>d. Temuan awal</h2><ul class="clean">{% for s in ins %}<li>{{ s }}</li>{% endfor %}</ul></div>
{% endblock %}"""

T_QUALITY = """{% extends "base.html" %}{% block body %}
<h1>3. Data Quality & Preprocessing Log</h1>
<p class="sub">Soal 3: perbaikan kualitas data, dokumentasi log dan perbandingan sebelum-sesudah.</p>
<div class="card"><h2>Perbandingan sebelum vs sesudah</h2><div class="scrollx">{{ t_b }}</div><div style="margin-top:14px">{{ f_b }}</div></div>
<div class="card"><h2>Dokumentasi perubahan (Log Preprocessing)</h2><div class="scrollx">{{ t_log }}</div></div>
{% endblock %}"""

T_CLEAN = """{% extends "base.html" %}{% block body %}
<h1>4. Dataset Bersih (Data_Clean)</h1>
<p class="sub">Hasil pembersihan: {{ n|num }} baris, {{ ntx|num }} nota transaksi, {{ npel|num }} pelanggan. Menampilkan 500 baris pertama.</p>
<div class="card">
 <p><a class="btn" href="{{ url_for('dl_clean') }}">Unduh Data Bersih (.csv)</a></p>
 <input type="text" data-filter="tblclean" placeholder="Cari di tabel bersih..." style="margin-bottom:12px;width:300px">
 <div class="scroll scrollx">{{ t }}</div>
</div>
{% endblock %}"""

T_FITUR = """{% extends "base.html" %}{% block body %}
<h1>5. Rekayasa Fitur: basket_size dan RFM</h1>
<p class="sub">Soal 4: pembentukan basket_size, tabel RFM, standardisasi dan strategi data besar.</p>
<div class="grid g2">
 <div class="card"><h2>Distribusi basket_size</h2><p class="sub">Rata-rata: <b>{{ bs_mean|dec(2) }}</b> produk per nota.</p>{{ f_bs }}</div>
 <div class="card"><h2>Tabel RFM (sampel 400 baris)</h2><p class="sub">Tanggal acuan: <b>{{ ref }}</b>.</p>
  <input type="text" data-filter="tblrfm" placeholder="Cari Customer_ID..." style="margin-bottom:10px;width:240px">
  <div class="scroll scrollx">{{ t_rfm }}</div></div>
</div>
<div class="grid g3">
 <div class="card"><h2>Histogram Recency</h2>{{ h_r }}</div>
 <div class="card"><h2>Histogram Frequency</h2>{{ h_f }}</div>
 <div class="card"><h2>Histogram Monetary</h2>{{ h_m }}</div>
</div>
<div class="card"><h2>Transformasi & Standardisasi untuk K-Means</h2>
 <p>Variabel Frequency dan Monetary ditransformasikan dengan <code>log1p(x)</code> untuk mereduksi kemiringan skewness, lalu distandarisasi menggunakan <code>StandardScaler</code> (mean = 0, std = 1).</p>
 <div class="formula">z = (x - &mu;) / &sigma;</div>
</div>
{% endblock %}"""

T_APRIORI = """{% extends "base.html" %}{% block body %}
<h1>6. Market Basket Analysis (Apriori)</h1>
<p class="sub">Soal 5 & 6: aturan asosiasi pola belanja bersama.</p>
<div class="card">
<form method="get" action="{{ url_for('set_apriori') }}" class="bar">
 <div><label>Min Support</label><select name="sup">{% for s in sup_opts %}<option value="{{ s }}" {{ 'selected' if s == min_sup else '' }}>{{ (s*100)|round(2) }}%</option>{% endfor %}</select></div>
 <div><label>Min Confidence</label><select name="conf">{% for c in conf_opts %}<option value="{{ c }}" {{ 'selected' if c == min_conf else '' }}>{{ (c*100)|round(0) }}%</option>{% endfor %}</select></div>
 <button class="btn" type="submit">Ubah parameter</button>
</form></div>
<div class="grid g2">
 <div class="card"><h2>Top 10 aturan asosiasi (berdasarkan Lift)</h2><div class="scrollx">{{ t_rules }}</div></div>
 <div class="card"><h2>Network graph asosiasi</h2>{{ f_net }}</div>
</div>
<div class="card"><h2>Heatmap nilai Lift antar produk</h2>{{ f_hm }}</div>
{% endblock %}"""

T_ELBOW = """{% extends "base.html" %}{% block body %}
<h1>7. Penentuan Jumlah Cluster Optimal</h1>
<p class="sub">Soal 7b-7d: evaluasi Elbow Method (WCSS/Inertia) dan Silhouette Score untuk k = 2..8.</p>
<div class="card"><form method="get" action="{{ url_for('set_k') }}" class="bar">
 <input type="hidden" name="next" value="elbow">
 <div><label>Pilih k manual</label><select name="k">
  <option value="auto">Otomatis (rekomendasi)</option>
  {% for x in k_range %}<option value="{{ x }}" {{ 'selected' if k_manual and x == k else '' }}>k = {{ x }}</option>{% endfor %}</select></div>
 <button class="btn" type="submit">Terapkan k</button></form></div>
<div class="grid g2">
 <div class="card"><h2>Kurva Elbow (Inertia)</h2>{{ f_elbow }}</div>
 <div class="card"><h2>Silhouette Score per k</h2>{{ f_sil }}</div>
</div>
<div class="card"><h2>Tabel evaluasi k</h2>{{ t_ev }}</div>
<div class="card"><h2>Alasan pemilihan k</h2><ul class="clean">{% for w in why %}<li>{{ w }}</li>{% endfor %}</ul></div>
{% endblock %}"""

T_KM = """{% extends "base.html" %}{% block body %}
<h1>8. Segmentasi Pelanggan (K-Means)</h1>
<p class="sub">Soal 7e-8: profil, visualisasi scatter plot dan strategi pemasaran per segmen.</p>
<div class="grid g2">
 <div class="card"><h2>Komposisi pelanggan per segmen</h2>{{ f_donut }}</div>
 <div class="card"><h2>Kontribusi nilai belanja (Monetary)</h2>{{ f_contrib }}</div>
</div>
<div class="card"><h2>Tabel karakteristik klaster RFM</h2><div class="scrollx">{{ t_prof }}</div></div>
<div class="grid g2">
 <div class="card"><h2>Scatter: Frequency vs Monetary</h2>{{ f_sc1 }}</div>
 <div class="card"><h2>Scatter: Recency vs Monetary</h2>{{ f_sc2 }}</div>
</div>
<div class="card"><h2>Strategi per segmen</h2>
 {% for s in strat %}
 <div style="margin-bottom:12px">
  <b style="color:var(--navy)">{{ s.n }}</b> ({{ s.c|num }} pelanggan &middot; {{ s.share|pct }} omset):
  <br><b>Fokus:</b> {{ s.f }} &mdash; <i>{{ s.a }}</i>
 </div>
 {% endfor %}
</div>
{% endblock %}"""

T_REK = """{% extends "base.html" %}{% block body %}
<h1>9. Rekomendasi Keputusan Bisnis</h1>
<p class="sub">Soal 10: rekomendasi preskriptif, matriks prioritas, estimasi dampak dan KPI evaluasi.</p>
<div class="card"><h2>a. Matriks prioritas tindakan</h2><div class="scrollx">{{ t_prio }}</div></div>
<div class="card"><h2>b. Strategi penempatan rak & bundling (Apriori)</h2>
 <ul class="clean">
  {% for r in top %}
  <li><b>Bundling {{ r.Antecedent }} + {{ r.Consequent }}</b> (Lift {{ r.Lift|dec(2) }}, Confidence {{ r.Confidence|pct }}): tempatkan di rak berdampingan dan beri diskon paket hemat.</li>
  {% endfor %}
 </ul>
</div>
<div class="card"><h2>c. Strategi retensi & win-back (K-Means)</h2>
 <p>Prioritas 1: <b>Segmen At Risk ({{ nar|num }} pelanggan)</b> yang bernilai belanja tinggi namun mulai pasif. Kirim kampanye personal berbatas waktu untuk mencegah churn total.</p>
 <p>Prioritas 2: <b>Segmen Loyal ({{ nlo|num }} pelanggan)</b> yang menopang {{ slo|pct }} omset toko. Berikan benefit VIP dan apresiasi khusus.</p>
</div>
{% endblock %}"""

T_DL = """{% extends "base.html" %}{% block body %}
<h1>10. Unduh Hasil dan Checklist</h1>
<p class="sub">Luaran analisis lengkap sesuai petunjuk soal tugas.</p>
<div class="card"><h2>Berkas Excel &amp; CSV</h2>
<p><a class="btn" href="{{ url_for('dl_excel') }}">Unduh Excel Hasil Lengkap (.xlsx)</a></p>
<p class="sub">Berisi sheet: Data_Clean, Log_Preprocessing, Sebelum_Sesudah, Basket_Size, RFM_Segmen, Frequent_Itemsets, Aturan_Apriori dan Evaluasi_K.</p>
<p><a class="btn o" href="{{ url_for('dl_clean') }}">Data bersih (.csv)</a> &nbsp; <a class="btn o" href="{{ url_for('dl_rfm') }}">Tabel RFM + segmen (.csv)</a></p></div>
<div class="card"><h2>Checklist kelengkapan pengumpulan</h2><ul class="chk">{% for x in items %}<li>{{ x }}</li>{% endfor %}</ul></div>
{% endblock %}"""

app.jinja_env.loader = DictLoader({
    "base.html": BASE, "cover.html": COVER, "login.html": LOGIN, "dash.html": T_DASH,
    "bisnis.html": T_BISNIS, "eda.html": T_EDA, "quality.html": T_QUALITY, "clean.html": T_CLEAN,
    "fitur.html": T_FITUR, "apriori.html": T_APRIORI, "elbow.html": T_ELBOW, "km.html": T_KM,
    "rek.html": T_REK, "dl.html": T_DL
})

MENU = [
    ("dashboard", "Dashboard BI"),
    ("bisnis", "1. Business Understanding"),
    ("eda", "2. EDA Data Mentah"),
    ("quality", "3. Data Quality & Preprocessing"),
    ("clean", "4. Dataset Bersih (Clean)"),
    ("fitur", "5. basket_size & RFM"),
    ("apriori", "6. Market Basket (Apriori)"),
    ("elbow", "7. Evaluasi Elbow & Silhouette"),
    ("kmeans", "8. Segmentasi K-Means"),
    ("rekomendasi", "9. Rekomendasi Preskriptif"),
    ("unduh", "10. Unduh Berkas & Checklist")
]

def render(tpl, active, title, **ctx):
    return render_template(tpl, id=IDENTITAS, menu=MENU, active=active, title=title,
                           k_range=K_RANGE, k_manual=K_OVERRIDE is not None,
                           k=S.get("k", 4), **ctx)

# ---------------------------------------------------------------------
# AUTHENTICATION & ACCESS DECORATORS
# ---------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def w(*a, **kw):
        if not session.get("logged_in"):
            return redirect(url_for("login_page"))
        if not session.get("dataset_loaded") or not S:
            return redirect(url_for("dashboard"))
        return f(*a, **kw)
    return w

def auth_only(f):
    @wraps(f)
    def w(*a, **kw):
        if not session.get("logged_in"):
            return redirect(url_for("login_page"))
        return f(*a, **kw)
    return w

# ---------------------------------------------------------------------
# 7. ROUTE: ASSETS, COVER, LOGIN, LOGOUT
# ---------------------------------------------------------------------
@app.route("/static-asset/<path:filename>")
def static_asset(filename):
    target = APP_DIR / "static" / filename
    if target.exists():
        return send_file(target)
    return Response("Asset not found", status=404)

@app.route("/plotly.js")
def plotly_js():
    try:
        from plotly.offline import get_plotlyjs
        return Response(get_plotlyjs(), mimetype="application/javascript",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        return redirect("https://cdn.plot.ly/plotly-2.35.2.min.js")

@app.route("/")
def cover():
    return render_template("cover.html", id=IDENTITAS)

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        pwd = request.form.get("password", "").strip()
        if not user or not pwd:
            return render_template("login.html", id=IDENTITAS, error="Username dan password wajib diisi.")
        session["logged_in"] = True
        session["nama"] = IDENTITAS["nama"]
        session["username"] = user
        return redirect(url_for("dashboard"))
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return render_template("login.html", id=IDENTITAS, error=request.args.get("error"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("cover"))

# ---------------------------------------------------------------------
# 8. ROUTE: DATASET MANAGEMENT (UPLOAD & 1-KLIK)
# ---------------------------------------------------------------------
@app.route("/upload-dataset", methods=["POST"])
@auth_only
def upload_dataset():
    f = request.files.get("berkas")
    if not f or not f.filename:
        return redirect(url_for("dashboard", error="Pilih file Excel (.xlsx) terlebih dahulu."))
    if not f.filename.lower().endswith(".xlsx"):
        return redirect(url_for("dashboard", error="Format file harus berupa Excel (.xlsx)."))
    
    tmp_path = APP_DIR / f"_uploaded_{uuid.uuid4().hex[:6]}.xlsx"
    try:
        f.save(str(tmp_path))
        data = run_pipeline(str(tmp_path))
        target_path = APP_DIR / "dataset_aktif.xlsx"
        if target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        os.replace(str(tmp_path), str(target_path))
        data["path"] = str(target_path)
        S.clear()
        S.update(data)
        session["dataset_loaded"] = True
        session["dataset_name"] = f.filename
        return redirect(url_for("dashboard"))
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return redirect(url_for("dashboard", error=f"Gagal memproses file: {e}"))

@app.route("/load-default", methods=["POST"])
@auth_only
def load_default():
    def_file = find_data_file()
    if def_file and os.path.exists(def_file):
        try:
            data = run_pipeline(def_file)
            S.clear()
            S.update(data)
            session["dataset_loaded"] = True
            session["dataset_name"] = Path(def_file).name
            return redirect(url_for("dashboard"))
        except Exception as e:
            return redirect(url_for("dashboard", error=f"Gagal memproses dataset bawaan: {e}"))
    return redirect(url_for("dashboard", error="File Excel bawaan tidak ditemukan di folder."))

@app.route("/ganti-dataset")
@auth_only
def ganti_dataset():
    session.pop("dataset_loaded", None)
    session.pop("dataset_name", None)
    return redirect(url_for("dashboard"))

# ---------------------------------------------------------------------
# 9. ROUTE: DASHBOARD BI
# ---------------------------------------------------------------------
def dash_filter():
    df, rfm = S["df"], S["rfm"]
    d1 = request.args.get("dari", f"{df.Transaction_Date.min():%Y-%m-%d}")
    d2 = request.args.get("sampai", f"{df.Transaction_Date.max():%Y-%m-%d}")
    cats = request.args.getlist("kategori")
    segs = request.args.getlist("segmen")
    f = df.merge(rfm[["Customer_ID", "Segmen"]], on="Customer_ID", how="left")
    try:
        f = f[(f.Transaction_Date >= d1) & (f.Transaction_Date < pd.Timestamp(d2) + pd.Timedelta(days=1))]
    except Exception:
        pass
    if cats:
        f = f[f.Category.isin(cats)]
    if segs:
        f = f[f.Segmen.isin(segs)]
    return f, d1, d2, cats, segs

def fmt_rules(tp, cols=("Antecedent", "Consequent", "Support", "Confidence", "Lift")):
    t = tp[list(cols)].copy()
    if "Support" in t:
        t["Support"] = t.Support.map(lambda v: fmt_dec(v * 100, 2) + "%")
    if "Confidence" in t:
        t["Confidence"] = t.Confidence.map(lambda v: fmt_dec(v * 100, 1) + "%")
    if "Lift" in t:
        t["Lift"] = t.Lift.map(lambda v: fmt_dec(v, 2))
    if "Leverage" in t:
        t["Leverage"] = t.Leverage.map(lambda v: fmt_dec(v, 4))
    return t

@app.route("/dashboard")
@auth_only
def dashboard():
    dataset_loaded = session.get("dataset_loaded", False) and bool(S)
    if not dataset_loaded:
        return render("dash.html", "dashboard", "Dashboard BI - Unggah Dataset",
                      upload_error=request.args.get("error"))
    
    rfm, df = S["rfm"], S["df"]
    f, d1, d2, cats, segs = dash_filter()
    base = dict(d1=d1, d2=d2, mn=f"{df.Transaction_Date.min():%Y-%m-%d}", mx=f"{df.Transaction_Date.max():%Y-%m-%d}",
                allcat=sorted(df.Category.unique()), cats=cats, allseg=sorted(rfm.Segmen.unique()), segs=segs)
    if f.empty:
        return render("dash.html", "dashboard", "Dashboard BI", empty=True, **base)
    
    bsz = f.groupby("Transaction_ID").Product_ID.nunique()
    kpi = dict(rev=f.Line_Total.sum(), trx=f.Transaction_ID.nunique(), pel=f.Customer_ID.nunique(), bs=bsz.mean())
    mon = f.groupby("Month").Line_Total.sum()
    topp = f.groupby("Product_Name").Line_Total.sum().nlargest(10).iloc[::-1]
    cat = f.groupby("Category").Line_Total.sum().sort_values(ascending=False)
    f_trend = fig([{"type": "scatter", "mode": "lines+markers", "x": L(mon.index), "y": L(mon.values),
                    "line": {"color": "#1E63C6", "width": 3}}], {"yaxis": {"title": "Rp"}}, 320, False)
    f_cat = fig([{"type": "pie", "labels": L(cat.index), "values": L(cat.values), "hole": 0.45}], {}, 320)
    f_top = fig([{"type": "bar", "orientation": "h", "x": L(topp.values), "y": L(topp.index),
                  "marker": {"color": "#10B981"}}], {"margin": {"l": 130}}, 340, False)
    matrix = pd.crosstab(f.Transaction_ID, f.Product_Name).clip(upper=1).astype(bool)
    t_rules = Markup("<div class='note'>Data terfilter terlalu sedikit untuk Apriori (minimal 100 nota).</div>")
    seg_rows = []
    if len(matrix) >= 100:
        sp, _ = run_apriori(matrix, 0.01, max_len=2)
        tp = top_pairs(make_rules(sp, 0.15, len(matrix)), 10)
        if len(tp):
            t_rules = df_html(fmt_rules(tp), sortable=True)
        for s, g in f.groupby("Segmen"):
            m2 = pd.crosstab(g.Transaction_ID, g.Product_Name).clip(upper=1).astype(bool)
            if len(m2) < 150:
                continue
            sp2, _ = run_apriori(m2, 0.015, max_len=2)
            for _, r in top_pairs(make_rules(sp2, 0.15, len(m2)), 3).iterrows():
                seg_rows.append((s, len(m2), f"{r.Antecedent} -> {r.Consequent}", fmt_dec(r.Confidence, 2), fmt_dec(r.Lift, 2)))
    t_seg = df_html(pd.DataFrame(seg_rows, columns=["Segmen", "Jumlah nota", "Aturan", "Confidence", "Lift"])) if seg_rows else ""
    rf = rfm[rfm.Customer_ID.isin(f.Customer_ID.unique())]
    cnt = rf.Segmen.value_counts()
    f_cnt = fig([{"type": "bar", "x": L(cnt.index), "y": L(cnt.values), "marker": {"color": [seg_color(s) for s in cnt.index]}}],
                {}, 320, False)
    prof = rf.groupby("Segmen").agg(Pelanggan=("Customer_ID", "count"), Recency=("Recency", "mean"),
                                    Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean")).reset_index()
    pf = prof.copy()
    pf["Pelanggan"] = pf.Pelanggan.map(fmt_int)
    pf["Recency"] = pf.Recency.map(lambda v: fmt_dec(v, 1))
    pf["Frequency"] = pf.Frequency.map(lambda v: fmt_dec(v, 2))
    pf["Monetary"] = pf.Monetary.map(fmt_rp)
    sc = [{"type": "scatter", "mode": "markers", "name": s, "x": L(g.Frequency), "y": L(g.Monetary), "text": L(g.Customer_ID),
           "marker": {"size": 6, "color": seg_color(s), "opacity": .7}} for s, g in rf.groupby("Segmen")]
    f_sc = fig(sc, {"xaxis": {"title": "Frequency"}, "yaxis": {"title": "Monetary (Rp)"}}, 380)
    return render("dash.html", "dashboard", "Dashboard BI", empty=False, kpi=kpi, f_trend=f_trend, f_cat=f_cat,
                  f_top=f_top, t_rules=t_rules, f_cnt=f_cnt, t_prof=df_html(pf), f_sc=f_sc, t_seg=t_seg, **base)

# ---------------------------------------------------------------------
# 10. ROUTE: SUB-HALAMAN SISTEM BI
# ---------------------------------------------------------------------
@app.route("/bisnis")
@login_required
def bisnis():
    return render("bisnis.html", "bisnis", "Business Understanding", i=S["info"])

@app.route("/eda")
@login_required
def eda():
    raw, info, m = S["raw"], S["info"], S["masalah"]
    dt = pd.DataFrame({"Kolom": raw.columns, "Tipe data": [str(t) for t in raw.dtypes],
                       "Non-null": raw.notna().sum().values, "Missing": raw.isna().sum().values})
    stat = pd.DataFrame([
        ("Jumlah baris x kolom", f"{fmt_int(info['baris'])} x {info['kolom']}"),
        ("Rentang waktu", f"{info['tgl_min']:%d-%m-%Y} s/d {info['tgl_max']:%d-%m-%Y}"),
        ("Jumlah transaksi unik (Transaction_ID)", fmt_int(info["transaksi_unik"])),
        ("Pelanggan unik (tanpa nilai kosong)", fmt_int(info["pelanggan_unik"])),
        ("Produk unik", fmt_int(info["produk_unik"])),
        ("Rata-rata Quantity per baris (mentah)", fmt_dec(info["rata_qty"], 2)),
        ("Median Quantity per baris", fmt_dec(info["median_qty"], 0)),
        ("Total penjualan sementara (Line_Total mentah)", fmt_rp(info["total_sementara"]))], columns=["Statistik", "Nilai"])
    r = raw.copy()
    r["Bulan"] = r.Transaction_Date.dt.to_period("M").astype(str)
    mon = r.groupby("Bulan").Line_Total.sum()
    top = r.groupby("Product_Name").Line_Total.sum().nlargest(10).iloc[::-1]
    qty = r.Quantity.dropna()
    f1 = fig([{"type": "scatter", "mode": "lines+markers", "x": L(mon.index), "y": L(mon.values),
               "line": {"color": "#1E63C6", "width": 3}}], {"yaxis": {"title": "Rp"}}, 340, False)
    f2 = fig([{"type": "bar", "orientation": "h", "x": L(top.values), "y": L(top.index), "marker": {"color": "#10B981"}}],
             {"margin": {"l": 130}}, 340, False)
    f3 = fig([{"type": "histogram", "x": L(qty[(qty >= -2) & (qty <= 10)]), "xbins": {"size": 1}, "marker": {"color": "#F59E0B"}}],
             {"xaxis": {"title": "Quantity", "dtick": 1}, "yaxis": {"title": "Baris"}}, 320, False)
    f4 = fig([{"type": "box", "y": L(qty[qty < 100]), "name": "Quantity (tanpa 999)", "marker": {"color": "#1E63C6"}}], {}, 320, False)
    mm = dict(zip(m.Masalah, m["Jumlah baris"]))
    n_miss = int(sum(v for k, v in mm.items() if k.startswith("Missing")))
    top_nm = top.index[-1]
    ins = [
        f"Penjualan tertinggi pada {mon.idxmax()} ({fmt_rp(mon.max())}) dan terendah pada {mon.idxmin()} ({fmt_rp(mon.min())}) pada data mentah.",
        f"Produk dengan nilai penjualan tertinggi adalah {top_nm} ({fmt_rp(top.iloc[-1])}); {', '.join(top.index[::-1][:3])} berada di tiga teratas.",
        f"Data mentah memuat {fmt_int(mm['Baris duplikat penuh'])} baris duplikat penuh, {fmt_int(n_miss)} nilai kosong dan {fmt_int(mm['Quantity ekstrem (> %d)' % MAX_QTY_WAJAR])} Quantity ekstrem (999).",
        f"Jumlah baris ({fmt_int(info['baris'])}) jauh lebih besar dari jumlah nota ({fmt_int(info['transaksi_unik'])}): satu nota berisi beberapa produk.",
    ]
    return render("eda.html", "eda", "EDA Data Mentah", t_stat=df_html(stat), t_dt=df_html(dt),
                  t_m=df_html(m.assign(**{"Jumlah baris": m["Jumlah baris"].map(fmt_int)})),
                  f1=f1, f2=f2, f3=f3, f4=f4, ins=ins)

@app.route("/quality")
@login_required
def quality():
    log, banding = S["log"], S["banding"]
    b = banding.copy()
    b["Sebelum"] = banding.apply(lambda r: fmt_rp(r.Sebelum) if "Rp" in r.Kondisi else fmt_int(r.Sebelum), axis=1)
    b["Sesudah"] = banding.apply(lambda r: fmt_rp(r.Sesudah) if "Rp" in r.Kondisi else fmt_int(r.Sesudah), axis=1)
    lg = log.copy()
    lg["Jumlah baris"] = lg["Jumlah baris"].map(fmt_int)
    f_b = fig([{"type": "bar", "x": ["Missing value", "Duplikat penuh", "Qty tidak wajar"],
                "y": [banding.Sebelum[3], banding.Sebelum[4], banding.Sebelum[5]], "name": "Sebelum", "marker": {"color": "#EF4444"}},
               {"type": "bar", "x": ["Missing value", "Duplikat penuh", "Qty tidak wajar"],
                "y": [banding.Sesudah[3], banding.Sesudah[4], banding.Sesudah[5]], "name": "Sesudah", "marker": {"color": "#10B981"}}],
              {"barmode": "group"}, 320)
    return render("quality.html", "quality", "Data Quality", t_log=df_html(lg), t_b=df_html(b), f_b=f_b)

@app.route("/data-bersih")
@login_required
def clean():
    df = S["df"]
    show = df.drop(columns=["Month"]).head(500).copy()
    show["Transaction_Date"] = show.Transaction_Date.dt.strftime("%Y-%m-%d %H:%M")
    return render("clean.html", "clean", "Data Bersih", n=len(df), ntx=df.Transaction_ID.nunique(),
                  npel=df.Customer_ID.nunique(), tampil=len(show), t=df_html(show, sortable=True, table_id="tblclean"))

@app.route("/unduh/data-bersih.csv")
@login_required
def dl_clean():
    return Response(S["df"].to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=Data_Clean_P8.csv"})

@app.route("/unduh/rfm.csv")
@login_required
def dl_rfm():
    return Response(S["rfm"].to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=RFM_Segmen_P8.csv"})

def hist(series, color, title):
    return fig([{"type": "histogram", "x": L(series), "nbinsx": 25, "marker": {"color": color}}],
               {"xaxis": {"title": title}, "yaxis": {"title": "Pelanggan"}}, 260, False)

@app.route("/fitur")
@login_required
def fitur():
    df, rfm, basket = S["df"], S["rfm"], S["basket"]
    bs = basket.value_counts().sort_index()
    f_bs = fig([{"type": "bar", "x": L(bs.index), "y": L(bs.values), "marker": {"color": "#8B5CF6"}}],
               {"xaxis": {"title": "basket_size", "dtick": 1}, "yaxis": {"title": "Jumlah nota"}}, 300, False)
    t = rfm[["Customer_ID", "Recency", "Frequency", "Monetary", "Log_Frequency", "Log_Monetary"]].copy()
    t["Monetary"] = t.Monetary.map(fmt_rp)
    t["Log_Frequency"] = t.Log_Frequency.map(lambda v: fmt_dec(v, 3))
    t["Log_Monetary"] = t.Log_Monetary.map(lambda v: fmt_dec(v, 3))
    return render("fitur.html", "fitur", "basket_size & RFM", f_bs=f_bs, bs_mean=basket.mean(), ms=S["matrix"].shape,
                  ref=S["ref_date"], npel=len(rfm), t_rfm=df_html(t.head(400), sortable=True, table_id="tblrfm"),
                  h_r=hist(rfm.Recency, "#1E63C6", "Hari"), h_f=hist(rfm.Frequency, "#10B981", "Nota"),
                  h_m=hist(rfm.Monetary, "#F59E0B", "Rp"), sk=S["skew"], n_item=len(S["raw"]))

def lift_heatmap(matrix):
    X = matrix.values.astype(float)
    n = len(X)
    p = X.mean(0)
    lift = ((X.T @ X) / n) / np.outer(p, p)
    np.fill_diagonal(lift, np.nan)
    return list(matrix.columns), np.round(lift, 2)

def network_fig(rules):
    sel = rules.head(15)
    deg = {}
    edges = []
    for _, r in sel.iterrows():
        edges.append((r.Antecedent, r.Consequent))
        deg[r.Antecedent] = deg.get(r.Antecedent, 0) + 1
        deg[r.Consequent] = deg.get(r.Consequent, 0) + 1
    nodes = sorted(deg, key=lambda x: -deg[x])
    ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (float(np.cos(a)), float(np.sin(a))) for n, a in zip(nodes, ang)}
    ex, ey = [], []
    for a, b in edges:
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    tr = [{"type": "scatter", "mode": "lines", "x": ex, "y": ey, "line": {"color": "#94A3B8", "width": 2}, "hoverinfo": "none"},
          {"type": "scatter", "mode": "markers+text", "x": [pos[n][0] for n in nodes], "y": [pos[n][1] for n in nodes],
           "text": nodes, "textposition": "top center", "marker": {"size": [14 + 5 * deg[n] for n in nodes], "color": "#1E63C6"}}]
    return fig(tr, {"showlegend": False, "xaxis": {"visible": False, "range": [-1.6, 1.6]}, "yaxis": {"visible": False, "range": [-1.4, 1.4]},
                    "margin": {"l": 10, "r": 10, "t": 10, "b": 10}}, 460, False)

@app.route("/apriori")
@login_required
def apriori():
    freq, rules, matrix = S["freq"], S["rules"], S["matrix"]
    top = top_pairs(rules, 10)
    ff = freq.copy()
    ff["Support"] = ff.Support.map(lambda v: fmt_dec(v * 100, 2) + "%")
    prods, hz = lift_heatmap(matrix)
    hm = fig([{"type": "heatmap", "z": hz.tolist(), "x": prods, "y": prods, "colorscale": "YlOrRd", "colorbar": {"title": "Lift"}}],
             {"margin": {"l": 120, "b": 120}}, 460, False)
    ip = [dict(a=r.Antecedent, c=r.Consequent, sup=r.Support, conf=r.Confidence, lift=r.Lift, n=r["Jumlah transaksi"])
          for _, r in top.head(5).iterrows()]
    return render("apriori.html", "apriori", "Apriori", ms=matrix.shape, min_sup=S["min_sup"], min_conf=S["min_conf"],
                  sup_opts=[0.005, 0.0075, 0.01, 0.015, 0.02], conf_opts=[0.1, 0.15, 0.2, 0.3],
                  t_freq=df_html(ff.head(60)), nf=len(freq), no3=not (freq.Ukuran >= 3).any(),
                  t_rules=df_html(fmt_rules(top, ("Antecedent", "Consequent", "Support", "Confidence", "Lift", "Leverage", "Jumlah transaksi")), sortable=True),
                  nr=len(rules), f_net=network_fig(rules) if len(rules) else Markup("<p>Tidak ada aturan.</p>"),
                  f_hm=hm, ip=ip)

@app.route("/set-apriori")
@login_required
def set_apriori():
    global MIN_SUP, MIN_CONF
    try:
        MIN_SUP = min(max(float(request.args.get("sup", MIN_SUP)), 0.002), 0.1)
        MIN_CONF = min(max(float(request.args.get("conf", MIN_CONF)), 0.05), 0.9)
    except ValueError:
        pass
    load_state(S.get("path"))
    return redirect(url_for("apriori"))

@app.route("/elbow")
@login_required
def elbow():
    ev, k, ka = S["ev"], S["k"], S["k_auto"]
    f1 = fig([{"type": "scatter", "mode": "lines+markers", "x": L(ev.k), "y": L(ev.Inertia), "name": "Inertia (WCSS)",
               "line": {"color": "#1E63C6", "width": 3}, "marker": {"size": 9}},
              {"type": "scatter", "mode": "markers+text", "x": [k], "y": [float(ev.set_index("k").Inertia[k])], "name": "k terpilih",
               "text": [f"k = {k}"], "textposition": "top right", "marker": {"size": 17, "color": "#F59E0B", "symbol": "star"}}],
             {"xaxis": {"title": "Jumlah cluster (k)", "dtick": 1}, "yaxis": {"title": "Inertia / WCSS"}}, 360)
    f2 = fig([{"type": "bar", "x": L(ev.k), "y": L(ev.Silhouette), "text": [fmt_dec(v, 3) for v in ev.Silhouette], "textposition": "outside",
               "marker": {"color": ["#F59E0B" if x == k else "#93C5FD" for x in ev.k]}}],
             {"xaxis": {"title": "Jumlah cluster (k)", "dtick": 1}, "yaxis": {"title": "Silhouette", "range": [0, float(ev.Silhouette.max()) * 1.2]}}, 360, False)
    t = ev.copy()
    t["Penurunan Inertia (%)"] = [("-" if i == 0 else fmt_dec((ev.Inertia[i - 1] - ev.Inertia[i]) / ev.Inertia[i - 1] * 100, 1)) for i in range(len(ev))]
    t["Inertia"] = t.Inertia.map(lambda v: fmt_dec(v, 1))
    t["Silhouette"] = t.Silhouette.map(lambda v: fmt_dec(v, 4))
    t["Catatan"] = ["terpilih" if x == k else "" for x in ev.k]
    sil = ev.set_index("k").Silhouette
    why = [f"Silhouette tertinggi mutlak ada pada k = {int(ev.loc[ev.Silhouette.idxmax(), 'k'])} ({fmt_dec(sil.max(), 3)}).",
           f"Di antara rentang k = 3-6 yang dapat diinterpretasi bisnis, k = {ka} memiliki silhouette tertinggi ({fmt_dec(sil[ka], 3)}); kurva Elbow mulai melandai pada kisaran ini.",
           f"k yang dipakai sekarang: k = {k} (silhouette {fmt_dec(sil[k], 3)})."]
    if k != ka:
        why.append(f"k = {k} dipilih manual; pilihan otomatis sistem adalah k = {ka}.")
    return render("elbow.html", "elbow", "Elbow & Silhouette", f_elbow=f1, f_sil=f2, t_ev=df_html(t), why=why)

@app.route("/set-k")
@login_required
def set_k():
    global K_OVERRIDE
    v = request.args.get("k", "auto")
    K_OVERRIDE = int(v) if v.isdigit() and int(v) in K_RANGE else None
    load_state(S.get("path"))
    nxt = request.args.get("next", "kmeans")
    return redirect(url_for(nxt if nxt in ("elbow", "kmeans") else "kmeans"))

@app.route("/kmeans")
@login_required
def kmeans():
    rfm = S["rfm"]
    prof = rfm.groupby("Segmen").agg(Pelanggan=("Customer_ID", "count"), Recency_rata=("Recency", "mean"), Recency_median=("Recency", "median"),
                                      Frequency_rata=("Frequency", "mean"), Monetary_rata=("Monetary", "mean"),
                                      Monetary_median=("Monetary", "median"), Total=("Monetary", "sum")).reset_index()
    prof["Share_Pel"] = prof.Pelanggan / prof.Pelanggan.sum()
    prof["Share_Nilai"] = prof.Total / prof.Total.sum()
    pf = pd.DataFrame({
        "Segmen": prof.Segmen, "Jumlah Pelanggan": prof.Pelanggan.map(fmt_int), "% Pelanggan": prof.Share_Pel.map(fmt_pct),
        "Avg Recency (hari)": prof.Recency_rata.map(lambda v: fmt_dec(v, 1)), "Median Recency": prof.Recency_median.map(lambda v: fmt_dec(v, 1)),
        "Avg Frequency": prof.Frequency_rata.map(lambda v: fmt_dec(v, 2)), "Avg Monetary": prof.Monetary_rata.map(fmt_rp),
        "Median Monetary": prof.Monetary_median.map(fmt_rp), "% Nilai Belanja": prof.Share_Nilai.map(fmt_pct)})
    mk = lambda col: [{"type": "scatter", "mode": "markers", "name": s, "x": L(g[col]), "y": L(g.Monetary), "text": L(g.Customer_ID),
                       "marker": {"size": 6, "color": seg_color(s), "opacity": .7}} for s, g in rfm.groupby("Segmen")]
    f1 = fig(mk("Frequency"), {"xaxis": {"title": "Frequency (nota)"}, "yaxis": {"title": "Monetary (Rp)"}}, 400)
    f2 = fig(mk("Recency"), {"xaxis": {"title": "Recency (hari)"}, "yaxis": {"title": "Monetary (Rp)"}}, 400)
    f_donut = fig([{"type": "pie", "labels": L(prof.Segmen), "values": L(prof.Pelanggan), "hole": 0.5,
                    "marker": {"colors": [seg_color(s) for s in prof.Segmen]}}], {}, 340)
    f_contrib = fig([{"type": "bar", "x": L(prof.Segmen), "y": L(prof.Total), "marker": {"color": [seg_color(s) for s in prof.Segmen]}}],
                    {"yaxis": {"title": "Total Monetary (Rp)"}}, 340, False)
    strat = []
    for _, r in prof.iterrows():
        f, a = strategi_for(r.Segmen)
        strat.append(dict(n=r.Segmen, f=f, a=a, c=r.Pelanggan, share=r.Share_Nilai))
    top = prof.sort_values("Share_Nilai", ascending=False).iloc[0]
    return render("km.html", "kmeans", "K-Means & Segmen", t_prof=df_html(pf), f_sc1=f1, f_sc2=f2, f_donut=f_donut,
                  f_contrib=f_contrib, strat=strat, top1=top.Segmen, top1_share=top.Share_Nilai)

@app.route("/rekomendasi")
@login_required
def rekomendasi():
    df, rfm, rules = S["df"], S["rfm"], S["rules"]
    top = top_pairs(rules, 5)
    if top.empty:
        return "<p style='font-family:Segoe UI;padding:30px'>Tidak ada aturan asosiasi pada parameter saat ini. <a href='/apriori'>Ke Apriori</a></p>"
    t1 = top.iloc[0]
    A, C = t1.Antecedent, t1.Consequent
    nA = df[df.Product_Name == A].Transaction_ID.nunique()
    nAC = int(t1["Jumlah transaksi"])
    hC = df[df.Product_Name == C].Unit_Price.median()
    a1, r1, r2 = 0.10, 0.10, 0.05
    e1 = (nA - nAC) * a1 * hC
    ar = rfm[rfm.Segmen.str.contains("At Risk")]
    ch = rfm[rfm.Segmen.str.contains("Churn")]
    lo = rfm[rfm.Segmen.str.contains("Loyal")]
    nota_ar = (ar.Monetary / ar.Frequency).mean() if len(ar) else 0
    nota_ch = (ch.Monetary / ch.Frequency).mean() if len(ch) else 0
    e2 = len(ar) * r1 * nota_ar * 2
    e3 = len(ch) * r2 * nota_ch
    slo = lo.Monetary.sum() / rfm.Monetary.sum() if len(lo) else 0
    prio = pd.DataFrame([
        ("Reaktivasi segmen At Risk", 5, 5, 4, 4), (f"Bundling {A} + {C}", 4, 4, 5, 5),
        ("Program loyalitas segmen Loyal", 3, 4, 3, 4), ("Win-back segmen Churn", 3, 2, 4, 4)],
        columns=["Rekomendasi", "Urgensi (1-5)", "Potensi manfaat (1-5)", "Biaya rendah (1-5)", "Kemudahan (1-5)"])
    prio["Skor"] = prio.iloc[:, 1:].sum(axis=1)
    prio = prio.sort_values("Skor", ascending=False).reset_index(drop=True)
    prio.insert(0, "Prioritas", range(1, len(prio) + 1))
    return render("rek.html", "rekomendasi", "Rekomendasi", A=A, C=C, t1=t1, nA=nA, nAC=nAC, hC=hC, a1=a1, e1=e1,
                  top=[r for _, r in top.iterrows()], nlo=len(lo), slo=slo, nar=len(ar), nch=len(ch), nota_ar=nota_ar,
                  r1=r1, r2=r2, e2=e2, e3=e3, t_prio=df_html(prio), bs=S["basket"].mean())

@app.route("/unduh")
@login_required
def unduh():
    items = ["Laporan audit kualitas data (sebelum cleaning)", "Dataset hasil preprocessing (Data_Clean)",
             "Dokumentasi perubahan (Log_Preprocessing, Sebelum_Sesudah)", "basket_size per transaksi",
             "Tabel RFM per pelanggan + tanggal acuan", "Matriks transaksi dan hasil Apriori (support, confidence, lift)",
             "Elbow dan Silhouette k = 2 sampai 8", "K-Means, profil dan label segmen", "Dashboard BI (tangkapan layar)",
             "Rekomendasi keputusan (fakta / estimasi / aktual)", "Kode program (tugas 8.py)", "Presentasi ringkas untuk manajemen"]
    return render("dl.html", "unduh", "Unduh", items=items)

@app.route("/unduh/hasil.xlsx")
@login_required
def dl_excel():
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        S["df"].to_excel(w, sheet_name="Data_Clean", index=False)
        S["log"].to_excel(w, sheet_name="Log_Preprocessing", index=False)
        S["banding"].to_excel(w, sheet_name="Sebelum_Sesudah", index=False)
        S["basket"].reset_index().to_excel(w, sheet_name="Basket_Size", index=False)
        S["rfm"].to_excel(w, sheet_name="RFM_Segmen", index=False)
        S["freq"].to_excel(w, sheet_name="Frequent_Itemsets", index=False)
        S["rules"].to_excel(w, sheet_name="Aturan_Apriori", index=False)
        S["ev"].to_excel(w, sheet_name="Evaluasi_K", index=False)
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name="Hasil_P8_UTS_BI.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------------------------------------------
def free_port(start=5000):
    import socket
    for p in range(start, start + 20):
        with socket.socket() as sk:
            if sk.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start

if __name__ == "__main__":
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print("=" * 65)
    print(" SISTEM BI ANALISIS TRANSAKSI TOKO - APRIORI + K-MEANS")
    print(f" Mahasiswa : {IDENTITAS['nama']} (NPM: {IDENTITAS['npm']})")
    print(f" Program   : {IDENTITAS['prodi']} - {IDENTITAS['univ']}")
    print(f" Akses URL : {url}")
    print("=" * 65)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)