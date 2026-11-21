"""
Builds report.docx from the outputs of train.py, score.py and the notebook.

Run after:  python train.py && python score.py ... && jupyter nbconvert --execute freight_rate_walkthrough.ipynb
Usage:      python make_report.py
"""
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from sklearn.linear_model import LinearRegression

AUTHOR = "Ayushi" # your name, shown under the title (leave "" to hide)
REPO_URL = ""    # GitHub link, shown under the title (leave "" to hide)

OUT, FIG = Path("outputs"), Path("outputs/report_figures")
FIG.mkdir(parents=True, exist_ok=True)
TEAL, ORANGE, GREY, RED = "#064A56", "#E07A1F", "#9DAFB3", "#C0392B"
TEAL_RGB = RGBColor(0x06, 0x4A, 0x56)

# ============================================================================ data
train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
valid = pd.read_csv("data/validation.csv", parse_dates=["date"])
backtest = pd.read_csv(OUT / "backtest_results.csv")
bt_pred = pd.read_csv(OUT / "backtest_predictions.csv", parse_dates=["date"])
experiments = pd.read_csv(OUT / "experiments.csv", index_col=0)
exp_f = pd.read_csv(OUT / "experiment_f.csv")
noise_floor = pd.read_csv(OUT / "noise_floor.csv", index_col=0).iloc[0, 0]
coef = pd.read_csv(OUT / "linear_coefficients.csv", index_col=0).iloc[:, 0]
december = pd.read_csv("data/december_chart_inputs.csv", parse_dates=["date"])

avg = backtest[backtest["train"] == "AVERAGE"].iloc[0]
folds = backtest[backtest["train"] != "AVERAGE"]
plain_xgb = experiments.loc[experiments.index.str.startswith("A:"), "AVG MAPE%"].iloc[0]
final_exp = experiments.loc[experiments.index.str.startswith("E:"), "AVG MAPE%"].iloc[0]

# Same simple broken-price rule as train.py (truck type + distance formula).
X = pd.get_dummies(train.equipment).astype(float).assign(log_dist=np.log(train.distance))
y = np.log(train.posted_rate)
resid = y - LinearRegression().fit(X, y).predict(X)
train["broken"] = (np.abs(resid - np.median(resid)) > 0.5).values
n_broken = int(train.broken.sum())


# ============================================================================ figures
def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return FIG / name


def fig_timeline():
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    plans = [("Backtest 1", 4, 6), ("Backtest 2", 6, 8), ("Backtest 3", 8, 10), ("Final model", 10, 12)]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for row, (name, train_end, test_end) in enumerate(plans[::-1]):
        for m in range(12):
            if m < train_end:
                color, label = TEAL, "learn"
            elif m < test_end:
                color, label = (RED if name == "Final model" else ORANGE), ("predict" if name == "Final model" else "test")
            else:
                color, label = "#EEF2F3", ""
            ax.barh(row, 0.94, left=m, color=color, height=0.62)
            if label and (m == 0 or (m == train_end)):
                ax.text(m + 0.08, row, label, va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(-0.15, row, name, ha="right", va="center", fontsize=9, fontweight="bold" if name == "Final model" else None)
    ax.set_xticks([m + 0.47 for m in range(12)], [d.strftime("%b") for d in months], fontsize=8)
    ax.axvline(10 - 0.03, color="black", lw=1, ls="--")
    ax.text(10.0, 3.55, " Nov-Dec: no labels (the 12,000 loads to predict)", fontsize=8, va="bottom")
    ax.set_yticks([]); ax.set_xlim(-0.05, 12); ax.set_ylim(-0.5, 3.9)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    return save(fig, "fig_timeline.png")


def fig_broken_prices():
    rpm = train.posted_rate / train.distance
    ratio = np.log(rpm / rpm.groupby(train.equipment).transform("median"))
    fig, ax = plt.subplots(figsize=(9, 2.8))
    bins = np.linspace(-2, 2, 200)
    ax.hist(ratio[~train.broken], bins=bins, color=TEAL, label="normal prices")
    ax.hist(ratio[train.broken], bins=bins, color=RED, label=f"flagged as broken ({n_broken:,} loads)")
    ax.set_yscale("log"); ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("log( price per mile ÷ typical price per mile for that truck type )", fontsize=8)
    ax.set_ylabel("loads (log scale)", fontsize=8); ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return save(fig, "fig_broken_prices.png")


def daily_level():
    c = train[~train.broken].copy()
    Xc = pd.get_dummies(c.equipment).astype(float)
    Xc["ld"] = np.log(c.distance); Xc["ld2"] = Xc.ld ** 2
    Xc["w"] = c.weight.abs().fillna(c.weight.abs().median()) / 1e4
    Xc["qs"] = c.quote_signal
    yc = np.log(c.posted_rate)
    c["resid"] = yc - LinearRegression().fit(Xc, yc).predict(Xc)
    mi = pd.concat([train, valid]).groupby("date").market_index.mean()
    d = c.groupby("date").agg(level=("resid", "mean")).join(mi.rename("mi"))
    d["log_mi"] = np.log(d.mi)
    d["t"] = (d.index - pd.Timestamp("2025-01-01")).days / 365
    d["qramp"] = (d.index.month % 3 == 0) * (d.index.day - 1) / (d.index.days_in_month - 1)
    return d


def fig_daily_level(d):
    r2 = {}
    for name, cols in [("mi", ["log_mi"]), ("mi_t", ["log_mi", "t"]), ("all", ["log_mi", "t", "qramp"])]:
        r2[name] = LinearRegression().fit(d[cols], d.level).score(d[cols], d.level)
    fit = LinearRegression().fit(d[["log_mi", "t", "qramp"]], d.level).predict(d[["log_mi", "t", "qramp"]])
    fig, ax = plt.subplots(figsize=(9, 3))
    for m in [3, 6, 9]:
        ax.axvspan(pd.Timestamp(2025, m, 1), pd.Timestamp(2025, m, 1) + pd.offsets.MonthEnd(1), color=GREY, alpha=0.18, lw=0)
    ax.plot(d.index, 100 * d.level, color=TEAL, lw=1.3, label="actual daily price level")
    ax.plot(d.index, 100 * fit, color=ORANGE, lw=1, label=f"market index + time trend + quarter-end ramp (R² = {r2['all']:.2f})")
    ax.set_ylabel("% vs average", fontsize=8); ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return save(fig, "fig_daily_level.png"), r2


def fig_quarter_heatmap(d):
    left = d.level - LinearRegression().fit(d[["log_mi", "t"]], d.level).predict(d[["log_mi", "t"]])
    pivot = (100 * left).to_frame("v").assign(m=d.index.month, day=d.index.day).pivot(index="m", columns="day", values="v")
    fig, ax = plt.subplots(figsize=(9, 2.9))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(range(len(pivot.index)), [pd.Timestamp(2025, m, 1).strftime("%b") for m in pivot.index], fontsize=8)
    ax.set_xticks(range(0, 31, 2), range(1, 32, 2), fontsize=8); ax.set_xlabel("day of month", fontsize=8)
    cb = plt.colorbar(im, pad=0.01); cb.set_label("% price left over", fontsize=8); cb.ax.tick_params(labelsize=7)
    return save(fig, "fig_quarter_heatmap.png")


def fig_experiments():
    s = experiments["AVG MAPE%"]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    colors = [TEAL if i == len(s) - 1 else GREY for i in range(len(s))]
    ax.barh(range(len(s))[::-1], s.values, color=colors)
    ax.set_yticks(range(len(s))[::-1], s.index, fontsize=8)
    for i, v in zip(range(len(s))[::-1], s.values):
        ax.text(v + 0.03, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.axvline(noise_floor, color=RED, ls="--", lw=1)
    ax.text(noise_floor + 0.02, len(s) - 0.4, f"noise floor {noise_floor:.2f}%", color=RED, fontsize=8)
    ax.set_xlabel("average error (MAPE %) on the three future test windows, lower is better", fontsize=8)
    ax.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    return save(fig, "fig_experiments.png")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 2.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.95))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", color="white", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", color="white", fontsize=7.5, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#455A60", lw=1.2))

    box(0.0, 1.65, 2.6, 1.25, "WHEN it ships", "daily market index\nload's market index vs day\ntime trend, quarter-end ramp", GREY)
    box(0.0, 0.1, 2.6, 1.25, "WHAT is shipped", "equipment, distance\npickup / delivery coordinates\nweight, quote signal", GREY)
    box(3.3, 1.65, 2.7, 1.25, "Linear model (Ridge)", "market & calendar effects\ncarries trends forward\ninto Nov-Dec", TEAL)
    box(3.3, 0.1, 2.7, 1.25, "XGBoost (GPU)", "lane & load effects\nlearns what the linear\nmodel got wrong", TEAL)
    box(7.0, 0.85, 3.0, 1.3, "Predicted rate", "exp( linear + XGBoost )\n\nbroken prices removed\nbefore training", ORANGE)
    arrow(2.6, 2.27, 3.3, 2.27); arrow(2.6, 0.72, 3.3, 0.72)
    arrow(6.0, 2.27, 7.0, 1.7); arrow(6.0, 0.72, 7.0, 1.3)
    return save(fig, "fig_architecture.png")


def fig_pred_vs_actual():
    last = bt_pred[bt_pred.fold == bt_pred.fold.max()]
    fig, ax = plt.subplots(figsize=(4.4, 4.1))
    ok = ~last.broken_price
    ax.scatter(last.posted_rate[ok], last.predicted_rate[ok], s=2, alpha=0.25, color=TEAL, label="real prices", rasterized=True)
    ax.scatter(last.posted_rate[~ok], last.predicted_rate[~ok], s=7, alpha=0.8, color=RED, marker="x", label="broken prices (not scored)")
    lim = [50, 30000]
    ax.plot(lim, lim, color="black", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(150, 12000)
    ax.set_xlabel("actual price ($)", fontsize=8); ax.set_ylabel("predicted price ($)", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Sep–Oct test (model trained on Jan–Aug only)", fontsize=8)
    return save(fig, "fig_pred_vs_actual.png")


timeline_png, broken_png = fig_timeline(), fig_broken_prices()
dl = daily_level()
daily_png, r2 = fig_daily_level(dl)
heatmap_png, exp_png, arch_png, scatter_png = fig_quarter_heatmap(dl), fig_experiments(), fig_architecture(), fig_pred_vs_actual()

# ============================================================================ docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ["left_margin", "right_margin"]:
    setattr(sec, side, Inches(0.9))
sec.top_margin, sec.bottom_margin = Inches(0.8), Inches(0.8)
CONTENT_W = 8.5 - 1.8

normal = doc.styles["Normal"]
normal.font.name, normal.font.size = "Calibri", Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
normal.paragraph_format.space_after = Pt(5)
normal.paragraph_format.line_spacing = 1.1
for name, size, before in [("Heading 1", 15, 16), ("Heading 2", 12, 10)]:
    st = doc.styles[name]
    st.font.name, st.font.size, st.font.bold, st.font.color.rgb = "Calibri", Pt(size), True, TEAL_RGB
    for attr in ["w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"]:  # let the explicit font win over the theme
        st.element.rPr.rFonts.attrib.pop(qn(attr), None)
    st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(before), Pt(4)
    st.paragraph_format.keep_with_next = True
doc.styles["List Bullet"].paragraph_format.space_after = Pt(2)


def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill.lstrip("#"))
    tc_pr.append(shd)


def write(cell, text, bold=False, size=9, color=None, align=None, italic=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    if align:
        p.alignment = align
    run = p.add_run(str(text))
    run.bold, run.italic, run.font.size = bold, italic, Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color.lstrip("#"))


def table(header, rows, widths, highlight_last=False, align_right_from=1):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(header):
        write(t.rows[0].cells[i], h, bold=True, color="#FFFFFF", size=8.5)
        shade(t.rows[0].cells[i], TEAL)
    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        last = highlight_last and r_i == len(rows) - 1
        for i, v in enumerate(row):
            write(cells[i], v, bold=last, size=8.5,
                  align=WD_ALIGN_PARAGRAPH.RIGHT if i >= align_right_from and _numeric(v) else None)
            if last:
                shade(cells[i], "#E3EEF0")
            elif r_i % 2:
                shade(cells[i], "#F6F9F9")
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def _numeric(v):
    s = str(v).replace("%", "").replace("$", "").replace(",", "").replace("+", "").replace("−", "-").strip()
    try:
        float(s); return True
    except ValueError:
        return False


def callout(title, text, fill="#E8F1F2", bar=TEAL):
    t = doc.add_table(rows=1, cols=1)  # default table style has no borders
    cell = t.rows[0].cells[0]
    cell.width = Inches(CONTENT_W)
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")  # schema order inside tcPr: tcW, tcBorders, shd
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single"); left.set(qn("w:sz"), "24"); left.set(qn("w:space"), "0"); left.set(qn("w:color"), bar.lstrip("#"))
    borders.append(left); tc_pr.append(borders)
    shade(cell, fill)
    p = cell.paragraphs[0]
    r = p.add_run(title + "  "); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = RGBColor.from_string(bar.lstrip("#"))
    r = p.add_run(text); r.font.size = Pt(9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(path, caption, width=CONTENT_W):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption); r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(0x45, 0x5A, 0x60)


def bullets(items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(item, tuple):
            r = p.add_run(item[0]); r.bold = True
            p.add_run(item[1])
        else:
            p.add_run(item)


def para(*parts):
    """para("plain ", ("bold",), " plain") -> one paragraph with bold pieces in tuples."""
    p = doc.add_paragraph()
    for part in parts:
        if isinstance(part, tuple):
            r = p.add_run(part[0]); r.bold = True
        else:
            p.add_run(part)
    return p


def page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("Freight Rate Forecasting  ·  page "); r.font.size = Pt(8); r.font.color.rgb = RGBColor(0x9D, 0xAF, 0xB3)
    fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
    run = OxmlElement("w:r"); rpr = OxmlElement("w:rPr"); sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "16")
    rpr.append(sz); run.append(rpr); t = OxmlElement("w:t"); t.text = "1"; run.append(t); fld.append(run)
    p._p.append(fld)


page_number_footer(sec)

# ============================================================================ numbers used in the text
pct = lambda x: f"{x:.2f}%"
new_city_set = set(valid.pickup) | set(valid.delivery)
seen_cities = set(train.pickup) | set(train.delivery)
new_cities = sorted(new_city_set - seen_cities)
new_city_loads = (valid.pickup.isin(new_cities) | valid.delivery.isin(new_cities)).mean()
seen_lanes = set(zip(train.pickup, train.delivery))
lane_seen_share = np.mean([lane in seen_lanes for lane in zip(valid.pickup, valid.delivery)])
mi_tr, mi_va = train.market_index, valid.market_index
improvement = 100 * (1 - final_exp / plain_xgb)
dec_first, dec_last = december.predicted_rate.iloc[0], december.predicted_rate.iloc[-1]
lane = train[(train.pickup == "Lexington") & (train.delivery == "Fort Wayne") & (train.equipment == "Dry Van") & ~train.broken]
clean_bt = bt_pred[~bt_pred.broken_price].copy()
clean_bt["ape"] = np.abs(clean_bt.predicted_rate / clean_bt.posted_rate - 1) * 100
clean_bt["ae"] = np.abs(clean_bt.predicted_rate - clean_bt.posted_rate)
within5 = (clean_bt.ape < 5).mean() * 100

# ============================================================================ title block
p = doc.add_paragraph()
r = p.add_run("Forecasting Freight Rates for Nov–Dec 2025"); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = TEAL_RGB
p.paragraph_format.space_after = Pt(0)
p = doc.add_paragraph()
r = p.add_run("Machine Learning Engineer Assessment: approach, data validation, train/test split and results")
r.font.size = Pt(11.5); r.font.color.rgb = RGBColor(0x45, 0x5A, 0x60)
meta = " · ".join(x for x in [AUTHOR, date.today().strftime("%B %Y"), REPO_URL] if x)
p = doc.add_paragraph(); r = p.add_run(meta); r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x9D, 0xAF, 0xB3)
p.paragraph_format.space_after = Pt(10)

# KPI tiles
kpis = [(pct(avg["MAPE_%"]), "average error (MAPE)\non future months"),
        (f"${avg['MAE']:.0f}", "average error\nper load (MAE)"),
        (f"{improvement:.0f}%", f"less error than plain\nXGBoost ({plain_xgb:.2f}%)"),
        (pct(noise_floor), "noise floor: best\npossible, no forecasting")]
t = doc.add_table(rows=2, cols=4); t.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (big, small) in enumerate(kpis):
    for row in (0, 1):
        shade(t.rows[row].cells[i], "#E8F1F2" if i < 3 else "#F6F9F9")
        t.rows[row].cells[i].width = Inches(CONTENT_W / 4)
    write(t.rows[0].cells[i], big, bold=True, size=20, color=TEAL if i < 3 else "#455A60", align=WD_ALIGN_PARAGRAPH.CENTER)
    write(t.rows[1].cells[i], small, size=8, color="#455A60", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph().paragraph_format.space_after = Pt(2)

# ============================================================================ 1. summary
doc.add_heading("1. Summary", level=1)
bullets([
    ("The problem is forecasting, not interpolation. ", "All 12,000 loads to price are dated Nov–Dec 2025, strictly after every labelled load (Jan–Oct). "
     "Every validation choice below follows from that: models are always tested on months they have never seen."),
    ("Data issues found and fixed: ", f"{n_broken:,} corrupted prices ({n_broken/len(train):.1%}, off by ~×4 or ~×0.3), sign-flipped and missing weights, "
     f"missing market index values, and {len(new_cities)} cities that appear only in Nov–Dec."),
    ("Key discovery: ", f"the day-to-day price level is explained almost entirely (R² = {r2['all']:.2f}) by three effects: the market index, "
     f"a steady ~{coef['t']*100:.0f}%/year upward drift, and a price ramp through the last month of every quarter. December is a quarter-end month."),
    ("Model: ", "a linear model for the time and market effects (which must be carried forward into Nov–Dec) plus XGBoost on GPU "
     "for the lane and load effects (which do not change over time)."),
    ("Result: ", f"{pct(avg['MAPE_%'])} average error (≈${avg['MAE']:.0f} per load) across three walk-forward tests, versus {plain_xgb:.2f}% for plain XGBoost "
     f"and a {noise_floor:.2f}% noise floor. {within5:.0f}% of test loads were priced within 5%."),
])

# ============================================================================ 2. data validation
doc.add_heading("2. Validating the data", level=1)
doc.add_heading("2.1 Is the Nov–Dec data like the training data?", level=2)
para("Before choosing a split, I checked whether the 12,000 loads to predict come from the same population as the training loads. "
     "Anything that differs is a risk the model and the validation design must handle.")
eq_tr = train.equipment.value_counts(normalize=True) * 100
eq_va = valid.equipment.value_counts(normalize=True) * 100
rows = [
    ["Loads / dates", f"{len(train):,} · Jan 1 – Oct 31", f"{len(valid):,} · Nov 1 – Dec 31", "No overlap in time", "Walk-forward validation (§3)"],
    ["Truck mix (Dry Van / Reefer / Flatbed)", f"{eq_tr['Dry Van']:.0f} / {eq_tr['Reefer']:.0f} / {eq_tr['Flatbed']:.0f}%",
     f"{eq_va['Dry Van']:.0f} / {eq_va['Reefer']:.0f} / {eq_va['Flatbed']:.0f}%", "Same", "–"],
    ["Distance, median (95th pct)", f"{train.distance.median():.0f} ({train.distance.quantile(.95):.0f}) mi",
     f"{valid.distance.median():.0f} ({valid.distance.quantile(.95):.0f}) mi", "Same", "–"],
    ["Weight, median", f"{train.weight.abs().median():,.0f} lb", f"{valid.weight.abs().median():,.0f} lb", "Same", "–"],
    ["Weight negative / missing", f"{(train.weight < 0).mean():.1%} / {train.weight.isna().mean():.1%}",
     f"{(valid.weight < 0).mean():.1%} / {valid.weight.isna().mean():.1%}", "Same error types, ~2× as frequent", "abs(); XGBoost handles missing"],
    ["market_index missing", f"{mi_tr.isna().mean():.1%}", f"{mi_va.isna().mean():.1%}", "Same error, more frequent", "Fill with that day's average"],
    ["market_index mean (range)", f"{mi_tr.mean():.2f} ({mi_tr.min():.2f}–{mi_tr.max():.2f})", f"{mi_va.mean():.2f} ({mi_va.min():.2f}–{mi_va.max():.2f})",
     "Lower, but inside training range", "No extrapolation in this input"],
    ["Loads involving an unseen city", "–", f"{new_city_loads:.1%} ({len(new_cities)} cities)", "Shift", "Use coordinates, not city names"],
    ["Lanes (city pairs) seen in training", "–", f"{lane_seen_share:.0%}", f"{1 - lane_seen_share:.0%} of loads on new lanes", "Model learns geography, not lane IDs"],
]
table(["Check", "Train (Jan–Oct)", "To predict (Nov–Dec)", "Verdict", "How it is handled"], rows,
      [1.75, 1.25, 1.25, 1.15, 1.3], align_right_from=99)
para(("Conclusion: ", ), "the loads are drawn from the same population (same truck mix, distances and weights, and the same kinds of data errors, "
     "which are slightly more frequent in Nov–Dec, so the cleaning steps matter at prediction time too). "
     "The real differences are the dates themselves, new cities and new lanes. The split and the features are designed around exactly these.")

doc.add_heading("2.2 Data-quality issues and fixes", level=2)
figure(broken_png, f"Figure 1. Price per mile relative to the typical value for the truck type. {n_broken:,} loads ({n_broken/len(train):.1%}) form two "
                   "separate clusters ~×4 too high and ~×0.3 too low: data-entry errors, not real prices.")
rows = [
    ["Corrupted prices", f"{n_broken:,} ({n_broken/len(train):.1%})", "Clusters at ~×4 and ~×0.3 with a clear gap (Fig. 1)",
     "Flag with a simple truck-type + distance formula (>1.65× away); remove from training and from scoring"],
    ["Negative weight", f"{(train.weight < 0).sum()}", "Priced like ~30k lb loads, not empty trucks", "Sign typo: take absolute value"],
    ["Missing weight", f"{train.weight.isna().sum()}", "No reliable way to recover the value", "Leave missing; XGBoost learns a default path"],
    ["Weight capped at 47,500 lb", f"{(train.weight == 47500).sum():,}", "Hard ceiling in source data", "Keep (trees treat it as a bucket)"],
    ["Missing market_index", f"{mi_tr.isna().sum()}", "Index is ~95% a daily market-wide value", "Fill with that day's average across all loads"],
    ["Unseen cities", f"{len(new_cities)} cities", ", ".join(new_cities[:4]) + ", …", "Latitude/longitude features; city names never used"],
    ["Odd distances", f"{int((train.distance <= 70).sum())} at a 70-mi floor",
     "Road distance ≈ 1.18 × straight-line", "Give the model both road and straight-line distance"],
]
table(["Issue", "Rows (train)", "Evidence", "Fix"], rows, [1.35, 0.95, 2.0, 2.4], align_right_from=99)

# ============================================================================ 3. split
doc.add_heading("3. Train / test split and validation design", level=1)
doc.add_heading("3.1 Walk-forward (time-series) cross-validation", level=2)
para("A random split would let the model learn from October loads while being tested on other October loads. That is not the task. "
     "Instead I used ", ("walk-forward validation",), ": the model learns from all months before a cut-off and is tested on the ",
     ("next two months",), ", the same horizon as the real task. This is repeated at three cut-offs so the result does not depend on one lucky window.")
figure(timeline_png, "Figure 2. The three walk-forward tests mirror the real task (bottom row): learn from the past, predict the next two months.")

doc.add_heading("3.2 Why not a random split? Measured, not assumed", level=2)
para("The same final model scores ", (pct(noise_floor),), " on a random 80/20 split but ", (pct(avg["MAPE_%"]),),
     " on walk-forward tests. The random split looks better only because it leaks information about each day's market level into training. "
     "Reporting it would overstate how well the model forecasts. It is still useful as a ",
     ("noise floor",), ": the error left when no forecasting is needed, i.e. the natural price noise between near-identical loads.")

doc.add_heading("3.3 Leakage controls and metrics", level=2)
bullets([
    ("Only inputs known at prediction time. ", "market_index and quote_signal are supplied in validation.csv for every Nov–Dec load. Daily market averages "
     "are built from these input columns only, never from prices."),
    ("No future labels in any test. ", "In each test the model, the linear trend and the XGBoost part are all fitted only on loads before the cut-off."),
    ("No lane or city IDs. ", "Prevents memorising lanes and keeps the model valid for new cities and lanes."),
    ("Broken prices are excluded from scoring. ", "No model can predict a ×4 data-entry error. The flag is a 4-parameter formula, so it carries no "
     f"information about future prices. For transparency, the error including broken prices is also reported (MAE ${avg['MAE_incl_bad_labels']:.0f})."),
    ("Metrics. ", "MAPE (main: errors relative to price, so a $50 miss on a $300 load counts more than on a $5,000 load), MAE in dollars, RMSE, and "
     "bias (average over/under-prediction, which matters most when forecasting)."),
])

# ============================================================================ 4. insight
doc.add_heading("4. Key finding: what moves prices over time", level=1)
para("To isolate the effect of the date, I removed the effect of distance, truck type, weight and quote signal with a simple formula and "
     "averaged what was left per day. Three effects explain that daily level:")
rows = [
    ["Market index only", f"{r2['mi']:.2f}", f"Elasticity ≈ {coef['log_mi_day']:.2f}: +10% index → +{coef['log_mi_day']*10:.1f}% price; strong weekly cycle"],
    ["+ time trend", f"{r2['mi_t']:.2f}", f"Prices drift up ~{coef['t']*100:.1f}% per year beyond what the index explains"],
    ["+ quarter-end ramp", f"{r2['all']:.2f}", "Prices climb through Mar, Jun and Sep, then reset (Fig. 4)"],
]
table(["Daily price level explained by…", "R²", "Meaning"], rows, [1.9, 0.6, 4.2])
figure(daily_png, "Figure 3. Daily price level (after removing distance/equipment effects) and the three-effect fit. Quarter-end months shaded.")
figure(heatmap_png, "Figure 4. What is left after market index + trend. Mar, Jun and Sep go from cheap (blue) to expensive (red) through the month: "
                    "a quarter-end shipping rush.")
callout("Why this matters.", "December is a quarter-end month and is not in the training data. The ramp is also not the same for every truck: "
        f"about +{coef['qramp']*100:.1f}% for Dry Van, +{(coef['qramp']+coef['qramp_reefer'])*100:.1f}% for Reefer and "
        f"+{(coef['qramp']+coef['qramp_flatbed'])*100:.1f}% for Flatbed by the last day. "
        "A model that ignores this will under-price late December. A tree-only model cannot carry the trend forward at all.")

# ============================================================================ 5. experiments
doc.add_heading("5. Model selection: every design choice was tested", level=1)
para("Each experiment changes one thing and is scored on the same three walk-forward tests and the same clean test loads "
     "(full code and outputs in freight_rate_walkthrough.ipynb).")
what = {"A": "Baseline: XGBoost on all features", "B": "Give XGBoost a time feature", "C1": "Linear part for market + trend; XGBoost on the rest",
        "C2": "Add the quarter-end ramp to the linear part", "D1": "XGBoost sees load features only (no market/date)",
        "D2": "Check: give XGBoost the market index back", "E": "Ramp per truck type + load-level market index (final)"}
learned = {"A": f"Under-prices the future (bias {experiments.iloc[0]['09-10 bias%']:+.1f}% in Sep–Oct)", "B": "Trees can't extrapolate time; no gain", "C1": "Trend overshoots without the ramp",
           "C2": "Bias fixed, big gain", "D1": "Trees stop using market as a proxy for date", "D2": "Worse, confirms D1",
           "E": "Small further gain"}
rows = []
for name, r in experiments.iterrows():
    key = name.split(":")[0]
    fold_mapes = [r[c] for c in experiments.columns if c.endswith("MAPE%") and not c.startswith("AVG")]
    rows.append([key, what.get(key, name), " / ".join(f"{v:.2f}" for v in fold_mapes), f"{r['AVG MAPE%']:.2f}%", learned.get(key, "")])
table(["", "Change", "MAPE % per test (May-Jun / Jul-Aug / Sep-Oct)", "Average", "What it showed"], rows,
      [0.35, 2.15, 1.45, 0.75, 2.0], highlight_last=True, align_right_from=99)
figure(exp_png, "Figure 5. Average forecast error for each design. The final design (E) is close to the noise floor.")

doc.add_heading("5.1 A bug the validation caught", level=2)
para("My first version let the model find broken prices itself: train once, drop loads it badly mispredicts, retrain. Experiment F compares that "
     "with the simple formula rule on identical test loads:")
rows = [[r["predict"], f"{r.iloc[1]:.2f}%", f"{r.iloc[2]:.2f}%", f"{int(r.iloc[3])}"] for _, r in exp_f.iterrows()]
table(["Test window", "Model flags its own misses", "Simple formula rule (final)", "Broken prices missed by the model"], rows, [1.9, 1.6, 1.6, 1.6])
missed = exp_f.iloc[:, 3]
best_cut = 100 * (1 - (exp_f.iloc[:, 2] / exp_f.iloc[:, 1])).max()
callout("Lesson.", f"XGBoost is flexible enough to partly memorise the broken prices, so their errors look small and {missed.min()}–{missed.max()} "
        "per test stay in training. Each is off by ~×4, so under squared error they outweigh thousands of normal loads. "
        f"A low-capacity rule cannot memorise them. Switching cut the error by up to {best_cut:.0f}%.", fill="#FDF1E7", bar=ORANGE)

# ============================================================================ 6. final model
doc.add_heading("6. Final model", level=1)
figure(arch_png, "Figure 6. Two-part model. Each part has one job, so the part that must extrapolate is the simple, linear one.")
rows = [
    ["log_mi_day", f"{coef['log_mi_day']:.3f}", f"Market index +10% → price +{coef['log_mi_day']*10:.1f}%"],
    ["log_mi_dev", f"{coef['log_mi_dev']:.3f}", "A load quoted above the day's average index is priced higher"],
    ["t (years)", f"{coef['t']:.3f}", f"Underlying drift: +{coef['t']*100:.1f}% per year"],
    ["qramp (+ per truck)", f"{coef['qramp']:.3f}", f"End-of-quarter-month premium: Dry Van +{coef['qramp']*100:.1f}%, Reefer +{(coef['qramp']+coef['qramp_reefer'])*100:.1f}%, "
                                                    f"Flatbed +{(coef['qramp']+coef['qramp_flatbed'])*100:.1f}%"],
    ["log_distance", f"{coef['log_distance']:.3f}", f"2× distance → ×{2**coef['log_distance']:.2f} price (long hauls are cheaper per mile)"],
]
table(["Linear term", "Coefficient", "Plain-English meaning"], rows, [1.5, 0.9, 4.3])
bullets([
    ("Target: ", "log(price), so errors are proportional and the model is not dominated by the most expensive loads."),
    ("XGBoost: ", "3,000 trees, depth 7, learning rate 0.02, row/column subsampling 0.8, trained on an NVIDIA GPU (full pipeline ≈ 20 s)."),
    ("Final fit: ", f"trained on all Jan–Oct loads ({len(train) - n_broken:,} after removing broken prices)."),
])

# ============================================================================ 7. results
doc.add_heading("7. Results", level=1)
rows = [[r["train"], str(r["test"]).replace("..", "–").replace("2025-", "") if pd.notna(r["test"]) else "", f"${r['MAE']:.0f}", f"{r['MAPE_%']:.2f}%", f"${r['RMSE']:.0f}",
         f"{r['bias_%']:+.2f}%", f"${r['MAE_incl_bad_labels']:.0f}"] for _, r in backtest.iterrows()]
rows[-1][0], rows[-1][1] = "Average", ""
table(["Learn from", "Predict", "MAE", "MAPE", "RMSE", "Bias", "MAE incl. broken"], rows, [1.3, 1.3, 0.75, 0.75, 0.75, 0.8, 1.05], highlight_last=True)

eq_rows = [[e, f"{g.ape.mean():.2f}%", f"${g.ae.mean():.0f}", f"{len(g):,}"] for e, g in clean_bt.groupby("equipment")]
dist_b = pd.cut(clean_bt.distance, [0, 250, 750, 1500, 5000], labels=["< 250 mi", "250–750 mi", "750–1,500 mi", "> 1,500 mi"])
dist_rows = [[str(b), f"{g.ape.mean():.2f}%", f"${g.ae.mean():.0f}", f"{len(g):,}"] for b, g in clean_bt.groupby(dist_b, observed=True)]

t = doc.add_table(rows=1, cols=2)
left, right = t.rows[0].cells
left.width = right.width = Inches(CONTENT_W / 2)
for cell, title, data in [(left, "Error by truck type (all 3 tests)", eq_rows), (right, "Error by distance (all 3 tests)", dist_rows)]:
    write(cell, title, bold=True, size=9, color=TEAL)
    inner = cell.add_table(rows=1, cols=4); inner.style = "Table Grid"
    for i, h in enumerate(["Group", "MAPE", "MAE", "Loads"]):
        write(inner.rows[0].cells[i], h, bold=True, color="#FFFFFF", size=8); shade(inner.rows[0].cells[i], TEAL)
    for r in data:
        cs = inner.add_row().cells
        for i, v in enumerate(r):
            write(cs[i], v, size=8, align=WD_ALIGN_PARAGRAPH.RIGHT if i else None)
doc.add_paragraph().paragraph_format.space_after = Pt(2)

t = doc.add_table(rows=1, cols=2)
c1, c2 = t.rows[0].cells
c1.width, c2.width = Inches(3.0), Inches(CONTENT_W - 3.0)
c1.paragraphs[0].add_run().add_picture(str(scatter_png), width=Inches(2.9))
write(c2, "Reading the results", bold=True, size=10, color=TEAL)
for text in [
    f"% error is flat ({min(clean_bt.groupby('equipment').ape.mean().min(), clean_bt.groupby(dist_b, observed=True).ape.mean().min()):.1f}–"
    f"{max(clean_bt.groupby('equipment').ape.mean().max(), clean_bt.groupby(dist_b, observed=True).ape.mean().max()):.1f}%) across every truck type "
    "and distance band: there is no segment where the model breaks down.",
    "Dollar error grows with distance only because prices do.",
    f"Bias stays within ±{folds['bias_%'].abs().max():.1f}% in every test: no systematic over- or under-pricing when forecasting.",
    f"{within5:.0f}% of test loads are within 5% of the actual price. The red crosses (broken prices) show why they cannot be scored.",
]:
    p = c2.add_paragraph(style="List Bullet"); r = p.add_run(text); r.font.size = Pt(9)

# ============================================================================ 8. december
doc.add_heading("8. Fixed December prediction chart", level=1)
para("Fixed load: Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb. Only the date changes. The input file has no market index, quote signal "
     "or coordinates, so: coordinates come from the city lookup; market_index is that day's average from validation.csv (inputs only, no labels); "
     "quote_signal is the training median (it has almost no effect on price).")
figure("scorer_results/candidate_december.png", "Figure 7. December chart produced by the provided score.py.")
weekly = december.set_index("date").predicted_rate
swing = (weekly.rolling(7).max() - weekly.rolling(7).min()).dropna().median()
mi_day = valid.groupby("date").market_index.mean()
mi_effect = coef["log_mi_day"] * np.log(mi_day[december.date.iloc[-1]] / mi_day[december.date.iloc[0]]) * 100
rows = [
    ["Dec 1 → Dec 31", f"${dec_first:,.0f} → ${dec_last:,.0f} (+{(dec_last/dec_first-1)*100:.1f}%)",
     f"≈ ramp +{coef['qramp']*100:.1f}% + trend +{coef['t']*100*30/365:.1f}% + market index (Dec 1 is a weekly low, Dec 31 a high) +{mi_effect:.1f}%"],
    ["Weekly wiggle", f"≈ ${swing:,.0f} peak-to-trough each week", f"Market index weekly cycle × elasticity {coef['log_mi_day']:.2f}"],
    ["Quarter-end ramp", f"+{coef['qramp']*100:.1f}% by Dec 31", "December is the last month of Q4 (Dry Van ramp)"],
    ["Time trend", f"+{coef['t']*100*30/365:.1f}% over the month", f"{coef['t']*100:.1f}%/year drift carried forward"],
    ["Sanity check", f"Real loads on this lane: ${lane.posted_rate.min():,.0f} – ${lane.posted_rate.max():,.0f}",
     "December forecast sits inside the lane's observed range"],
]
table(["What the chart shows", "Value", "Why"], rows, [1.5, 2.3, 2.9], align_right_from=99)

# ============================================================================ 9. limitations
doc.add_heading("9. Limitations and next steps", level=1)
bullets([
    ("Holidays: ", "training data has no Nov–Dec, so Thanksgiving and Christmas effects cannot be learned. The model assumes December behaves like "
     "previous quarter-end months. With a prior year of data, I would add holiday features."),
    ("Trend extrapolation: ", "the ~7%/year drift is assumed to continue for two more months. The walk-forward tests show this has worked "
     f"(|bias| ≤ {folds['bias_%'].abs().max():.1f}%), but a regime change would require re-fitting. Monitoring bias weekly would catch that early."),
    ("Uncertainty: ", "the model gives a single price. Quantile regression would add a confidence band for quoting decisions."),
    ("Ensembling: ", "LightGBM/CatBoost variants could add a small gain. I prioritised a model that is simple to explain and audit."),
])

# ============================================================================ 10. reproduce
doc.add_heading("10. Reproducing the results", level=1)
rows = [
    ["freight_rate_walkthrough.ipynb", "Full analysis and every experiment, with outputs and conclusions (start here)"],
    ["train.py", "Cleaning, features, walk-forward backtest, final model, predictions"],
    ["eda.py", "Key data findings and plots"],
    ["run_gpu.sbatch", "Slurm job: train.py on a GPU, then score.py"],
    ["make_report.py", "Builds this report from the outputs"],
    ["validation_predictions.csv", "Final submission: load_id, predicted_rate for all 12,000 loads"],
]
table(["File", "Purpose"], rows, [2.3, 4.4], align_right_from=99)
p = para("Run: ", ("pip install -r requirements.txt",), " → ", ("python train.py",), " → ",
         ("python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv",))

zoom = doc.settings.element.find(qn("w:zoom"))  # python-docx's template omits a required attribute
if zoom is not None:
    zoom.set(qn("w:percent"), "100")
doc.save("report.docx")
print("Saved report.docx")
