---
marp: true
theme: default
paginate: true
header: 'NEMO REAL ESTATE ANALYTICS'
footer: 'SENIOR ANALYST REPORT 2026'
backgroundColor: #F5F500
style: |
  section {
    font-family: 'Arial Black', sans-serif;
    color: #000000;
    padding: 40px;
  }
  h1 {
    font-size: 50pt;
    text-transform: uppercase;
    background-color: #FFFFFF;
    border: 4pt solid #000000;
    padding: 20px;
    box-shadow: 10px 10px 0px #000000;
    display: inline-block;
  }
  h2 {
    font-size: 35pt;
    text-transform: uppercase;
    background-color: #0000FF;
    color: #FFFFFF;
    border: 3pt solid #000000;
    padding: 10px 20px;
    box-shadow: 7px 7px 0px #000000;
    margin-bottom: 30px;
  }
  h3 {
    font-family: 'Courier New', monospace;
    font-size: 20pt;
    text-decoration: underline;
  }
  p, li {
    font-family: 'Courier New', monospace;
    font-size: 16pt;
    font-weight: bold;
  }
  .card {
    background-color: #FFFFFF;
    border: 4pt solid #000000;
    padding: 20px;
    box-shadow: 10px 10px 0px #000000;
    margin: 20px 0;
  }
  .accent-card {
    background-color: #FF2D55;
    color: #FFFFFF;
    border: 4pt solid #000000;
    padding: 20px;
    box-shadow: 10px 10px 0px #000000;
  }
  .big-number {
    font-size: 80pt;
    font-family: 'Arial Black';
    -webkit-text-stroke: 2pt black;
    color: #FFFFFF;
    text-shadow: 8px 8px 0px #000000;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    border: 4pt solid #000000;
    box-shadow: 10px 10px 0px #000000;
    background: white;
  }
  th {
    background-color: #0000FF;
    color: white;
    border: 3pt solid #000000;
    padding: 10px;
  }
  td {
    border: 2pt solid #000000;
    padding: 10px;
    font-family: 'Courier New', monospace;
  }
  img {
    border: 4pt solid #000000;
    box-shadow: 10px 10px 0px #000000;
  }
---

# NEMO EDA
# REPORT

**BY SENIOR ANALYST**
**2026. 04. 29**

---

## 01. DATA CHECK

<div class="card">
<span class="big-number">678</span>
<p>TOTAL PROPERTY SAMPLES ANALYZED</p>
</div>

- **VARIABLES**: 16 COLUMNS
- **QUALITY**: 100% CLEANED
- **DUPLICATES**: ZERO

---

## 02. CORE STATS

| ITEM | MEAN | MEDIAN | MAX |
| :--- | :--- | :--- | :--- |
| **DEPOSIT** | 6,867만 | 4,000만 | 10.8억 |
| **RENT** | 532만 | 335만 | 9,000만 |
| **PREMIUM** | 4,610만 | 0원 | 9억 |

<div class="accent-card">
WARNING: EXTREME POLARIZATION DETECTED. AVERAGE IS A TRAP.
</div>

---

## 03. DEPOSIT

![w:700 center](../images/plot_1.png)

### MAJOR ZONE: 30M - 50M
### MARKET SPLIT: BUDGET VS LUXURY

---

## 04. MONTHLY RENT

![w:700 center](../images/plot_2.png)

### SWEET SPOT: 1M - 3M (SMALL BIZ)
### RISK: OVER 5M REQUIRES BRAND POWER

---

## 05. FLOOR & BIZ

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
<div class="card">
<h3>BIZ TYPE</h3>
<ul>
<li>OTHERS: 327</li>
<li>RESTAURANT: 95</li>
<li>SERVICE: 89</li>
</ul>
</div>
<div class="card" style="background-color: #CCFF00;">
<h3>FLOOR</h3>
<ul>
<li>1F: 211 (DOMINANT)</li>
<li>B1: 124 (RISING)</li>
<li>2F: 108</li>
</ul>
</div>
</div>

---

## 06. CORRELATION

![w:600 center](../images/plot_5.png)

<div class="big-number">0.95</div>
<p>LINEAR TRANSPARENCY: NO FREE LUNCH IN THIS MARKET.</p>

---

## 07. AREA VS RENT

![w:600 center](../images/plot_6.png)

### SPACE PREMIUM: SMALLER = HIGHER P/㎡
### STRATEGY: MAXIMIZE DENSITY

---

## 08. TOP RENT BIZ

![w:700 center](../images/plot_7.png)

### HEAVYWEIGHTS: PUBS & LARGE DINING
### CHECK YOUR LIMIT BEFORE ENTRY

---

## 09. MULTIVARIATE

![w:600 center](../images/plot_8.png)

### FIND THE OUTLIERS
### OPPORTUNITY LIES IN THE EDGES

---

## 10. KEYWORDS

![w:700 center](../images/plot_9.png)

<div class="card">
<h3>#NO_PREMIUM #STATION_AREA</h3>
<p>MARKET DESIRE REFLECTED. VALIDATE THE RISK BEHIND "FREE".</p>
</div>

---

## 11. FLOOR PREMIUM

![w:700 center](../images/plot_10.png)

### 1F: ASSET VALUE
### OTHERS: SUNK COST RISK

---

## 12. ENGAGEMENT

![w:700 center](../images/plot_11.png)

### FOCUS ON "WISHLIST DENSITY"
### NOT JUST VIEWS.

---

## 13. FINAL ADVICE

<div class="accent-card">
1. ESCAPE THE AVERAGE
</div>
<div class="card">
2. CALCULATE RE-INVESTMENT
</div>
<div class="accent-card" style="background-color: #0000FF;">
3. DIGITAL VISIBILITY IS KING
</div>

---

# Q&A
# THANK YOU.

**NEMO PROPERTY ANALYTICS**
