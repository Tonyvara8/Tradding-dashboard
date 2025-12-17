import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import time
import concurrent.futures
import json
import os
import uuid

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Market God v9.4 (ULTIMATE)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 5px; height: 50px; font-weight: bold;}
    .live-badge {background-color: #00ff00; color: black; padding: 5px 10px; border-radius: 5px; animation: pulse 2s infinite;}
    @keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(0,255,0,0.7);} 70% {box-shadow: 0 0 0 10px rgba(0,255,0,0);} 100% {box-shadow: 0 0 0 0 rgba(0,255,0,0);}}
    .pnl-green {color: #00ff00; font-weight: bold; font-size: 1.2em;}
    .pnl-red {color: #ff4b4b; font-weight: bold; font-size: 1.2em;}
    .metric-box {background-color: #222; padding: 10px; border-radius: 5px; border: 1px solid #444;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SAUVEGARDE (PERSISTANCE)
# ==========================================
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
                for item in data:
                    if 'id' not in item:
                        item['id'] = str(uuid.uuid4())
                return data
        except:
            return []
    return []

def save_portfolio(data):
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"Erreur de sauvegarde: {e}")

def delete_trade_callback(trade_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p.get('id') != trade_id]
    save_portfolio(st.session_state.portfolio)
    if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
        st.session_state.analysis_results = [r for r in st.session_state.analysis_results if r.get('id') != trade_id]
    st.success("Position supprimée !")

# ==========================================
# 3. LISTE DES MARCHÉS (VERSION COMPLÈTE)
# ==========================================
MARKETS = {
    "Cryptomonnaies": {
        "BTC-USD": "₿ Bitcoin (BTC)",
        "ETH-USD": "Ξ Ethereum (ETH)",
        "SOL-USD": "◎ Solana (SOL)",
        "BNB-USD": "🟡 Binance Coin (BNB)",
        "XRP-USD": "💧 Ripple (XRP)",
        "DOGE-USD": "🐶 Dogecoin (DOGE)",
        "AVAX-USD": "🔺 Avalanche (AVAX)",
        "ADA-USD": "🔵 Cardano (ADA)",
        "DOT-USD": "🟣 Polkadot (DOT)",
        "LINK-USD": "🔗 Chainlink (LINK)",
        "MATIC-USD": "🔷 Polygon (MATIC)",
        "SHIB-USD": "🐕 Shiba Inu"
    },
    "Tech & Actions US": {
        "NVDA": "🤖 Nvidia",
        "TSLA": "🚗 Tesla",
        "AAPL": "🍎 Apple",
        "MSFT": "💻 Microsoft",
        "AMZN": "📦 Amazon",
        "GOOG": "🔍 Google",
        "AMD": "⚙️ AMD",
        "META": "∞ Meta (Facebook)",
        "NFLX": "🎬 Netflix",
        "INTC": "💾 Intel",
        "COIN": "🪙 Coinbase",
        "PLTR": "🧠 Palantir",
        "MSTR": "💰 MicroStrategy",
        "PYPL": "💳 PayPal",
        "UBER": "🚗 Uber"
    },
    "Indices & Matières": {
        "^GSPC": "🇺🇸 S&P 500",
        "^DJI": "🇺🇸 Dow Jones 30",
        "^IXIC": "🇺🇸 Nasdaq 100",
        "^FCHI": "🇫🇷 CAC 40",
        "^GDAXI": "🇩🇪 DAX (Allemagne)",
        "GC=F": "🥇 Or (Gold)",
        "SI=F": "🥈 Argent (Silver)",
        "CL=F": "🛢️ Pétrole (WTI)",
        "HG=F": "🥉 Cuivre"
    },
    "Forex (Devises)": {
        "EURUSD=X": "💶 EUR/USD",
        "GBPUSD=X": "💷 GBP/USD",
        "USDJPY=X": "💴 USD/JPY",
        "EURGBP=X": "💶 EUR/GBP",
        "AUDUSD=X": "🇦🇺 AUD/USD",
        "USDCAD=X": "🇨🇦 USD/CAD",
        "USDCHF=X": "🇨🇭 USD/CHF"
    }
}

ALL_TICKERS = []
TICKER_TO_NAME = {}
NAME_TO_TICKER = {}

for categorie in MARKETS.values():
    for ticker, nom in categorie.items():
        ALL_TICKERS.append(ticker)
        TICKER_TO_NAME[ticker] = nom
        NAME_TO_TICKER[nom] = ticker

# ==========================================
# 4. MOTEUR INTELLIGENT
# ==========================================
def get_clean_data(ticker, period="2y"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if not df.empty:
            return df[['Close', 'Open', 'High', 'Low', 'Volume']]
        return None
    except:
        return None

def add_indicators(df):
    d = df.copy()
    # RSI
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger
    d['SMA_20'] = d['Close'].rolling(20).mean()
    std = d['Close'].rolling(20).std()
    d['BB_Pct'] = (d['Close'] - (d['SMA_20'] - 2*std)) / (4*std)
    
    # Volatilité
    d['Ret_1d'] = d['Close'].pct_change()
    d['Vol_20d'] = d['Ret_1d'].rolling(20).std()
    return d.dropna()

def run_quantum_analysis(df):
    features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
    horizons = [3, 7, 14]
    
    models = {
        "Random Forest 🌳": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting 🚀": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "Logistic Reg ⚖️": LogisticRegression(random_state=42)
    }
    
    results = []
    best_sim = None
    best_f1 = -1
    
    # Calcul du retour journalier suivant pour simulation réaliste
    df['Next_Day_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    
    for h in horizons:
        temp = df.copy()
        temp['Target'] = (temp['Close'].shift(-h) > temp['Close']).astype(int)
        temp = temp.dropna()
        
        X = temp[features]
        y = temp['Target']
        
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Returns pour le test
        daily_ret_test = temp['Next_Day_Return'].iloc[split:].fillna(0)
        
        # Benchmark (Buy & Hold)
        buy_hold_return = (1 + daily_ret_test).prod() - 1
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Simulation réaliste
                strat_ret = preds * daily_ret_test
                cum_ret = (1 + strat_ret).prod() - 1
                
                last_row = df.iloc[[-1]][features]
                prob = model.predict_proba(last_row)[:, 1][0]
                
                # F1 Score & Accuracy
                f1 = f1_score(y_test, preds)
                acc = accuracy_score(y_test, preds)
                
                res = {
                    "Modèle": name,
                    "Horizon": f"{h}j",
                    "Précision": acc,
                    "F1 Score": f1,
                    "Gain IA": cum_ret,
                    "Marché (Ref)": buy_hold_return,
                    "Prédiction": prob,
                    "Returns": strat_ret
                }
                results.append(res)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_sim = res
            except:
                continue
            
    return pd.DataFrame(results), best_sim

def analyze_portfolio_position(ticker, position_type, entry_price):
    try:
        df = get_clean_data(ticker)
        if df is None: return None
        df = add_indicators(df)
        
        features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        valid = df.dropna()
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(valid[features][:-50], valid['Target'][:-50])
        prob = model.predict_proba(df.iloc[[-1]][features])[:, 1][0]
        
        curr = df['Close'].iloc[-1]
        
        pnl = 0.0
        if "Long" in position_type:
            pnl = (curr - entry_price) / entry_price
        else:
            pnl = (entry_price - curr) / entry_price
        
        conseil, danger, msg = "NEUTRE", False, ""
        
        if "Long" in position_type:
            if prob < 0.35:
                conseil, danger, msg = "🔴 VENDRE", True, f"Chute prévue ({100-prob*100:.0f}%)"
            elif prob > 0.60:
                conseil, msg = "🟢 GARDER", "Hausse continue"
            else:
                conseil, msg = "🟡 ATTENDRE", "Indécis"
        else:
            if prob > 0.65:
                conseil, danger, msg = "🔴 COUPER", True, f"Hausse prévue ({prob*100:.0f}%)"
            elif prob < 0.40:
                conseil, msg = "🟢 GARDER", "Baisse continue"
            else:
                conseil, msg = "🟡 ATTENDRE", "Indécis"
            
        return {
            "Ticker": ticker, "Price": curr, "Entry": entry_price, 
            "PnL": pnl, "Prob": prob, "Conseil": conseil, 
            "Msg": msg, "Danger": danger
        }
    except:
        return None

def quick_analyze(ticker):
    try:
        df = get_clean_data(ticker, "1y")
        if df is None: return None
        df = add_indicators(df)
        
        features = ['RSI','BB_Pct','Ret_1d','Vol_20d']
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        valid = df.dropna()
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5)
        model.fit(valid[features], valid['Target'])
        prob = model.predict_proba(df.iloc[[-1]][features])[:, 1][0]
        
        return {
            "Ticker": ticker, 
            "Nom": TICKER_TO_NAME.get(ticker, ticker), 
            "Price": df['Close'].iloc[-1], 
            "Change": df['Close'].pct_change().iloc[-1], 
            "Prob": prob, 
            "RSI": df['RSI'].iloc[-1]
        }
    except:
        return None

# ==========================================
# 5. INTERFACE
# ==========================================
st.sidebar.header("🎛️ NAVIGATION")
app_mode = st.sidebar.radio("Mode", ["💼 PORTEFEUILLE", "🔍 ANALYSE IA", "📡 LIVE SCANNER"])

# --- PAGE PORTEFEUILLE ---
if app_mode == "💼 PORTEFEUILLE":
    st.title("💼 Mon Portefeuille")
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_portfolio()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    with st.expander("➕ Ajouter Trade"):
        c1, c2 = st.columns(2)
        cat = c1.selectbox("Catégorie", list(MARKETS.keys()))
        nom = c1.selectbox("Actif", list(MARKETS[cat].values()))
        sens = c2.selectbox("Sens", ["ACHAT (Long)", "VENTE (Short)"])
        entry = c2.number_input("Prix Entrée ($)", 0.0)
        
        if st.button("Ajouter"):
            ticker_code = NAME_TO_TICKER[nom]
            st.session_state.portfolio.append({
                "id": str(uuid.uuid4()), 
                "Ticker": ticker_code, 
                "Nom": nom, 
                "Sens": sens, 
                "Entry": entry
            })
            save_portfolio(st.session_state.portfolio)
            st.success("Ajouté !")
            st.rerun()

    if len(st.session_state.portfolio) > 0:
        if st.button("🔄 ACTUALISER"):
            with st.spinner("Audit IA en cours..."):
                res = []
                for p in st.session_state.portfolio:
                    a = analyze_portfolio_position(p['Ticker'], p['Sens'], p['Entry'])
                    if a:
                        a.update(p)
                        res.append(a)
                st.session_state.analysis_results = res
        
        # Affichage
        disp = st.session_state.analysis_results if st.session_state.analysis_results else st.session_state.portfolio
        
        for i in disp:
            has_a = 'PnL' in i
            border = "#ff4b4b" if has_a and i['Danger'] else ("#00ff00" if has_a and "GARDER" in i['Conseil'] else "#444")
            
            pnl_html = "N/A"
            if has_a:
                color_class = 'pnl-green' if i['PnL'] >= 0 else 'pnl-red'
                pnl_html = f"<span class='{color_class}'>{i['PnL']:.2%}</span>"
            
            with st.container():
                c1, c2 = st.columns([6, 1])
                
                c1.markdown(f"""
                <div style="border:2px solid {border}; border-radius:10px; padding:10px; margin-bottom:10px; background:#1e1e1e;">
                    <div style="display:flex; justify-content:space-between;">
                        <h3>{i['Nom']} <small>({i['Sens']})</small></h3>
                        {pnl_html}
                    </div>
                    <hr style="margin:5px 0; border-color:#333;">
                    <div style="display:flex; justify-content:space-between;">
                        <div>Entrée: <b>{i['Entry']}$</b> | Actuel: <b>{i.get('Price','?'):.2f}$</b></div>
                        <div><b>{i.get('Conseil','')}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c2.write("")
                c2.write("")
                c2.button("🗑️", key=f"d_{i['id']}", on_click=delete_trade_callback, args=(i['id'],))
    else:
        st.info("Votre portefeuille est vide.")

# --- PAGE ANALYSE IA ---
elif app_mode == "🔍 ANALYSE IA":
    st.title("🧠 Analyse Deep Learning")
    cat = st.selectbox("Marché", list(MARKETS.keys()))
    nom = st.selectbox("Actif", list(MARKETS[cat].values()))
    
    if st.button(f"LANCER SCAN {nom}", type="primary"):
        with st.spinner("Backtest sur 1 an..."):
            ticker = NAME_TO_TICKER[nom]
            df = get_clean_data(ticker, "5y")
            
            if df is not None:
                df = add_indicators(df)
                curr = df['Close'].iloc[-1]
                
                st.metric("Prix Actuel", f"{curr:.2f} $", f"{df['Ret_1d'].iloc[-1]:.2%}")
                
                res_df, best = run_quantum_analysis(df)
                
                st.subheader("📊 Performance Historique (1 An)")
                st.caption("Comparaison : Gain de l'IA vs Gain si on avait juste acheté et attendu (Marché).")
                
                def style_col(v):
                    if isinstance(v, float):
                        if v > 0.5: return 'color:#00ff00;font-weight:bold'
                        if v < 0: return 'color:#ff4b4b'
                    return ''
                
                # Fix for the format error: removed .format() and relied on raw display or custom formatting if needed
                # To apply percent formatting safely, we use styling only on numeric columns
                
                st.dataframe(
                    res_df[['Modèle','Horizon','F1 Score','Gain IA','Marché (Ref)','Prédiction']].style.format({
                        'F1 Score': "{:.1%}",
                        'Gain IA': "{:.1%}",
                        'Marché (Ref)': "{:.1%}",
                        'Prédiction': "{:.1%}"
                    }).applymap(style_col, subset=['Gain IA', 'Marché (Ref)']),
                    use_container_width=True
                )
                
                if best:
                    st.markdown("---")
                    st.subheader(f"🔮 Simulation Levier ({best['Modèle']})")
                    direction = "HAUSSE 🟩" if best['Prédiction'] > 0.5 else "BAISSE 🟥"
                    st.info(f"Prévision : {direction} ({best['Prédiction']:.1%})")
                    
                    levs = [1, 2, 5, 10, 20]
                    cols = st.columns(len(levs))
                    
                    for i, l in enumerate(levs):
                        with cols[i]:
                            st.markdown(f"<div class='metric-box'><h4 style='text-align:center;'>x{l}</h4>", unsafe_allow_html=True)
                            
                            strat_ret = best['Returns'] * l
                            # Remplacer les NaN par 0 pour éviter les crashs de calcul
                            strat_ret = strat_ret.fillna(0)
                            cap = (1 + strat_ret).cumprod()
                            
                            if len(cap) > 0:
                                gain = cap.iloc[-1] - 1
                                dd = ((cap - cap.cummax()) / cap.cummax()).min()
                                
                                if dd < -0.9:
                                    st.write("💀 LIQUIDÉ")
                                else:
                                    c_g = "green" if gain > 0 else "red"
                                    st.write(f"Gain: :{c_g}[{gain:+.0%}]")
                                    st.write(f"Max DD: :orange[{dd:.0%}]")
                            else:
                                st.write("Données insuffsantes")
                                
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Erreur de données.")

# --- PAGE LIVE SCANNER ---
elif app_mode == "📡 LIVE SCANNER":
    st.title("📡 Live Scanner")
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
        
    c1, c2 = st.columns([1, 5])
    if c1.button("▶️ START"): st.session_state.monitoring = True
    if c1.button("⏹️ STOP"): st.session_state.monitoring = False
    
    if st.session_state.monitoring:
        c2.markdown('<span class="live-badge">ON AIR</span>', unsafe_allow_html=True)
    
    ph = st.empty()
    
    if st.session_state.monitoring:
        with st.spinner("Scan..."):
            res = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
                futures = {exe.submit(quick_analyze, t): t for t in ALL_TICKERS}
                for f in concurrent.futures.as_completed(futures):
                    r = f.result()
                    if r: res.append(r)
            
            if res:
                df = pd.DataFrame(res).sort_values(by='Prob', ascending=False)
                
                def color(v):
                    return 'color:#00ff00;font-weight:bold' if v > 0.6 else ('color:#ff4b4b;font-weight:bold' if v < 0.4 else 'color:gray')
                
                with ph.container():
                    st.dataframe(
                        df[['Nom','Price','Change','RSI','Prob']].style.format({'Price':'{:.2f}','Change':'{:.2%}','Prob':'{:.1%}'}).applymap(color, subset=['Prob']),
                        use_container_width=True,
                        height=600
                    )
                    
                    for _, r in df.iterrows():
                        if r['Prob'] > 0.75:
                            st.toast(f"🚀 {r['Nom']} UP!", icon="✅")
                        elif r['Prob'] < 0.25:
                            st.toast(f"📉 {r['Nom']} DOWN!", icon="🔻")
            
            time.sleep(50)
            st.rerun()