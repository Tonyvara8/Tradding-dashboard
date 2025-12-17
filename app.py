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
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Market God v9 (Quantum)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 5px; height: 50px; font-weight: bold;}
    .live-badge {
        background-color: #00ff00; color: black; padding: 5px 10px; 
        border-radius: 5px; font-weight: bold; animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
    }
    .pnl-green { color: #00ff00; font-weight: bold; font-size: 1.2em; }
    .pnl-red { color: #ff4b4b; font-weight: bold; font-size: 1.2em; }
    .metric-box { background-color: #222; padding: 10px; border-radius: 5px; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GESTION SAUVEGARDE
# ==========================================
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
                for item in data:
                    if 'id' not in item: item['id'] = str(uuid.uuid4())
                return data
        except: return []
    return []

def save_portfolio(data):
    try:
        with open(PORTFOLIO_FILE, "w") as f: json.dump(data, f)
    except: pass

def delete_trade_callback(trade_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p.get('id') != trade_id]
    save_portfolio(st.session_state.portfolio)
    if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
        st.session_state.analysis_results = [r for r in st.session_state.analysis_results if r.get('id') != trade_id]
    st.success("Supprimé !")

# ==========================================
# 3. MARCHÉS
# ==========================================
MARKETS = {
    "Cryptomonnaies": {
        "BTC-USD": "₿ Bitcoin (BTC)", "ETH-USD": "Ξ Ethereum (ETH)", "SOL-USD": "◎ Solana (SOL)",
        "BNB-USD": "🟡 Binance Coin", "XRP-USD": "💧 Ripple (XRP)", "DOGE-USD": "🐶 Dogecoin"
    },
    "Tech & Actions": {
        "NVDA": "🤖 Nvidia", "TSLA": "🚗 Tesla", "AAPL": "🍎 Apple", "MSFT": "💻 Microsoft",
        "AMZN": "📦 Amazon", "GOOG": "🔍 Google", "AMD": "⚙️ AMD", "META": "∞ Meta"
    },
    "Indices & Forex": {
        "^GSPC": "🇺🇸 S&P 500", "^DJI": "🇺🇸 Dow Jones", "^IXIC": "🇺🇸 Nasdaq",
        "GC=F": "🥇 OR (Gold)", "EURUSD=X": "💶 EUR/USD", "USDJPY=X": "💴 USD/JPY"
    }
}
ALL_TICKERS = []
TICKER_TO_NAME = {} 
NAME_TO_TICKER = {}
for c in MARKETS.values():
    for t, n in c.items():
        ALL_TICKERS.append(t); TICKER_TO_NAME[t]=n; NAME_TO_TICKER[n]=t

# ==========================================
# 4. MOTEUR IA
# ==========================================
def get_clean_data(ticker, period="2y"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if df.empty: return None
        return df[['Close', 'Open', 'High', 'Low', 'Volume']]
    except: return None

def add_indicators(df):
    d = df.copy()
    d['RSI'] = 100 - (100 / (1 + d['Close'].diff().clip(lower=0).rolling(14).mean() / -d['Close'].diff().clip(upper=0).rolling(14).mean()))
    d['SMA_20'] = d['Close'].rolling(20).mean()
    d['BB_Pct'] = (d['Close'] - (d['SMA_20'] - 2*d['Close'].rolling(20).std())) / (4*d['Close'].rolling(20).std())
    d['Ret_1d'] = d['Close'].pct_change()
    d['Vol_20d'] = d['Ret_1d'].rolling(20).std()
    return d.dropna()

# --- FONCTION QUANTUM (Le nouveau monstre) ---
def run_quantum_analysis(df):
    """Teste 3 modèles sur 3 horizons et renvoie les stats complètes"""
    features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
    horizons = [3, 7, 14]
    
    classifiers = {
        "Random Forest 🌳": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting 🚀": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "Logistic Reg ⚖️": LogisticRegression(random_state=42)
    }
    
    results_matrix = []
    
    for h in horizons:
        temp = df.copy()
        temp['Target'] = (temp['Close'].shift(-h) > temp['Close']).astype(int)
        temp = temp.dropna()
        X = temp[features]
        y = temp['Target']
        
        # Split Train/Test (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        for name, model in classifiers.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Métriques
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds)
                
                # Prédiction Future
                last_row = df.iloc[[-1]][features]
                prob = model.predict_proba(last_row)[:, 1][0]
                
                results_matrix.append({
                    "Modèle": name,
                    "Horizon": f"{h} Jours",
                    "Précision (Accuracy)": acc,
                    "Score F1 (Fiabilité)": f1,
                    "Prédiction Actuelle": prob
                })
            except: continue
            
    return pd.DataFrame(results_matrix)

def analyze_portfolio_position(ticker, position_type, entry_price):
    try:
        df = get_clean_data(ticker)
        if df is None: return None
        df = add_indicators(df)
        features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        valid = df.dropna()
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(valid[features][:-50], valid['Target'][:-50])
        prob = model.predict_proba(df.iloc[[-1]][features])[:, 1][0]
        
        curr = df['Close'].iloc[-1]
        pnl = (curr - entry_price)/entry_price if "Long" in position_type else (entry_price - curr)/entry_price
        
        conseil = "NEUTRE"
        danger = False
        msg = ""
        if "Long" in position_type:
            if prob < 0.35: conseil, danger, msg = "🔴 VENDRE", True, f"Chute probable ({100-prob*100:.0f}%)"
            elif prob > 0.60: conseil, msg = "🟢 GARDER", "Hausse probable"
            else: conseil, msg = "🟡 ATTENDRE", "Indécis"
        else:
            if prob > 0.65: conseil, danger, msg = "🔴 COUPER", True, f"Hausse probable ({prob*100:.0f}%)"
            elif prob < 0.40: conseil, msg = "🟢 GARDER", "Baisse probable"
            else: conseil, msg = "🟡 ATTENDRE", "Indécis"
            
        return {"Ticker": ticker, "Price": curr, "Entry": entry_price, "PnL": pnl, "Prob_Up": prob, "Conseil": conseil, "Message": msg, "Danger": danger}
    except: return None

def quick_analyze(ticker):
    try:
        df = get_clean_data(ticker, "1y")
        if df is None: return None
        df = add_indicators(df)
        features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        valid = df.dropna()
        model = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=1)
        model.fit(valid[features], valid['Target'])
        prob = model.predict_proba(df.iloc[[-1]][features])[:, 1][0]
        return {"Ticker": ticker, "Nom": TICKER_TO_NAME[ticker], "Price": df['Close'].iloc[-1], "Change": df['Close'].pct_change().iloc[-1], "Prob_Up": prob, "RSI": df['RSI'].iloc[-1]}
    except: return None

# ==========================================
# 5. INTERFACE
# ==========================================
st.sidebar.header("🎛️ NAVIGATION")
app_mode = st.sidebar.radio("Mode", ["💼 GESTION PORTFOLIO", "🔍 ANALYSE PROFONDE", "📡 LIVE MONITOR"])

# --- PORTFOLIO ---
if app_mode == "💼 GESTION PORTFOLIO":
    st.title("💼 Mon Portefeuille")
    if 'portfolio' not in st.session_state: st.session_state.portfolio = load_portfolio()
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

    with st.expander("➕ Ajouter Position"):
        c1, c2 = st.columns(2)
        cat = c1.selectbox("Catégorie", list(MARKETS.keys()))
        nom = c1.selectbox("Actif", list(MARKETS[cat].values()))
        sens = c2.selectbox("Sens", ["ACHAT (Long)", "VENTE (Short)"])
        entry = c2.number_input("Prix Entrée ($)", 0.0)
        if st.button("Ajouter"):
            st.session_state.portfolio.append({"id": str(uuid.uuid4()), "Ticker": NAME_TO_TICKER[nom], "Nom": nom, "Sens": sens, "Entry": entry})
            save_portfolio(st.session_state.portfolio)
            st.success("Ajouté !")
            st.rerun()

    if len(st.session_state.portfolio) > 0:
        if st.button("🔄 ACTUALISER"):
            with st.spinner("Analyse..."):
                res = []
                for p in st.session_state.portfolio:
                    a = analyze_portfolio_position(p['Ticker'], p['Sens'], p['Entry'])
                    if a: 
                        a.update(p)
                        res.append(a)
                st.session_state.analysis_results = res
        
        display = st.session_state.analysis_results if st.session_state.analysis_results else st.session_state.portfolio
        for i in display:
            has_a = 'PnL' in i
            border = "#ff4b4b" if has_a and i['Danger'] else ("#00ff00" if has_a and "GARDER" in i['Conseil'] else "#444")
            pnl_html = f"<span class='{'pnl-green' if has_a and i['PnL']>=0 else 'pnl-red'}'>{i['PnL']:.2%}</span>" if has_a else "N/A"
            
            with st.container():
                c_main, c_del = st.columns([6, 1])
                c_main.markdown(f"""
                <div style="border: 2px solid {border}; border-radius: 10px; padding: 15px; margin-bottom: 10px; background:#1e1e1e;">
                    <div style="display:flex; justify-content:space-between;">
                        <h3 style="margin:0;">{i['Nom']} <small>({i['Sens']})</small></h3>
                        <div style="text-align:right;">{pnl_html}</div>
                    </div>
                    <hr style="margin:5px 0; border-color:#333;">
                    <div style="display:flex; justify-content:space-between;">
                        <div>Entrée: <b>{i['Entry']}$</b> | Actuel: <b>{i.get('Price', '?'):.2f}$</b></div>
                        <div><b>{i.get('Conseil', '')}</b></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                c_del.write(""); c_del.write(""); c_del.button("🗑️", key=f"d_{i['id']}", on_click=delete_trade_callback, args=(i['id'],))
    else: st.info("Vide.")

# --- ANALYSE PROFONDE (QUANTUM) ---
elif app_mode == "🔍 ANALYSE PROFONDE":
    st.title("🧠 Analyse Deep Learning")
    cat = st.selectbox("Marché", list(MARKETS.keys()))
    nom = st.selectbox("Actif", list(MARKETS[cat].values()))
    ticker = NAME_TO_TICKER[nom]
    
    # Bouton principal
    if st.button(f"LANCER L'ANALYSE DE {nom}", type="primary"):
        with st.spinner("Téléchargement des données & Calculs tensoriels..."):
            df = get_clean_data(ticker, period="5y") # 5 ans pour bien backtester
            
            if df is not None:
                df = add_indicators(df)
                curr_price = df['Close'].iloc[-1]
                volatility = df['Vol_20d'].iloc[-1]
                
                # 1. Analyse Technique de base
                st.metric("Prix Actuel", f"{curr_price:.2f} $", f"{df['Ret_1d'].iloc[-1]:.2%}")
                
                # 2. RUN QUANTUM ANALYSIS
                results_df = run_quantum_analysis(df)
                
                # Affichage du Tableau de Performance
                st.subheader("📊 Performance des Modèles (Backtest)")
                
                # Mise en forme conditionnelle
                def highlight_good(val):
                    if isinstance(val, float):
                        if val > 0.60: return 'color: #00ff00; font-weight: bold'
                        if val < 0.50: return 'color: #ff4b4b'
                    return ''

                st.dataframe(
                    results_df.style.format({
                        "Précision (Accuracy)": "{:.1%}",
                        "Score F1 (Fiabilité)": "{:.1%}",
                        "Prédiction Actuelle": "{:.1%}"
                    }).applymap(highlight_good, subset=["Précision (Accuracy)", "Score F1 (Fiabilité)"]),
                    use_container_width=True
                )
                
                # Récupération du meilleur modèle pour la simulation
                best_model = results_df.loc[results_df['Score F1 (Fiabilité)'].idxmax()]
                best_prob = best_model['Prédiction Actuelle']
                direction = "HAUSSE 🟩" if best_prob > 0.5 else "BAISSE 🟥"
                
                st.markdown("---")
                st.subheader(f"🔮 Simulation de Levier (Basé sur {direction})")
                st.info(f"Modèle de référence : **{best_model['Modèle']}** (F1: {best_model['Score F1 (Fiabilité)']:.1%}) prévoit une {direction} avec **{best_prob:.1%}** de confiance.")
                
                # 3. SIMULATEUR DE LEVIER
                leviers = [1, 2, 5, 10, 20]
                cols = st.columns(len(leviers))
                
                # Estimation du mouvement moyen (basé sur la volatilité récente)
                move_est = volatility * curr_price 
                
                for i, lev in enumerate(leviers):
                    with cols[i]:
                        st.markdown(f"<div class='metric-box'><h4 style='text-align:center;'>x{lev}</h4>", unsafe_allow_html=True)
                        
                        # Gain Potentiel (Si le marché bouge de 1 écart type)
                        gain_pct = volatility * lev
                        gain_usd = move_est * lev
                        
                        # Prix de Liquidation (Approximation : Prix entrée - (Prix entrée / Levier))
                        # Si Long
                        if best_prob > 0.5:
                            liq_price = curr_price * (1 - (1/lev))
                            liq_dist = - (1/lev)
                        else: # Short
                            liq_price = curr_price * (1 + (1/lev))
                            liq_dist = (1/lev)
                            
                        st.write(f"**Gain Est.:** :green[+{gain_pct:.1%}]")
                        st.write(f"*(~{gain_usd:.2f}$)*")
                        st.divider()
                        st.write(f"**💀 Liquidation:**")
                        st.write(f":red[{liq_price:.2f} $]")
                        st.write(f"*(Dist: {liq_dist:.1%})*")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                st.warning("⚠️ Ces calculs sont des estimations basées sur la volatilité passée. Le trading à levier comporte des risques de perte totale.")
                
            else:
                st.error("Données indisponibles.")

# --- LIVE MONITOR ---
elif app_mode == "📡 LIVE MONITOR":
    st.title("📡 Live Scanner")
    if 'monitoring' not in st.session_state: st.session_state.monitoring = False
    c1, c2 = st.columns([1, 5])
    if c1.button("▶️ START"): st.session_state.monitoring = True
    if c1.button("⏹️ STOP"): st.session_state.monitoring = False
    if st.session_state.monitoring: c2.markdown('<span class="live-badge">ON AIR</span>', unsafe_allow_html=True)
    
    placeholder = st.empty()
    if st.session_state.monitoring:
        with st.spinner("Scan..."):
            res = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
                futures = {exe.submit(quick_analyze, t): t for t in ALL_TICKERS}
                for f in concurrent.futures.as_completed(futures):
                    r = f.result()
                    if r: res.append(r)
            
            if res:
                df = pd.DataFrame(res)
                df = df.sort_values(by='Prob_Up', ascending=False)
                
                def color_row(v):
                    if v > 0.6: return 'color:#00ff00;font-weight:bold'
                    if v < 0.4: return 'color:#ff4b4b;font-weight:bold'
                    return 'color:gray'
                
                with placeholder.container():
                    st.dataframe(df[['Nom', 'Price', 'Change', 'RSI', 'Prob_Up']].style.format({'Price':'{:.2f}','Change':'{:.2%}','Prob_Up':'{:.1%}'}).applymap(color_row, subset=['Prob_Up']), use_container_width=True, height=600)
                    for _, r in df.iterrows():
                        if r['Prob_Up']>0.75: st.toast(f"🚀 {r['Nom']} UP!", icon="✅")
                        elif r['Prob_Up']<0.25: st.toast(f"📉 {r['Nom']} DOWN!", icon="🔻")
            time.sleep(15); st.rerun()