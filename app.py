import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import concurrent.futures

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Market God v8 (Fusion)", layout="wide", page_icon="⚡")

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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LISTE DES MARCHÉS
# ==========================================
MARKETS = {
    "Cryptomonnaies": {
        "BTC-USD": "₿ Bitcoin (BTC)",
        "ETH-USD": "Ξ Ethereum (ETH)",
        "SOL-USD": "◎ Solana (SOL)",
        "BNB-USD": "🟡 Binance Coin (BNB)",
        "XRP-USD": "💧 Ripple (XRP)",
        "DOGE-USD": "🐶 Dogecoin (DOGE)",
        "AVAX-USD": "🔺 Avalanche (AVAX)"
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
        "INTC": "💾 Intel"
    },
    "Indices & Forex": {
        "^GSPC": "🇺🇸 S&P 500 (Indice US)",
        "^DJI": "🇺🇸 Dow Jones 30",
        "^IXIC": "🇺🇸 Nasdaq (Tech)",
        "GC=F": "🥇 OR (Gold)",
        "EURUSD=X": "💶 Euro vs Dollar",
        "USDJPY=X": "💴 Dollar vs Yen"
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
# 3. FONCTIONS MOTEUR IA
# ==========================================

def get_clean_data(ticker, period="2y"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if df.empty: return None
        df = df[['Close', 'Open', 'High', 'Low', 'Volume']]
        return df
    except: return None

def add_indicators(df):
    data = df.copy()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['SMA_20'] = data['Close'].rolling(20).mean()
    std = data['Close'].rolling(20).std()
    data['Upper'] = data['SMA_20'] + (std * 2)
    data['Lower'] = data['SMA_20'] - (std * 2)
    range_bb = data['Upper'] - data['Lower']
    range_bb = range_bb.replace(0, 0.0001)
    data['BB_Pct'] = (data['Close'] - data['Lower']) / range_bb
    
    data['Ret_1d'] = data['Close'].pct_change()
    data['Vol_20d'] = data['Ret_1d'].rolling(20).std()
    return data.dropna()

# --- 1. FONCTION POUR LE PORTFOLIO (AVEC PRIX D'ENTRÉE) ---
def analyze_portfolio_position(ticker, position_type, entry_price):
    try:
        df = get_clean_data(ticker, period="2y")
        if df is None: return None
        df = add_indicators(df)
        
        features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
        horizon = 5
        
        models = [
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        ]
        
        df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
        valid = df.dropna()
        X = valid[features]
        y = valid['Target']
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        
        probs = []
        for model in models:
            model.fit(X_train, y_train)
            last_row = df.iloc[[-1]][features]
            probs.append(model.predict_proba(last_row)[:, 1][0])
            
        avg_prob_up = np.mean(probs)
        curr_price = df['Close'].iloc[-1]
        
        # PnL Calculation
        pnl_pct = 0.0
        if position_type == "ACHAT (Long)":
            pnl_pct = (curr_price - entry_price) / entry_price
        else: 
            pnl_pct = (entry_price - curr_price) / entry_price

        # Conseil
        conseil = "NEUTRE"
        danger = False
        message = ""
        
        if position_type == "ACHAT (Long)":
            if avg_prob_up < 0.35: 
                conseil = "🔴 VENDRE"
                danger = True
                message = f"Risque de chute ({100-avg_prob_up*100:.0f}%)."
            elif avg_prob_up > 0.60:
                conseil = "🟢 GARDER"
                message = "Hausse probable."
            else:
                conseil = "🟡 ATTENDRE"
                message = "Marché indécis."
                
        elif position_type == "VENTE (Short)":
            if avg_prob_up > 0.65:
                conseil = "🔴 COUPER"
                danger = True
                message = f"Risque de hausse ({avg_prob_up*100:.0f}%)."
            elif avg_prob_up < 0.40:
                conseil = "🟢 GARDER"
                message = "Baisse probable."
            else:
                conseil = "🟡 ATTENDRE"
                message = "Marché indécis."

        return {
            "Ticker": ticker, "Price": curr_price, "Entry": entry_price,
            "PnL": pnl_pct, "Prob_Up": avg_prob_up,
            "Conseil": conseil, "Message": message, "Danger": danger
        }
    except: return None

# --- 2. FONCTION COMPLEXE (ANALYSE SOLO - v5 Style) ---
def deep_scan_asset(df):
    features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
    horizons = [3, 5, 10]
    models = {
        "Random Forest 🌳": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting 🚀": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    }
    best_res = None
    best_score = 0
    
    for h in horizons:
        temp = df.copy()
        temp['Target'] = (temp['Close'].shift(-h) > temp['Close']).astype(int)
        temp = temp.dropna()
        X = temp[features]
        y = temp['Target']
        split = len(X) - 100
        if split < 50: continue
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                if acc > best_score:
                    last_row = df.iloc[[-1]][features]
                    prob = model.predict_proba(last_row)[:, 1][0]
                    best_score = acc
                    best_res = {"algo": name, "horizon": h, "accuracy": acc, "prob": prob}
            except: continue
    return best_res

# --- 3. FONCTION RAPIDE (LIVE MONITOR - v5 Style avec RSI) ---
def quick_analyze(ticker):
    try:
        df = get_clean_data(ticker, period="1y")
        if df is None: return None
        df = add_indicators(df)
        if len(df) < 50: return None
        features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        valid = df.dropna()
        model = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=1)
        model.fit(valid[features], valid['Target'])
        last_row = df.iloc[[-1]][features]
        prob = model.predict_proba(last_row)[:, 1][0]
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = (curr_price - prev_price) / prev_price
        return {
            "Ticker": ticker,
            "Nom": TICKER_TO_NAME[ticker],
            "Price": curr_price,
            "Change": change,
            "Prob_Up": prob,
            "RSI": df['RSI'].iloc[-1] # Le retour du RSI
        }
    except: return None

# ==========================================
# 4. INTERFACE PRINCIPALE
# ==========================================
st.sidebar.header("🎛️ NAVIGATION")
app_mode = st.sidebar.radio("Choisir le mode :", ["💼 GESTION PORTFOLIO", "🔍 ANALYSE PROFONDE (Solo)", "📡 LIVE MONITOR (Multi)"])

# ------------------------------------------
# MODE 1 : GESTION PORTFOLIO
# ------------------------------------------
if app_mode == "💼 GESTION PORTFOLIO":
    st.title("💼 Mon Portefeuille & PnL")
    if 'portfolio' not in st.session_state: st.session_state.portfolio = []

    with st.expander("➕ Ajouter une position", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            cat_p = st.selectbox("Catégorie", list(MARKETS.keys()), key="p_cat")
            nom_asset = st.selectbox("Actif", list(MARKETS[cat_p].values()), key="p_asset")
        with c2:
            sens = st.selectbox("Sens", ["ACHAT (Long)", "VENTE (Short)"], key="p_sens")
            entry_input = st.number_input("Prix d'entrée ($)", min_value=0.0, step=0.01, format="%.2f")
            
        if st.button("Ajouter"):
            if entry_input > 0:
                ticker_code = NAME_TO_TICKER[nom_asset]
                if not any(d['Ticker'] == ticker_code for d in st.session_state.portfolio):
                    st.session_state.portfolio.append({
                        "Ticker": ticker_code, "Nom": nom_asset, "Sens": sens, "Entry": entry_input
                    })
                    st.success("Ajouté !")
                else: st.warning("Déjà présent.")

    st.markdown("---")
    if len(st.session_state.portfolio) > 0:
        if st.button("🔄 ACTUALISER PROFITS", type="primary"):
            with st.spinner("Analyse..."):
                results_p = []
                for pos in st.session_state.portfolio:
                    res = analyze_portfolio_position(pos['Ticker'], pos['Sens'], pos['Entry'])
                    if res:
                        res.update(pos)
                        results_p.append(res)
                for item in results_p:
                    border = "#ff4b4b" if item['Danger'] else ("#00ff00" if "GARDER" in item['Conseil'] else "#eba434")
                    pnl_class = "pnl-green" if item['PnL'] >= 0 else "pnl-red"
                    signe = "+" if item['PnL'] >= 0 else ""
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="border: 2px solid {border}; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: #1e1e1e;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h3 style="margin:0;">{item['Nom']} <span style="font-size:0.7em; color:gray;">({item['Sens']})</span></h3>
                                <div style="text-align:right;">
                                    <span style="font-size:0.9em;">PnL:</span> <span class="{pnl_class}">{signe}{item['PnL']:.2%}</span>
                                </div>
                            </div>
                            <hr style="margin:5px 0; border-color:#333;">
                            <div style="display:flex; justify-content:space-between;">
                                <div><p style="margin:0;">Entrée: <b>{item['Entry']:.2f}$</b> | Actuel: <b>{item['Price']:.2f}$</b></p></div>
                                <div style="text-align:right;"><b style="color:{border};">{item['Conseil']}</b></div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"🗑️ {item['Ticker']}", key=f"del_{item['Ticker']}"):
                            st.session_state.portfolio = [x for x in st.session_state.portfolio if x['Ticker'] != item['Ticker']]
                            st.rerun()
        else:
            for pos in st.session_state.portfolio: st.write(f"🔹 {pos['Nom']} ({pos['Entry']}$)")
            if st.button("Tout effacer"):
                st.session_state.portfolio = []
                st.rerun()

# ------------------------------------------
# MODE 2 : ANALYSE PROFONDE (SOLO)
# ------------------------------------------
elif app_mode == "🔍 ANALYSE PROFONDE (Solo)":
    st.title("🧠 Analyse Deep Learning")
    cat = st.selectbox("Catégorie", list(MARKETS.keys()))
    choix_nom = st.selectbox("Actif", list(MARKETS[cat].values()))
    ticker = NAME_TO_TICKER[choix_nom]
    
    if st.button(f"LANCER L'ANALYSE DE {choix_nom}", type="primary"):
        with st.spinner("Analyse..."):
            df = get_clean_data(ticker)
            if df is not None:
                df = add_indicators(df)
                last_price = df['Close'].iloc[-1]
                st.metric("Prix", f"{last_price:.2f}")
                
                res = deep_scan_asset(df)
                if res:
                    st.subheader(f"Probabilité Hausse : {res['prob']:.1%}")
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=res['prob']*100, 
                        gauge={'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 40], 'color': "red"}, {'range': [60, 100], 'color': "green"}]}))
                    st.plotly_chart(fig)
                    st.line_chart(df['Close'].tail(100))
            else: st.error("Erreur données")

# ------------------------------------------
# MODE 3 : LIVE MONITOR (RESTAURÉ STYLE v5)
# ------------------------------------------
elif app_mode == "📡 LIVE MONITOR (Multi)":
    st.title("📡 LIVE SCANNER")
    if 'monitoring' not in st.session_state: st.session_state.monitoring = False
    
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("▶️ START", type="primary"): st.session_state.monitoring = True
        if st.button("⏹️ STOP"): st.session_state.monitoring = False
    with c2:
        if st.session_state.monitoring: st.markdown('<span class="live-badge">SCAN EN COURS...</span>', unsafe_allow_html=True)
    
    placeholder = st.empty()
    if st.session_state.monitoring:
        with st.spinner("Analyse parallèle..."):
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(quick_analyze, t): t for t in ALL_TICKERS}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res: results.append(res)
            
            if results:
                df_res = pd.DataFrame(results)
                df_res['Signal'] = abs(df_res['Prob_Up'] - 0.5)
                df_res = df_res.sort_values(by='Signal', ascending=False)
                
                with placeholder.container():
                    # Top 3
                    top = df_res.head(3)
                    cols = st.columns(3)
                    for i, (idx, row) in enumerate(top.iterrows()):
                        label = "🟢 ACHAT" if row['Prob_Up'] > 0.5 else "🔴 VENTE"
                        cols[i].metric(f"{row['Nom']}", f"{row['Price']:.2f}", f"{row['Change']:.2%}")
                    
                    st.markdown("---")
                    
                    # === LA PARTIE QUE TU VOULAIS RECUPERER (COULEURS) ===
                    def color_rows(val):
                        if val > 0.65: return 'color: #00ff00; font-weight: bold' # Vert
                        if val < 0.35: return 'color: #ff4b4b; font-weight: bold' # Rouge
                        return 'color: gray'

                    # J'ai remis le RSI et la coloration
                    st.dataframe(
                        df_res[['Nom', 'Price', 'Change', 'RSI', 'Prob_Up']].style.format({
                            'Price': '{:.2f}', 'Change': '{:.2%}', 'RSI': '{:.0f}', 'Prob_Up': '{:.1%}'
                        }).applymap(color_rows, subset=['Prob_Up']),
                        use_container_width=True,
                        height=600
                    )
                    
                    # === LA PARTIE NOTIFICATIONS (TOASTS) ===
                    for _, row in df_res.iterrows():
                        if row['Prob_Up'] > 0.75:
                            st.toast(f"🚀 {row['Nom']} : HAUSSE PROBABLE ({row['Prob_Up']:.0%})", icon="🟢")
                        elif row['Prob_Up'] < 0.25:
                            st.toast(f"📉 {row['Nom']} : CHUTE PROBABLE ({row['Prob_Up']:.0%})", icon="🔴")
            
            time.sleep(60)
            st.rerun()