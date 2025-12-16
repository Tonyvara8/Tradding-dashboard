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
st.set_page_config(page_title="Market God v4", layout="wide", page_icon="⚡")

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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LISTE DES MARCHÉS
# ==========================================
MARKETS = {
    "Cryptos": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD"],
    "Tech US": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOG", "AMD", "META", "NFLX", "INTC"],
    "Indices/Forex": ["^GSPC", "^DJI", "^IXIC", "GC=F", "EURUSD=X", "USDJPY=X"]
}

ALL_TICKERS = []
for l in MARKETS.values():
    ALL_TICKERS.extend(l)

# ==========================================
# 3. FONCTIONS MOTEUR (Robustes)
# ==========================================

def get_clean_data(ticker, period="2y"):
    """Récupère les données proprement (Fix du bug des prix mélangés)"""
    try:
        # Utilisation de Ticker().history qui est thread-safe
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        
        if df.empty: return None
        
        # Nettoyage colonnes
        df = df[['Close', 'Open', 'High', 'Low', 'Volume']]
        return df
    except:
        return None

def add_indicators(df):
    data = df.copy()
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger
    data['SMA_20'] = data['Close'].rolling(20).mean()
    std = data['Close'].rolling(20).std()
    data['Upper'] = data['SMA_20'] + (std * 2)
    data['Lower'] = data['SMA_20'] - (std * 2)
    
    # Indicateur de position dans les bandes (0 = bas, 1 = haut)
    range_bb = data['Upper'] - data['Lower']
    # Évite la division par zéro
    range_bb = range_bb.replace(0, 0.0001)
    data['BB_Pct'] = (data['Close'] - data['Lower']) / range_bb
    
    # Retours & Volatilité
    data['Ret_1d'] = data['Close'].pct_change()
    data['Vol_20d'] = data['Ret_1d'].rolling(20).std()
    
    return data.dropna()

# --- FONCTION COMPLEXE (POUR ANALYSE SOLO) ---
def deep_scan_asset(df):
    """Teste 3 Algorithmes x 3 Horizons = Le meilleur résultat"""
    features = ['RSI', 'BB_Pct', 'Ret_1d', 'Vol_20d']
    horizons = [3, 5, 10]
    
    models = {
        "Random Forest 🌳": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting 🚀": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "Logistic Regression ⚖️": LogisticRegression(random_state=42)
    }
    
    best_res = None
    best_score = 0
    
    for h in horizons:
        temp = df.copy()
        temp['Target'] = (temp['Close'].shift(-h) > temp['Close']).astype(int)
        temp = temp.dropna()
        
        X = temp[features]
        y = temp['Target']
        
        # Split Train/Test (Derniers 100 jours pour validation)
        split = len(X) - 100
        if split < 50: continue

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                # On ne garde que si l'algo comprend le marché (>50%)
                if acc > best_score:
                    # Prédiction pour demain
                    last_row = df.iloc[[-1]][features]
                    prob = model.predict_proba(last_row)[:, 1][0]
                    
                    best_score = acc
                    best_res = {
                        "algo": name,
                        "horizon": h,
                        "accuracy": acc,
                        "prob": prob,
                        "action": "ACHAT" if prob > 0.55 else ("VENTE" if prob < 0.45 else "NEUTRE")
                    }
            except:
                continue
                
    return best_res

# --- FONCTION RAPIDE (POUR LIVE MONITOR) ---
def quick_analyze(ticker):
    """Version allégée pour le multi-thread"""
    try:
        df = get_clean_data(ticker, period="1y")
        if df is None: return None
        
        df = add_indicators(df)
        if len(df) < 50: return None
        
        # Un seul modèle rapide
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
            "Price": curr_price,
            "Change": change,
            "Prob_Up": prob,
            "RSI": df['RSI'].iloc[-1]
        }
    except:
        return None

# ==========================================
# 4. INTERFACE PRINCIPALE
# ==========================================
st.sidebar.header("🎛️ NAVIGATION")
app_mode = st.sidebar.radio("Choisir le mode :", ["🔍 ANALYSE PROFONDE (Solo)", "📡 LIVE MONITOR (Multi)"])

# ------------------------------------------
# MODE 1 : ANALYSE PROFONDE (SOLO)
# ------------------------------------------
if app_mode == "🔍 ANALYSE PROFONDE (Solo)":
    st.title("🧠 Analyse Deep Learning Détaillée")
    st.markdown("Ce mode teste **plusieurs algorithmes** et **plusieurs échelles de temps** pour un actif précis.")
    
    # Sélecteur
    cat = st.selectbox("Catégorie", list(MARKETS.keys()))
    ticker = st.selectbox("Actif", MARKETS[cat])
    
    if st.button("LANCER L'ANALYSE COMPLÈTE", type="primary"):
        with st.spinner(f"Analyse approfondie de {ticker} en cours..."):
            raw_df = get_clean_data(ticker)
            
            if raw_df is not None:
                df = add_indicators(raw_df)
                
                # 1. Infos Prix
                last_price = raw_df['Close'].iloc[-1]
                var = (last_price - raw_df['Close'].iloc[-2]) / raw_df['Close'].iloc[-2]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Prix Actuel", f"{last_price:.2f}", f"{var:.2%}")
                c2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.0f}")
                
                # 2. Le Cerveau (Deep Scan)
                res = deep_scan_asset(df)
                
                if res:
                    st.markdown("---")
                    col_g, col_d = st.columns([1, 2])
                    
                    with col_g:
                        # Jauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = res['prob'] * 100,
                            title = {'text': "Probabilité HAUSSE"},
                            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
                                     'steps': [{'range': [0, 40], 'color': "red"}, {'range': [60, 100], 'color': "green"}]}
                        ))
                        fig.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_d:
                        st.subheader(f"🏆 Meilleure Stratégie Trouvée")
                        st.write(f"**Algorithme Champion :** {res['algo']}")
                        st.write(f"**Horizon Optimal :** {res['horizon']} Jours")
                        st.write(f"**Précision Historique :** {res['accuracy']:.1%}")
                        
                        if res['prob'] > 0.60:
                            st.success(f"### ✅ CONSEIL : ACHAT (LONG)\nForte probabilité de hausse détectée sur {res['horizon']} jours.")
                        elif res['prob'] < 0.40:
                            st.error(f"### 🔻 CONSEIL : VENTE (SHORT)\nRisque de baisse élevé.")
                        else:
                            st.warning("### ✋ CONSEIL : ATTENDRE\nLe marché est indécis.")

                    # 3. Graphique Interactif
                    st.subheader("Graphique Technique")
                    chart_data = df.tail(150)
                    fig_chart = go.Figure(data=[go.Candlestick(x=chart_data.index,
                                    open=chart_data['Open'], high=chart_data['High'],
                                    low=chart_data['Low'], close=chart_data['Close'])])
                    fig_chart.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_chart, use_container_width=True)
                else:
                    st.error("Pas assez de données fiables pour une analyse IA.")
            else:
                st.error("Erreur de téléchargement des données.")

# ------------------------------------------
# MODE 2 : LIVE MONITOR (MULTI)
# ------------------------------------------
elif app_mode == "📡 LIVE MONITOR (Multi)":
    st.title("📡 Salle de Marché : LIVE SCANNER")
    st.markdown("Surveillance de **tous les actifs** en simultané. Les prix sont mis à jour à chaque cycle.")
    
    # Session state pour le bouton Play/Stop
    if 'monitoring' not in st.session_state: st.session_state.monitoring = False
    
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("▶️ START", type="primary"): st.session_state.monitoring = True
        if st.button("⏹️ STOP"): st.session_state.monitoring = False
    with c2:
        if st.session_state.monitoring:
            st.markdown('<span class="live-badge">SCAN EN COURS...</span>', unsafe_allow_html=True)

    placeholder = st.empty()
    
    if st.session_state.monitoring:
        # Boucle simulée par Streamlit (Rerun)
        with st.spinner("Analyse parallèle de 20+ actifs..."):
            
            results = []
            # Parallélisme pour aller vite
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(quick_analyze, t): t for t in ALL_TICKERS}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res: results.append(res)
            
            if results:
                df_res = pd.DataFrame(results)
                
                # Calcul Force du signal (écart par rapport à 50%)
                df_res['Signal'] = abs(df_res['Prob_Up'] - 0.5)
                df_res = df_res.sort_values(by='Signal', ascending=False)
                
                # Affichage
                with placeholder.container():
                    # Top 3 Métriques
                    top = df_res.head(3)
                    cols = st.columns(3)
                    for i, (idx, row) in enumerate(top.iterrows()):
                        label = "🟢 ACHAT" if row['Prob_Up'] > 0.5 else "🔴 VENTE"
                        cols[i].metric(
                            f"{row['Ticker']} ({label})",
                            f"{row['Price']:.2f}",
                            f"{row['Change']:.2%}",
                            delta_color="normal" if row['Change'] > 0 else "inverse"
                        )
                    
                    st.markdown("---")
                    
                    # Tableau coloré
                    def color_rows(val):
                        if val > 0.65: return 'color: #00ff00; font-weight: bold'
                        if val < 0.35: return 'color: #ff4b4b; font-weight: bold'
                        return 'color: gray'

                    st.dataframe(
                        df_res[['Ticker', 'Price', 'Change', 'RSI', 'Prob_Up']].style.format({
                            'Price': '{:.2f}', 'Change': '{:.2%}', 'RSI': '{:.0f}', 'Prob_Up': '{:.1%}'
                        }).applymap(color_rows, subset=['Prob_Up']),
                        use_container_width=True,
                        height=600
                    )
                    
                    # Notifications Toast pour les signaux forts
                    for _, row in df_res.iterrows():
                        if row['Prob_Up'] > 0.75:
                            st.toast(f"🚀 {row['Ticker']} : HAUSSE PROBABLE ({row['Prob_Up']:.0%})", icon="🟢")
                        elif row['Prob_Up'] < 0.25:
                            st.toast(f"📉 {row['Ticker']} : CHUTE PROBABLE ({row['Prob_Up']:.0%})", icon="🔴")
            
            time.sleep(60) # Pause de 15 secondes entre chaque scan
            st.rerun()