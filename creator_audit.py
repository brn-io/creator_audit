# -------------------------------------------------------------------
# CreatorAudit – Versão Final de Portfólio (Didática e Comentada)
# -------------------------------------------------------------------
"""
Este script representa a versão final e autoexplicativa do projeto CreatorAudit.
O objetivo é criar um pipeline de análise de dados de ponta a ponta que seja
robusto, fácil de executar e que demonstre um pensamento analítico sofisticado,
ideal para um portfólio de Analista de Dados ou Cientista de Dados.

A lógica principal foi refinada para:
1.  Treinar um único "Modelo de Mercado" para entender o que gera sucesso no nicho.
2.  Aplicar este modelo para fazer uma análise comparativa em cada canal.
3.  Gerar sugestões estratégicas combinando as melhores oportunidades de mercado
    com os pontos fortes e objetivos do canal.
4.  Utilizar uma LLM (se configurada) para enriquecer a apresentação dos insights.
5.  Ser totalmente autônomo e não interativo para garantir execução sem atrito.
"""

# --- Seção de Imports ---
from pathlib import Path
import sqlite3
import json
import os
import random
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import altair as alt

# --- Correção: Carregar variáveis do arquivo .env, se existir ---
try:
    from dotenv import load_dotenv
    load_dotenv(".env")  # Corrigido para usar o arquivo .env
except ImportError:
    print("[AVISO] python-dotenv não instalado. Variáveis de ambiente do arquivo .env não serão carregadas.")

# --- Configuração Inicial e Constantes ---
# Seeds fixos para garantir que os resultados aleatórios sejam os mesmos a cada execução.
# Essencial para reprodutibilidade em um projeto de análise.
random.seed(42)
np.random.seed(42)

# Usando caminhos relativos para máxima compatibilidade de ambiente.
# O script criará uma pasta 'data' no mesmo diretório onde for executado.
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "creator.db"
CSV_PATH = DATA_DIR / "videos_dataset.csv"
USER_PROFILE_PATH = DATA_DIR / "user_profile.json"
DATA_DIR.mkdir(exist_ok=True) # Cria a pasta 'data' se ela não existir.

# Lista de stop words em português para ser usada pelo TfidfVectorizer.
# Isso evita a necessidade de instalar bibliotecas pesadas como NLTK e corrige
# o erro da biblioteca scikit-learn, que não tem uma lista nativa para 'portuguese'.
PORTUGUESE_STOP_WORDS = [
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'na',
    'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser'
]

# --- 1. Gerenciamento de Dados e Perfil do Usuário ---

def create_default_user_profile():
    """
    Cria um perfil de usuário padrão em um arquivo JSON.
    Esta função é chamada se nenhum perfil for encontrado, garantindo que o script
    seja não interativo e evite o `EOFError` em ambientes automatizados.
    O perfil pode ser editado manualmente para testar diferentes cenários.
    """
    print("\n=== Gerando Perfil do Canal Padrão ===")
    profile = {
        "temas": ["python", "carreira", "tutorial"],
        "tom": "técnico e didático",
        "objetivo": "aumentar o número de inscritos"
    }
    with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4)
    print(f"Perfil padrão salvo com sucesso em '{USER_PROFILE_PATH}'.")
    return profile

def load_user_profile():
    """Carrega o perfil do usuário do arquivo JSON ou cria um novo se não existir."""
    if USER_PROFILE_PATH.exists():
        print(f"Carregando perfil de usuário existente de '{USER_PROFILE_PATH}'.")
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return create_default_user_profile()

def generate_fake_dataset(n_channels: int = 3, videos_per_channel: int = 100):
    """
    Gera um dataset CSV fictício.
    A lógica cria um dataset com padrões discerníveis (canais com temas preferidos,
    tópicos com diferentes potenciais de CTR) para que o modelo de ML tenha o que aprender.
    """
    topics = {"python": 0.08, "javascript": 0.07, "review": 0.06, "tutorial": 0.09, "vlog": 0.03, "unboxing": 0.05, "setup": 0.04, "carreira": 0.06}
    channels = [f"channel_{i+1}" for i in range(n_channels)]
    rows = []
    for ch in channels:
        channel_fav_topics = random.sample(list(topics.keys()), 3)
        base_dt = datetime.now() - timedelta(days=365)
        for idx in range(videos_per_channel):
            title_topic = random.choice(channel_fav_topics) if random.random() < 0.7 else random.choice(list(topics.keys()))
            title = f"O melhor {title_topic} de tech para iniciantes em 2025"
            base_ctr = topics[title_topic]
            ctr = max(0, np.random.normal(loc=base_ctr + (channels.index(ch) * 0.005), scale=0.015))
            views = int(np.random.exponential(scale=5000) * (1 + ctr * 15))
            likes = int(views * (0.03 + ctr))
            rows.append({"channel_id": ch, "video_id": f"{ch}_vid_{idx}", "title": title, "published_at": (base_dt + timedelta(days=idx * 3)).strftime("%Y-%m-%d"), "views": views, "likes": likes, "click_through_rate": ctr, "duration_sec": np.random.randint(120, 1800)})
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Dataset fictício gerado e salvo em '{CSV_PATH}'")
    return df

def load_dataset() -> pd.DataFrame:
    """Carrega o dataset do CSV, ou o gera se o arquivo não existir."""
    if not CSV_PATH.exists(): return generate_fake_dataset()
    return pd.read_csv(CSV_PATH)

# --- 2. Estruturação do Banco de Dados ---

def init_db(df: pd.DataFrame, user_profile=None):
    """
    Atua como o "arquiteto" do sistema, inicializando o banco de dados SQLite.
    A função apaga o banco de dados anterior a cada execução para garantir um
    ambiente limpo e reprodutível, uma prática recomendada para este tipo de script.
    """
    if DB_PATH.exists(): DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # A criação de múltiplas tabelas demonstra um bom entendimento de modelagem de dados.
    cur.executescript("""
        CREATE TABLE channels (channel_id TEXT PRIMARY KEY, title TEXT, category TEXT);
        CREATE TABLE videos (video_id TEXT PRIMARY KEY, channel_id TEXT, published_at TEXT, title TEXT, views INTEGER, likes INTEGER, click_through_rate REAL, duration_sec INTEGER, FOREIGN KEY (channel_id) REFERENCES channels (channel_id));
        CREATE TABLE audits (audit_id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT, finding_type TEXT, finding_text TEXT, created_at TEXT);
        CREATE TABLE suggestions (suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT, title TEXT, reasoning TEXT, created_at TEXT);
        CREATE TABLE user_profile (id INTEGER PRIMARY KEY, temas TEXT, tom TEXT, objetivo TEXT);
    """)
    channels_meta = df[["channel_id"]].drop_duplicates()
    channels_meta["title"] = channels_meta["channel_id"].str.replace("_", " ").str.title()
    channels_meta["category"] = "Technology"
    channels_meta.to_sql("channels", conn, if_exists="append", index=False)
    df.to_sql("videos", conn, if_exists="append", index=False)
    if user_profile:
        cur.execute("INSERT INTO user_profile (temas, tom, objetivo) VALUES (?, ?, ?)", (", ".join(user_profile["temas"]), user_profile["tom"], user_profile["objetivo"]))
    conn.commit()
    conn.close()
    print(f"Banco de dados '{DB_PATH}' inicializado com sucesso.")

# --- 3. Modelagem de Mercado e Análise de Canal ---

def build_market_model(df: pd.DataFrame):
    """
    Coração analítico do projeto. Treina um único "Modelo de Mercado" para
    entender os fatores de sucesso no nicho como um todo. Esta abordagem é mais
    robusta do que treinar modelos pequenos e fracos para cada canal.
    """
    df_copy = df.copy()
    # Define "sucesso" como um vídeo no quartil superior de CTR (75%).
    df_copy["is_high_ctr"] = (df_copy["click_through_rate"] > df_copy["click_through_rate"].quantile(0.75)).astype(int)
    X, y = df_copy["title"], df_copy["is_high_ctr"]
    # Divide os dados em treino e teste para avaliar o modelo de forma justa.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Cria um pipeline que primeiro vetoriza o texto e depois aplica a classificação.
    pipe = Pipeline([
        # TfidfVectorizer transforma texto em vetores numéricos, dando mais importância a palavras raras e significativas.
        # ngram_range=(1, 2) analisa tanto palavras individuais quanto pares de palavras (ex: "python tech").
        ("vect", TfidfVectorizer(stop_words=PORTUGUESE_STOP_WORDS, ngram_range=(1, 2))),
        # Regressão Logística é um classificador simples, rápido e interpretável.
        # class_weight='balanced' ajuda o modelo a lidar com datasets onde uma classe (ex: sucesso) é mais rara que a outra.
        ("clf", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    
    # Avalia a performance do modelo no conjunto de teste.
    y_pred_prob, y_pred = pipe.predict_proba(X_test)[:, 1], pipe.predict(X_test)
    auc, acc = roc_auc_score(y_test, y_pred_prob), accuracy_score(y_test, y_pred)
    print(f"\n=== Modelo de Mercado (Visão Geral) ===")
    print(f"  Métricas de Performance: AUC = {auc:.3f}, Acurácia = {acc:.2f}")
    
    return pipe

def get_top_market_keywords(pipe, n=10):
    """
    "Pergunta" ao modelo quais palavras/frases ele aprendeu que são mais
    positivamente correlacionadas com o sucesso (alto CTR).
    """
    vectorizer, classifier = pipe.named_steps['vect'], pipe.named_steps['clf']
    feature_names, coefs = vectorizer.get_feature_names_out(), classifier.coef_[0]
    top_indices = np.argsort(coefs)[-n:]
    return {feature_names[i]: coefs[i] for i in top_indices}

def analyze_channel(channel_df: pd.DataFrame, market_keywords: dict):
    """
    Realiza a análise comparativa: cruza os dados de um canal específico
    com as tendências gerais do mercado identificadas pelo modelo.
    """
    channel_df_copy = channel_df.copy()
    channel_vectorizer = TfidfVectorizer(stop_words=PORTUGUESE_STOP_WORDS).fit(channel_df_copy["title"])
    channel_keywords = set(channel_vectorizer.get_feature_names_out())
    
    # Quais tendências de mercado o canal já utiliza?
    signature_keywords = channel_keywords.intersection(market_keywords.keys())
    # Quais tendências de mercado o canal está ignorando?
    untapped_opportunities = set(market_keywords.keys()) - channel_keywords
    
    # Qual formato de vídeo (tutorial, review, etc.) funciona melhor para este canal?
    known_formats = ["tutorial", "review", "vlog", "unboxing", "setup", "live", "shorts"]
    pattern = fr'\b({"|".join(known_formats)})\b'
    channel_df_copy['format'] = channel_df_copy['title'].str.extract(pattern, flags=re.IGNORECASE, expand=False).str.lower()
    channel_df_copy['format'].fillna('geral', inplace=True)
    successful_formats = channel_df_copy.loc[channel_df_copy['click_through_rate'].nlargest(5).index, 'format'].mode()
    best_format = successful_formats[0] if not successful_formats.empty else "tutorial"

    return {"signature": list(signature_keywords), "untapped": list(untapped_opportunities), "best_format": best_format}

# --- 4. Geração de Sugestões e Visualização ---

def get_suggestion_components(findings: dict, user_profile: dict):
    """
    Prepara os "ingredientes" para a sugestão final. Esta função contém a
    lógica de negócio que alinha as oportunidades de mercado com o perfil do usuário.
    """
    untapped_aligned = [k for k in findings["untapped"] if k in user_profile["temas"]]
    best_keyword, reason_intro = "", ""
    if untapped_aligned:
        best_keyword = untapped_aligned[0]
        reason_intro = f"Analisamos que '{best_keyword}' é um tema com alto potencial de mercado, alinhado ao seu foco em '{best_keyword}', que você ainda não explorou."
    elif findings["untapped"]:
        non_format_opportunities = [k for k in findings["untapped"] if k not in findings["best_format"]]
        best_keyword = random.choice(non_format_opportunities) if non_format_opportunities else random.choice(findings["untapped"])
        reason_intro = f"Identificamos '{best_keyword}' como uma palavra-chave com alto potencial de mercado que representa uma nova avenida de conteúdo para seu canal."
    else:
        best_keyword = findings["signature"][0] if findings["signature"] else "tech"
        reason_intro = f"Nossa análise indica que dobrar a aposta em '{best_keyword}', sua palavra-chave de maior sucesso, é uma estratégia sólida."
    return {"best_keyword": best_keyword, "best_format": findings["best_format"], "reason_intro": reason_intro}

def generate_output_with_llm(components: dict, user_profile: dict):
    """
    Usa uma LLM (Gemini) para gerar o título e o raciocínio.
    Implementa um sistema de fallback robusto: se a chave de API não estiver
    disponível ou a chamada falhar, ele recorre a um método baseado em regras,
    garantindo que o script nunca trave.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    # Função interna para a lógica de fallback baseada em regras.
    # Isso evita duplicação de código e torna a função principal mais limpa.
    def rule_based_fallback():
        best_format = components['best_format']
        best_keyword = components['best_keyword']
        # Lógica de título aprimorada para evitar frases estranhas.
        if best_format == 'geral':
            title = f"Um Olhar Aprofundado em {best_keyword.title()} para 2025"
        else:
            title = f"{best_format.title()} de {best_keyword}: Guia Definitivo"
        
        reasoning = (
            f"{components['reason_intro']} "
            f"Além disso, vídeos no formato '{best_format}' historicamente geram os melhores resultados de CTR para o seu canal."
        )
        return title, f"(LLM não configurada ou falhou) • {reasoning}"

    # Se não há chave, usa o fallback imediatamente.
    if not api_key:
        return rule_based_fallback()
    
    # Se há chave, tenta usar a LLM.
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Você é o "CreatorOS", um estrategista de conteúdo para o YouTube. Sua tarefa é criar uma sugestão completa (título e raciocínio) para um criador.

        DADOS DA ANÁLISE:
        - Tópico/Palavra-chave com alto potencial: "{components['best_keyword']}"
        - Formato de vídeo de maior sucesso para o canal: "{components['best_format']}"
        - Objetivo principal do criador: "{user_profile['objetivo']}"
        - Tom de voz do canal: "{user_profile['tom']}"

        INSTRUÇÕES:
        1.  Crie um Título de vídeo cativante e otimizado para busca (SEO) que combine o tópico e o formato.
        2.  Escreva um parágrafo de "Raciocínio Estratégico" explicando de forma encorajadora e profissional por que esta é uma boa sugestão, usando os dados da análise.

        Responda em formato JSON, com as chaves "titulo_sugerido" e "raciocinio_estrategico".
        """
        response = model.generate_content(prompt, generation_config={"temperature": 0.7})
        # Limpa o output da LLM para garantir que seja um JSON válido
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_text)
        return result["titulo_sugerido"], result["raciocinio_estrategico"]
    except Exception as e:
        print(f"  [AVISO] Erro na API da LLM: {e}. Usando fallback.")
        # CORREÇÃO: Chama a função de fallback se a API falhar, garantindo que o programa não trave.
        return rule_based_fallback()

def plot_ctr_distribution(df: pd.DataFrame, channel_id: str):
    """
    Gera a especificação JSON de um gráfico de distribuição de CTR.
    Esta abordagem evita salvar arquivos HTML, eliminando o `FileNotFoundError`
    em ambientes restritos (sandbox).
    """
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('click_through_rate:Q', bin=alt.Bin(maxbins=30), title='Taxa de Cliques (CTR)'),
        y=alt.Y('count():Q', title='Nº de Vídeos'),
        tooltip=['count()', 'click_through_rate']
    ).properties(
        title={'text': f'Distribuição de CTR - {channel_id}', 'subtitle': 'A maioria dos vídeos se concentra em uma faixa de CTR específica.'}
    ).configure_axis(grid=False).configure_view(strokeOpacity=0)
    
    chart_json = chart.to_json()
    print(f"\n  --- GRÁFICO DE DISTRIBUIÇÃO DE CTR (JSON) ---")
    print("  Copie o código JSON abaixo e cole no Editor do Vega-Lite para visualizar: https://vega.github.io/editor/")
    print(chart_json)

# --- 5. Persistência e Execução ---

def store_results(channel_id, findings, suggestion_title, final_reasoning):
    """Salva os resultados da análise e sugestão no banco de dados."""
    ts = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO audits(channel_id, finding_type, finding_text, created_at) VALUES (?, ?, ?, ?)", (channel_id, "signature_keywords", ", ".join(findings['signature']), ts))
    cur.execute("INSERT INTO audits(channel_id, finding_type, finding_text, created_at) VALUES (?, ?, ?, ?)", (channel_id, "untapped_opportunities", ", ".join(findings['untapped']), ts))
    cur.execute("INSERT INTO suggestions(channel_id, title, reasoning, created_at) VALUES (?, ?, ?, ?)", (channel_id, suggestion_title, final_reasoning, ts))
    conn.commit()
    conn.close()

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    # 1. PREPARAÇÃO: Carrega o perfil do usuário e os dados.
    user_profile = load_user_profile()
    full_df = load_dataset()
    init_db(full_df, user_profile)

    # 2. MODELAGEM DE MERCADO: Treina o modelo uma única vez com todos os dados.
    market_model = build_market_model(full_df)
    market_keywords = get_top_market_keywords(market_model)
    print(f"  Principais Palavras-Chave do Mercado: {list(market_keywords.keys())}")

    # 3. ANÁLISE POR CANAL: Itera sobre cada canal para aplicar o modelo e gerar insights.
    for ch_id in full_df["channel_id"].unique():
        print(f"\n▶ Análise Estratégica para o Canal: {ch_id}")
        channel_df = full_df[full_df["channel_id"] == ch_id].copy()
        
        # Gera os achados analíticos
        findings = analyze_channel(channel_df, market_keywords)
        print(f"  - [Auditoria] Palavras-assinatura do canal: {findings['signature']}")
        print(f"  - [Auditoria] Oportunidades de mercado não exploradas: {findings['untapped']}")
        print(f"  - [Auditoria] Formato de maior sucesso do canal: '{findings['best_format']}'")

        # Prepara os componentes para a sugestão
        suggestion_components = get_suggestion_components(findings, user_profile)
        # Gera o output final, usando a LLM se disponível
        suggestion_title, final_reasoning = generate_output_with_llm(suggestion_components, user_profile)
        
        print("\n  --- SUGESTÃO ESTRATÉGICA GERADA ---")
        print(f"  Título Sugerido: {suggestion_title}")
        print(f"  Raciocínio: {final_reasoning}")
        
        # Salva os resultados e gera o gráfico
        store_results(ch_id, findings, suggestion_title, final_reasoning)
        plot_ctr_distribution(channel_df, ch_id)

    print(f"\n✅ Análise concluída. Verifique o banco '{DB_PATH}'.")