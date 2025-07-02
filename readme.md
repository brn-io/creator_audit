# CreatorAudit

O **CreatorAudit** é um projeto de portfólio criado para demonstrar, na prática, a aplicação de técnicas de análise de dados e machine learning em um problema real: a avaliação estratégica de canais do YouTube. Como profissional em transição para a área de Dados, utilizei Python, SQLite e bibliotecas modernas para construir um pipeline completo — da ingestão de dados à geração de insights visuais — de forma automatizada, leve e reprodutível. incluindo ingestão de dados, configuração do banco de dados, modelagem com machine learning, geração de insights e visualizações interativas.

## Principais recursos

- **Pipeline auto contido** — um único comando constrói o banco SQLite, treina o "Modelo de Mercado", audita os canais e salva os resultados.
- **Reproduzível** — sementes de aleatoriedade fixas, caminhos relativos e um `requirements.txt` com versões travadas.
- **NLP leve** — TF-IDF + Regressão Logística com lista de stopwords em português feita à mão (sem necessidade de baixar pacotes pesados como NLTK).
- **Enriquecimento com LLM opcional** — integração com o Gemini via `GEMINI_API_KEY` para sugestões de título e raciocínio, com fallback grácil caso a chave não esteja presente.
- **Compatível com Vega-Lite** — gráficos Altair são exportados em JSON puro e podem ser colados diretamente no [https://vega.github.io/editor/](https://vega.github.io/editor/).

## Início rápido

```bash
# Clone ou baixe o repositório
$ git clone https://github.com/<SEU_USUARIO>/creator-audit.git
$ cd creator-audit

# (Opcional) crie e ative um ambiente virtual
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instale as dependências
$ pip install -r requirements.txt

# Adicione sua chave da API Gemini ao arquivo .env (ou exporte como variável de ambiente)
$ echo "GEMINI_API_KEY=sk-..." > .env

# Execute o pipeline
$ python creator_audit.py
```

## Estrutura do repositório

```text
creator-audit/
├─ creator_audit.py        # Script principal do pipeline
├─ requirements.txt        # Dependências Python
├─ .gitignore              # Ignora .env, pasta data/, *.db, etc.
├─ .env.example            # Modelo de variáveis de ambiente
├─ data/                   # Dados gerados: CSV, banco SQLite, perfis (auto-criados)
└─ README.md               # Este arquivo
```

## Variáveis de ambiente

| Nome             | Finalidade                                            |
| ---------------- | ----------------------------------------------------- |
| GEMINI\_API\_KEY | (Opcional) Chave da API Gemini para sugestões via LLM |

Copie o `.env.example` para `.env` e preencha sua chave. **Nunca envie sua chave real para o repositório.**

## Criando / atualizando o dataset

O script gera automaticamente um dataset sintético caso `data/videos_dataset.csv` esteja ausente, permitindo que o projeto rode "fora da caixa". Para auditar canais reais, substitua o CSV mantendo os mesmos nomes de colunas.

## Visualizando a distribuição do CTR

Cada histograma de CTR por canal é impresso no terminal como um JSON Vega-Lite. Cole esse código no [Editor Vega-Lite](https://vega.github.io/editor/) para uma visualização interativa.

## Contribuições

Pull requests são bem-vindos! Se encontrar algum bug ou tiver sugestões de melhorias, abra uma issue para conversarmos.

## Licença

[MIT](LICENSE)

