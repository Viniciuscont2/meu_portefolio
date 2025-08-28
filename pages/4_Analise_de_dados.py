# Importação das bibliotecas necessárias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# --- Configuração da Página ---
# Define o título da página, o ícone e o layout. O layout "wide" aproveita melhor o espaço da tela.
st.set_page_config(
    page_title="Dashboard Profissional - Análise de Mercado de IA",
    page_icon="🤖",
    layout="wide"
)

# --- Carregamento dos Dados ---
# Função para carregar os dados. O decorator @st.cache_data evita que os dados sejam recarregados
# a cada interação do usuário, otimizando a performance do dashboard.
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("ai_job_market_insights.csv")
        return df
    except FileNotFoundError:
        st.error("Erro: O arquivo 'ai_job_market_insights.csv' não foi encontrado.")
        st.info("Por favor, certifique-se de que o arquivo de dados está na mesma pasta que o script Python.")
        return None

df = carregar_dados()

# Se o dataframe não for carregado (devido ao erro), interrompe a execução do script.
if df is None:
    st.stop()

# --- Transformação dos Tipos de Dados ---
# Convertendo as variáveis ordinais para o tipo 'category' com a ordem correta.
# Isso garante que as visualizações e análises respeitem a hierarquia dos dados.
ordem_company_size = ['Small', 'Medium', 'Large']
ordem_adoption_level = ['Low', 'Medium', 'High']
ordem_automation_risk = ['Low', 'Medium', 'High']

df['Company_Size'] = pd.Categorical(df['Company_Size'], categories=ordem_company_size, ordered=True)
df['AI_Adoption_Level'] = pd.Categorical(df['AI_Adoption_Level'], categories=ordem_adoption_level, ordered=True)
df['Automation_Risk'] = pd.Categorical(df['Automation_Risk'], categories=ordem_automation_risk, ordered=True)

# --- Título Principal do Dashboard ---
st.title("📊 Dashboard de Análise do Mercado de Trabalho em IA")
st.markdown("---")

# --- Início da Aba de Análise de Dados (conforme o enunciado do CP1) ---

# --- SEÇÃO 1: Apresentação dos Dados e Tipos de Variáveis ---
st.header("1. Apresentação e Estrutura dos Dados")

# Expansor para manter a UI limpa. O usuário pode clicar para ver os detalhes.
with st.expander("Clique para ver a estrutura do Dataset"):
    st.subheader("Amostra dos Dados")
    st.write("Abaixo estão as cinco primeiras linhas do conjunto de dados para uma visualização inicial.")
    st.dataframe(df.head())

    st.subheader("Dimensões do Dataset")
    st.write(f"O conjunto de dados possui **{df.shape[0]} linhas** (observações) e **{df.shape[1]} colunas** (variáveis).")

# --- TABELA ÚNICA DE CLASSIFICAÇÃO DE VARIÁVEIS ---
st.subheader("Classificação das Variáveis")
st.write("A tabela a seguir apresenta cada variável, seu tipo de dado técnico (identificado pelo Pandas) e sua classificação estatística.")

# Mapeamento da classificação estatística para cada variável
classificacao = {
    'Job_Title': 'Qualitativa Nominal',
    'Industry': 'Qualitativa Nominal',
    'Location': 'Qualitativa Nominal',
    'Required_Skills': 'Qualitativa Nominal',
    'Remote_Friendly': 'Qualitativa Nominal',
    'Job_Growth_Projection': 'Qualitativa Nominal',
    'Company_Size': 'Qualitativa Ordinal',
    'AI_Adoption_Level': 'Qualitativa Ordinal',
    'Automation_Risk': 'Qualitativa Ordinal',
    'Salary_USD': 'Quantitativa Contínua'
}

# Criação do DataFrame para a tabela de classificação
df_tipos = pd.DataFrame({
    "Variável": df.columns,
    "Tipo de Dado (Pandas)": [str(dtype) for dtype in df.dtypes],
    "Classificação Estatística": [classificacao.get(col, "Não Classificado") for col in df.columns]
})
st.table(df_tipos)


st.subheader("Perguntas de Análise Definidas")
st.markdown("""
- Qual é a distribuição salarial para profissionais no mercado de IA?
- O tamanho da empresa ou a indústria influenciam o salário?
- **Existe uma diferença salarial estatisticamente significativa entre vagas que são `Remote_Friendly` e as que não são?** (Foco do Teste de Hipótese)
""")
st.markdown("---")


# --- SEÇÃO 2: Medidas Centrais e Análise Descritiva ---
st.header("2. Análise Descritiva e Medidas Centrais")
st.write("Nesta seção, focamos na análise da variável quantitativa principal: `Salary_USD`.")

# Garante que a coluna de salário existe antes de tentar fazer os cálculos
if "Salary_USD" in df.columns:
    salarios = df["Salary_USD"].dropna()

    # Layout em colunas para melhor organização
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Estatísticas Descritivas do Salário")
        st.write(f"**Média Salarial:** ${salarios.mean():,.2f} USD")
        st.write(f"**Mediana Salarial:** ${salarios.median():,.2f} USD")
        st.write(f"**Desvio Padrão:** ${salarios.std():,.2f} USD")
        st.write(f"**Salário Mínimo:** ${salarios.min():,.2f} USD")
        st.write(f"**Salário Máximo:** ${salarios.max():,.2f} USD")

    with col2:
        st.subheader("Distribuição dos Salários (USD)")
        # Criação do gráfico de histograma
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(salarios, kde=True, ax=ax, bins=30, color='skyblue')
        ax.set_title("Histograma da Distribuição Salarial no Mercado de IA")
        ax.set_xlabel("Salário (USD)")
        ax.set_ylabel("Frequência")
        st.pyplot(fig)
else:
    st.warning("A coluna 'Salary_USD' não foi encontrada no dataset.")
st.markdown("---")

# --- SEÇÃO 3: Análises Adicionais e Respostas a Perguntas Específicas ---
st.header("3. Análises Adicionais e Respostas a Perguntas Específicas")

# Pergunta 1: Quais cargos estão mais associados ao crescimento de emprego?
st.subheader("Quais cargos estão mais associados ao crescimento de emprego?")
cargos_crescimento = df[df['Job_Growth_Projection'] == 'Growth']['Job_Title'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x=cargos_crescimento.values, y=cargos_crescimento.index, palette='viridis', ax=ax)
ax.set_title('Top 10 Cargos com Projeção de Crescimento')
ax.set_xlabel('Número de Vagas')
ax.set_ylabel('Cargo')
st.pyplot(fig)

# Pergunta 2 e 7: Como o salário varia e quais localizações oferecem os maiores salários?
st.subheader("Quais localizações oferecem os maiores salários para cargos de IA?")
maiores_salarios_local = df.groupby('Location')['Salary_USD'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x=maiores_salarios_local.values, y=maiores_salarios_local.index, palette='plasma', ax=ax)
ax.set_title('Top 10 Localizações por Salário Médio em IA')
ax.set_xlabel('Salário Médio (USD)')
ax.set_ylabel('Localização')
st.pyplot(fig)

# Pergunta 3: Correlação entre nível de adoção de IA e projeção de crescimento
st.subheader("Existe correlação entre o nível de adoção de IA e a projeção de crescimento?")
crosstab_ia_growth = pd.crosstab(df['AI_Adoption_Level'], df['Job_Growth_Projection'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(crosstab_ia_growth, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
ax.set_title('Adoção de IA vs. Projeção de Crescimento de Empregos')
ax.set_xlabel('Projeção de Crescimento')
ax.set_ylabel('Nível de Adoção de IA')
st.pyplot(fig)

# Pergunta 4: Habilidades mais comuns em empregos com alta projeção de crescimento
st.subheader("Quais habilidades são mais comuns em empregos com alta projeção de crescimento?")
habilidades_crescimento = df[df['Job_Growth_Projection'] == 'Growth']['Required_Skills'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x=habilidades_crescimento.values, y=habilidades_crescimento.index, palette='rocket', ax=ax)
ax.set_title('Top 10 Habilidades em Vagas com Projeção de Crescimento')
ax.set_xlabel('Número de Vagas')
ax.set_ylabel('Habilidade')
st.pyplot(fig)

# Pergunta 6: Como o tamanho da empresa influencia o salário e a projeção de crescimento?
st.subheader("Como o tamanho da empresa influencia o salário?")
fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(x='Company_Size', y='Salary_USD', data=df, palette='muted', ax=ax)
ax.set_title('Distribuição Salarial por Tamanho da Empresa')
ax.set_xlabel('Tamanho da Empresa')
ax.set_ylabel('Salário (USD)')
st.pyplot(fig)

# Pergunta 8: Quais cargos têm maior risco de automação e como isso afeta o salário?
st.subheader("Como o risco de automação afeta o salário?")
fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(x='Automation_Risk', y='Salary_USD', data=df, palette='crest', ax=ax)
ax.set_title('Distribuição Salarial por Nível de Risco de Automação')
ax.set_xlabel('Risco de Automação')
ax.set_ylabel('Salário (USD)')
st.pyplot(fig)
st.markdown("---")

# --- SEÇÃO 4: Testes de Hipótese e Intervalos de Confiança ---
st.header("4. Testes de Hipótese e Intervalos de Confiança")

# Explicação simples sobre Testes de Hipótese
st.info("""
**O que é um Teste de Hipótese?**

É uma forma de usar dados para tomar uma decisão. Partimos de uma **Hipótese Nula (H₀)**, que diz que "não há diferença" (ex: as médias são iguais). O objetivo é ver se temos evidências fortes o suficiente para rejeitar essa ideia.

**Como decidimos?** Olhamos para o **p-valor**:
- **Se p-valor < 0.05:** É um resultado "raro". Rejeitamos a H₀ e concluímos que existe uma diferença significativa.
- **Se p-valor >= 0.05:** É um resultado "comum". Não temos evidências para rejeitar a H₀.

O **Intervalo de Confiança (IC)** complementa o teste, nos dando uma faixa de valores prováveis para a média real.
""")

# Função para calcular o intervalo de confiança
def get_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

# Teste 1: Salário vs. Trabalho Remoto
st.subheader("Teste 1: Salário por Modalidade de Trabalho (Remoto vs. Não Remoto)")
st.write("Este teste verifica se a diferença observada na média salarial entre vagas remotas e não remotas é estatisticamente significativa ou apenas uma coincidência.")

if "Salary_USD" in df.columns and "Remote_Friendly" in df.columns:
    salarios_remoto = df[df['Remote_Friendly'] == 'Yes']['Salary_USD'].dropna()
    salarios_nao_remoto = df[df['Remote_Friendly'] == 'No']['Salary_USD'].dropna()

    col_teste1, col_visual1 = st.columns(2)
    with col_teste1:
        st.markdown("**Hipótese Nula (H₀):** A média salarial das vagas remotas é IGUAL à média salarial das vagas não remotas.")
        st.markdown("**Hipótese Alternativa (H₁):** A média salarial das vagas remotas é DIFERENTE da média salarial das vagas não remotas.")

        if len(salarios_remoto) > 1 and len(salarios_nao_remoto) > 1:
            t_stat, p_valor = stats.ttest_ind(salarios_remoto, salarios_nao_remoto, equal_var=False)
            ic_remoto = get_confidence_interval(salarios_remoto)
            ic_nao_remoto = get_confidence_interval(salarios_nao_remoto)
            
            st.write(f"**P-valor:** `{p_valor:.4f}`")
            if p_valor < 0.05:
                st.success("**Conclusão:** Rejeitamos H₀. Há uma diferença estatisticamente significativa.")
            else:
                st.info("**Conclusão:** Não rejeitamos H₀. Não há diferença estatisticamente significativa.")
            
            st.markdown("**Intervalos de Confiança (95%) para a Média Salarial:**")
            st.write(f"- **Remoto:** Entre ${ic_remoto[0]:,.2f} e ${ic_remoto[1]:,.2f}")
            st.write(f"- **Não Remoto:** Entre ${ic_nao_remoto[0]:,.2f} e ${ic_nao_remoto[1]:,.2f}")

        else:
            st.warning("Dados insuficientes para o teste.")
    with col_visual1:
        fig, ax = plt.subplots()
        sns.boxplot(x='Remote_Friendly', y='Salary_USD', data=df, palette="coolwarm", ax=ax)
        ax.set_title("Comparação de Salários: Remoto vs. Não Remoto")
        st.pyplot(fig)
else:
    st.warning("Colunas 'Salary_USD' ou 'Remote_Friendly' não encontradas.")

st.markdown("---")

# Teste 2: Salário vs. Tamanho da Empresa
st.subheader("Teste 2: Salário por Tamanho da Empresa")
st.write("Este teste nos ajuda a determinar se o tamanho da empresa (Pequena, Média, Grande) tem um impacto real na média salarial, comparando os três grupos de uma só vez.")

if "Salary_USD" in df.columns and "Company_Size" in df.columns:
    salario_pequena = df[df['Company_Size'] == 'Small']['Salary_USD'].dropna()
    salario_media = df[df['Company_Size'] == 'Medium']['Salary_USD'].dropna()
    salario_grande = df[df['Company_Size'] == 'Large']['Salary_USD'].dropna()

    col_teste2, col_visual2 = st.columns(2)
    with col_teste2:
        st.markdown("**Hipótese Nula (H₀):** As médias salariais são IGUAIS para todos os tamanhos de empresa.")
        st.markdown("**Hipótese Alternativa (H₁):** Pelo menos uma das médias salariais é DIFERENTE das outras.")
        
        if len(salario_pequena) > 1 and len(salario_media) > 1 and len(salario_grande) > 1:
            f_stat, p_valor_anova = stats.f_oneway(salario_pequena, salario_media, salario_grande)
            ic_pequena = get_confidence_interval(salario_pequena)
            ic_media = get_confidence_interval(salario_media)
            ic_grande = get_confidence_interval(salario_grande)

            st.write(f"**P-valor:** `{p_valor_anova:.4f}`")
            if p_valor_anova < 0.05:
                st.success("**Conclusão:** Rejeitamos H₀. Há uma diferença estatisticamente significativa nos salários entre os diferentes tamanhos de empresa.")
            else:
                st.info("**Conclusão:** Não rejeitamos H₀. Não há diferença estatisticamente significativa.")

            st.markdown("**Intervalos de Confiança (95%) para a Média Salarial:**")
            st.write(f"- **Pequena:** Entre ${ic_pequena[0]:,.2f} e ${ic_pequena[1]:,.2f}")
            st.write(f"- **Média:** Entre ${ic_media[0]:,.2f} e ${ic_media[1]:,.2f}")
            st.write(f"- **Grande:** Entre ${ic_grande[0]:,.2f} e ${ic_grande[1]:,.2f}")
        else:
            st.warning("Dados insuficientes para o teste.")
    with col_visual2:
        # Reutiliza o gráfico da seção anterior para consistência visual
        fig, ax = plt.subplots()
        sns.boxplot(x='Company_Size', y='Salary_USD', data=df, palette='muted', ax=ax)
        ax.set_title('Distribuição Salarial por Tamanho da Empresa')
        st.pyplot(fig)
else:
    st.warning("Colunas 'Salary_USD' ou 'Company_Size' não encontradas.")

