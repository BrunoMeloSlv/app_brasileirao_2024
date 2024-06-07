# Selecione o folder
# .\footstats\Scripts\activate
# streamlit run footstats_st.py

import streamlit as st
import pandas as pd #pandas==2.0.2
import plotly.express as px #plotly==5.15.0
import openpyxl as op
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import statsmodels.api as sm # biblioteca de modelagem estatística
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.formula.api as smf # estimação de modelos
import requests
import plotly.graph_objs as go



# Configuração da página
st.set_page_config(layout='wide')

wb = op.load_workbook('dados_multinomial.xlsx')
ws = wb['Sheet1']
dados_multinomial = pd.DataFrame(ws.values)
dados_multinomial.columns = dados_multinomial.iloc[0]
dados_multinomial = dados_multinomial[1:]


### Historico Brasileiraõ
hist = pd.read_csv('brasileirao.csv')

reg_hist = hist.copy()

reg_hist['pontos'] = reg_hist['points']/reg_hist['played']
reg_hist['vitorias'] = reg_hist['won']/reg_hist['played']
reg_hist['empates'] = reg_hist['draw']/reg_hist['played']
reg_hist['derrotas'] = reg_hist['loss']/reg_hist['played']
reg_hist['gols_pro'] = reg_hist['goals_for']/reg_hist['played']
reg_hist['gols_contra'] = reg_hist['goals_against']/reg_hist['played']
reg_hist.drop(columns=['played','won','draw','loss','goals_for','goals_against','goals_diff'], inplace= True)

hist.rename(
    columns={
        'place':'Posição',
        'acronym':'Sigla',
        'team':'Time',
        'points':'Pontos',
        'played':'Jogos',
        'won':'Vitórias',
        'draw':'Empates',
        'loss':'Derrotas',
        'goals_for':'Gols Pró',
        'goals_against':'Gols Contra',
        'goals_diff':'Saldo de Gols'
    }, inplace=True
)


## escudos

escudo = pd.read_excel('Escudos.xlsx')


##  footstats
wb1 = op.load_workbook('footstats.xlsx')
ws1 = wb1['Sheet1']
dados_footstats = pd.DataFrame(ws1.values)
dados_footstats.columns = dados_footstats.iloc[0]
dados_footstats = dados_footstats[1:]

colunas = dados_footstats.copy()
colunas.drop(columns=['Equipe','Jogador'], inplace= True)
# Loop através de todas as colunas do DataFrame
for column in colunas.columns:
    # Tente converter os valores da coluna para numéricos
    try:
        dados_footstats[column] = pd.to_numeric(dados_footstats[column])
    # Se ocorrer um erro, imprima uma mensagem de aviso
    except Exception as e:
        print(f"Erro ao converter a coluna {column}: {e}")

tabela = pd.read_html('https://fbref.com/en/comps/24/Serie-A-Stats')
tabela_brasileirao = tabela[0]

tabela_brasileirao.rename(
    columns={
        'Rk':'Posição',
        'Squad':'Time',
        'MP':'Jogos',
        'W':'Vitórias',
        'D':'Empates',
        'L':'Derrotas',
        'GF':'Gols Pró',
        'GA':'Gols Contra',
        'GD':'Saldo de Gols',
        'Pts':'Pontos',
        'Pts/MP':'Pontos por partida'
    }, inplace=True
)

tabela_brasileirao = tabela_brasileirao.iloc[:, 0:14]

colunas = list(tabela_brasileirao.columns)
colunas.insert(2, colunas.pop(colunas.index('Pontos')))
tabela_brasileirao = tabela_brasileirao[colunas]

tabela_brasileirao['Aproveitamento'] = ((tabela_brasileirao['Pontos'] / (tabela_brasileirao['Jogos'] * 3)) * 100).round(2)



col = ['mandante', 'visitante', 'vencedor', 'Temporada',0,1,2,]
df_prev = dados_multinomial[col]
#col_prob = ['mandante', 'visitante',0,1,2,'Temporada']
#df_prob = dados_multinomial[col_prob]

#### Valor de mercado 
# Lista de tuplas contendo (nome do clube, URL) dos times da Série A do Brasileirão 2024 no Transfermarkt
transfermarket = pd.read_excel('dados_transfermarkt_serie_a_2024.xlsx')


def get_exchange_rate(api_key, base_currency='EUR', target_currency='BRL'):
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200 or 'error' in data:
        print(f"Erro ao obter a taxa de câmbio: {data.get('error', 'Unknown error')}")
        return None
    exchange_rate = data['conversion_rates'][target_currency]
    return exchange_rate

def convert_currency(amount, exchange_rate):
    return amount * exchange_rate

# Substitua pelo seu API Key
#api_key = "a329da6b8195b6754616929a"

# Obter a taxa de câmbio EUR/BRL
#exchange_rate = get_exchange_rate(api_key)

#if exchange_rate:
    # Exemplo de conversão
#    euros = 1  # valor em euros
#    reais = convert_currency(euros, exchange_rate)
    
reais = 5.78

# Supondo que a coluna 'Valor de Mercado' contenha os valores em formato de string com o símbolo do euro
transfermarket['Valor de Mercado'] = transfermarket['Valor de Mercado'].str.replace('€', '').str.replace('mi.', '0000').str.replace('K', '0').str.replace(',', '').str.replace(' ', '')
transfermarket['Valor de Mercado'] = pd.to_numeric(transfermarket['Valor de Mercado'], errors='coerce')

# Agora, podemos multiplicar os valores da coluna 'Valor de Mercado' pelo valor em reais
transfermarket['Valor em Reais'] = transfermarket['Valor de Mercado'] * reais



# Obter a taxa de câmbio EUR/BRL
#exchange_rate = get_exchange_rate(api_key)

transfermarket.rename( columns={'Clube':'Equipe','Nome do Jogador':'Jogador'}, inplace=True)

# Chaves de relacionamento
chaves = ['Equipe', 'Jogador']

# Junção dos DataFrames usando as chaves de relacionamento
dados_footstats = pd.merge(dados_footstats, transfermarket, on=chaves)

# Função para selecionar o melhor jogador por posição
def selecionar_melhor_jogador(df, posicao, criterios):
    # Verificar se a coluna 'Posição' existe no DataFrame
    if 'Posição' not in df.columns:
        print("Aviso: A coluna 'Posição' não está presente no DataFrame. Não é possível selecionar os melhores jogadores.")
        return pd.DataFrame()  # Retorna um DataFrame vazio
    
    # Filtrar jogadores pela posição
    jogadores_posicao = df[df['Posição'] == posicao]
    
    # Verificar se os critérios existem no DataFrame
    for criterio in criterios:
        if criterio not in df.columns:
            print(f"Aviso: A coluna '{criterio}' não está presente no DataFrame. Não é possível calcular a pontuação.")
            return pd.DataFrame()  # Retorna um DataFrame vazio
    
    # Verificar se há jogadores disponíveis para essa posição
    if jogadores_posicao.empty:
        print(f"Aviso: Não há jogadores disponíveis para a posição '{posicao}'.")
        return pd.DataFrame()  # Retorna um DataFrame vazio
    
    # Calcular a pontuação
    jogadores_posicao['Pontuação'] = jogadores_posicao[criterios].sum(axis=1)
    
    # Selecionar os dois jogadores com a maior pontuação (para zagueiros e meias)
    if posicao in ['Zagueiro', 'Meia Ofensivo']:
        melhores_jogadores = jogadores_posicao.nlargest(2, 'Pontuação')
    else:
        melhores_jogadores = jogadores_posicao.nlargest(1, 'Pontuação')
    
    return melhores_jogadores

# Seleção do campeonato
selecao = pd.DataFrame()

# Defina os critérios de pontuação por posição
criterios_por_posicao = {
    'Centroavante': ['Gols', 'Assistência finalização', 'Finalização certa', 'Assistência gol', 'Dribles'],
    'Ponta Esquerda': ['Gols', 'Assistência finalização', 'Finalização certa', 'Assistência gol', 'Lançamento certo', 'Cruzamento certo'],
    'Ponta Direita': ['Gols', 'Assistência finalização', 'Finalização certa', 'Assistência gol', 'Lançamento certo', 'Cruzamento certo'],
    'Meia Ofensivo': ['Gols', 'Assistência finalização', 'Passe certo', 'Finalização certa', 'Assistência gol', 'Lançamento certo'],
    'Meio Central': ['Assistência finalização', 'Passe certo', 'Assistência gol', 'Interceptação certa', 'Finalização certa'],
    'Volante': ['Interceptação certa', 'Passe certo', 'Virada de jogo certa'],
    'Lateral Esq.': ['Interceptação certa', 'Assistência finalização', 'Passe certo', 'Finalização certa', 'Assistência gol', 'Lançamento certo', 'Virada de jogo certa'],
    'Lateral Dir.': ['Interceptação certa', 'Passe certo', 'Virada de jogo certa'],
    'Zagueiro': ['Interceptação certa', 'Assistência finalização', 'Passe certo', 'Finalização certa', 'Assistência gol', 'Lançamento certo', 'Virada de jogo certa'],
    'Goleiro': ['Rebatida', 'Defesa', 'Passe certo', 'Defesa difícil']  
}

for posicao, criterios in criterios_por_posicao.items():
    try:
        melhor_jogador = selecionar_melhor_jogador(dados_footstats, posicao, criterios)
        if not melhor_jogador.empty:
            selecao = pd.concat([selecao, melhor_jogador], ignore_index=True)
    except KeyError as e:
        print(f"Erro: {e}")

# Remova a coluna de pontuação antes de salvar
if 'Pontuação' in selecao.columns:
    selecao = selecao.drop(columns=['Pontuação'])


### Regressão



modelo = sm.OLS.from_formula('points ~ pontos + vitorias + empates + derrotas + gols_pro + gols_contra',
                         data=reg_hist).fit()

reg_hist['pontosfit'] = modelo.fittedvalues

reg_tabela_brasileirao = tabela_brasileirao.copy()

reg_tabela_brasileirao['pontos'] = reg_tabela_brasileirao['Pontos']/reg_tabela_brasileirao['Jogos']
reg_tabela_brasileirao['vitorias'] = reg_tabela_brasileirao['Vitórias']/reg_tabela_brasileirao['Jogos']
reg_tabela_brasileirao['empates'] = reg_tabela_brasileirao['Empates']/reg_tabela_brasileirao['Jogos']
reg_tabela_brasileirao['derrotas'] = reg_tabela_brasileirao['Derrotas']/reg_tabela_brasileirao['Jogos']
reg_tabela_brasileirao['gols_pro'] = reg_tabela_brasileirao['Gols Pró']/reg_tabela_brasileirao['Jogos']
reg_tabela_brasileirao['gols_contra'] = reg_tabela_brasileirao['Gols Contra']/reg_tabela_brasileirao['Jogos']
tabelabr = reg_tabela_brasileirao.copy()
tabelabr.drop(columns=['Pontos','Time','Posição','Jogos','Vitórias','Empates','Derrotas','Gols Pró','Gols Contra','xG','xGA','xGD','Pontos por partida','Saldo de Gols'], inplace= True)
predicoes_treino = modelo.predict(tabelabr)

tabela_brasileirao['Previsão'] = predicoes_treino
def ajustar_previsoes(valor):
    if valor < 0:
        return 0
    else:
        return valor

# Aplicando a função à coluna 'previsoes'
tabela_brasileirao['Previsão'] = tabela_brasileirao['Previsão'].apply(ajustar_previsoes)

### AHP

def top10_(jogador, clube):
    df = dados_footstats

    df_cluster2 = dados_footstats.copy()
    df_cluster2 = df_cluster2[df_cluster2['Jogos'] != 0]

    df_cluster_a_padronizar = df_cluster2.copy()
    df_cluster_a_padronizar.drop(columns=['Equipe', 'Jogador','Posição','Idade'], inplace=True)

    # Converter nomes das colunas para strings
    df_cluster_a_padronizar.columns = df_cluster_a_padronizar.columns.astype(str)

    sc_X = StandardScaler(with_mean=True, with_std=True)
    df_cluster_padronizado = sc_X.fit_transform(df_cluster_a_padronizar)

    # Converter para o formato de tabela - StandardScaler 
    df_cluster_padronizado = pd.DataFrame(data=df_cluster_padronizado, columns=df_cluster_a_padronizar.columns)
    
    # Preencher valores NaN com zero
    df_cluster_padronizado.fillna(0, inplace=True)
    
    pca = PCA(n_components=4, random_state=1224)
    df_cluster_pca_atacante = pca.fit_transform(df_cluster_padronizado)
    projection = pd.DataFrame(data=df_cluster_pca_atacante)
    kmeans_pca = KMeans(n_clusters=13, verbose=True, random_state=1224)
    kmeans_pca.fit(projection)
    projection['cluster_pca'] = kmeans_pca.predict(projection)
    projection['Jogador'] = df['Jogador']
    projection['Equipe'] = df['Equipe']
    projection['jogador/time'] = (projection['Jogador'] + '/' + projection['Equipe']) 

    nome_jogador = jogador + '/' + clube

    cluster = list(projection[projection['jogador/time'] == nome_jogador]['cluster_pca'])[0]

    jogador_recomendado = projection[projection['cluster_pca'] == cluster][[0, 1, 2, 3, 'Jogador', 'Equipe']]
    a_jogador = list(projection[projection['jogador/time'] == nome_jogador][0])[0]
    b_jogador = list(projection[projection['jogador/time'] == nome_jogador][1])[0]
    c_jogador = list(projection[projection['jogador/time'] == nome_jogador][2])[0]
    d_jogador = list(projection[projection['jogador/time'] == nome_jogador][3])[0]
    distancias = euclidean_distances(jogador_recomendado[[0, 1, 2, 3]], [[a_jogador, b_jogador, c_jogador, d_jogador]])
    jogador_recomendado['distancias'] = distancias
    tabela = jogador_recomendado.sort_values('distancias').head(11)
    dados_jogador = df_cluster2[df_cluster2['Jogador'].isin(tabela['Jogador']) & df_cluster2['Equipe'].isin(tabela['Equipe'])]
    recomendado = pd.merge(tabela, dados_jogador, on=['Equipe', 'Jogador'], how='inner')

    recomendado = recomendado[list(df_cluster2.columns)]
    recomendado.drop(0, axis=0, inplace=True)

    jogador_equipe = recomendado[['Jogador', 'Equipe']]

    positivos  = recomendado[['Jogos', 'Passe certo', 'Finalização certa', 'Cruzamento certo', 'Interceptação certa', 'Dribles', 'Virada de jogo certa', 'Gols', 'Assistência gol', 'Assistência finalização', 
                               'Rebatida', 'Falta recebida', 'Lançamento certo']]

    negativos = recomendado[['Falta cometida', 'Perda da posse de bola', 'Finalização errada', 'Cruzamento errado',
                             'Passe errado', 'Lançamento errado', 'Interceptação errada', 'Drible errado', 'Virada de jogo errada','Valor de Mercado']]

    colunas = recomendado[list(positivos.columns)]

    # Converter colunas de positivos para numéricas
    positivos = positivos.apply(pd.to_numeric, errors='coerce')

    def ahp_positivos(tabela):
        table = []
        for i in tabela.columns:
            a = tabela[i] / tabela[i].sum()
            table.append(a)
        table = pd.DataFrame(table).T
        return table

    positivos = ahp_positivos(positivos)

    # Converter colunas de negativos para numéricas
    negativos = negativos.apply(pd.to_numeric, errors='coerce')

    def numeros_negativos(tabela):
        table = []
        for i in tabela.columns:
            a = 1 / tabela[i]
            table.append(a)
        table = pd.DataFrame(table).T
        tab_final = []
        for i in table.columns:
            b = table[i] / table[i].sum()
            tab_final.append(b)
        tab_final = pd.DataFrame(tab_final).T
        return tab_final

    negativos = numeros_negativos(negativos)

    tabela_ahp = pd.concat([positivos, negativos], axis=1)

    medias = pd.DataFrame(tabela_ahp.mean(), columns=['media'])
    desvio = pd.DataFrame(tabela_ahp.std(), columns=['desvio'])
    fator_ahp = pd.concat([medias, desvio], axis=1)
    fator_ahp['desvio'] = fator_ahp['desvio'].fillna(np.mean(fator_ahp['desvio']))
    fator_ahp['desvio/media'] = fator_ahp['desvio'] / fator_ahp['media']
    fator_ahp['fator_gaussiano'] = fator_ahp['desvio/media'] / fator_ahp['desvio/media'].sum()
    fator = fator_ahp['fator_gaussiano']
    fator = pd.DataFrame(fator).T
    colunas_para_calculo = fator.columns

    def matriz_de_decisao(tabela, fator):
        table = []
        for i in colunas_para_calculo:
            a = tabela[i] * fator[i][0]
            table.append(a)
        table = pd.DataFrame(table).T
        return table

    resultado_ahp = matriz_de_decisao(tabela_ahp, fator)

    soma = resultado_ahp.sum(axis=1)
    soma = pd.DataFrame(soma, columns=['soma'])

    melhores_escolhas = pd.concat([soma, jogador_equipe], axis=1)
    melhores_escolhas = melhores_escolhas.sort_values(by='soma', ascending=False)
    melhores_escolhas.rename(columns={'soma':'AHP Gaussiano'}, inplace= True)
    melhores_escolhas = melhores_escolhas.reset_index(drop=True)
    # Chaves de relacionamento
    chaves = ['Equipe', 'Jogador']

  # Realizar a junção dos DataFrames usando as chaves de relacionamento
    melhores_escolhas = pd.merge(melhores_escolhas, dados_footstats[['Equipe', 'Jogador', 'Valor em Reais']], on=chaves)
    # Formatando o valor de mercado como dinheiro
    melhores_escolhas['Valor em Reais'] = melhores_escolhas['Valor em Reais'].apply(lambda x: f"R${x:,.2f}" if pd.notna(x) else 'N/A')

    
    return melhores_escolhas

# Função para aplicar estilos personalizados
def aplicar_estilos(df):
    # Definir estilos de cor de fundo e fonte para diferentes posições
    def estilos_linhas(s):
        if s.name < 4:  # Top 4
            return ['background-color: #023047; color: white'] * len(s)
        elif s.name == 4 or s.name == 5:  # 5º e 6º lugar
            return ['background-color: #dad7cd; color: black'] * len(s)
        elif s.name >= len(df) - 4:  # Últimos 4 lugares
            return ['background-color: #c1121f; color: white'] * len(s)
        else:
            return [''] * len(s)
    
    styled_df = df.style.apply(estilos_linhas, axis=1)
    return styled_df

# Função para filtrar dados por clube
def filtrar_dados_por_clube(dados_footstats, clube_selecionado):
    if clube_selecionado == "Todos":
        return dados_footstats
    else:
        return dados_footstats[dados_footstats['Equipe'] == clube_selecionado]
    
    

# Títulos e criação das abas
st.title('Dashboard Brasileirão 2024 ⚽')

# Criar abas
abas = st.tabs(["Classificação","Probabilidade", "Estatísticas dos Jogadores","Top 10 Jogadores Semelhantes","Histórico","Seleção do Campeonato","Relatório Clube"])

# Primeira aba: Classificação
with abas[0]:
    st.header("Classificação do Campeonato de Futebol")

    tabela_brasileirao['Pontos por partida'] = tabela_brasileirao['Pontos por partida'].astype(float).map("{:.2f}".format)
    tabela_brasileirao['xG'] = tabela_brasileirao['xG'].astype(float).map("{:.2f}".format)
    tabela_brasileirao['xGA'] = tabela_brasileirao['xGA'].astype(float).map("{:.2f}".format)
    tabela_brasileirao['xGD'] = tabela_brasileirao['xGD'].astype(float).map("{:.2f}".format)
    tabela_brasileirao['Aproveitamento'] = tabela_brasileirao['Aproveitamento'].astype(float).map("{:.2f}".format)
    tabela_brasileirao['Previsão'] = tabela_brasileirao['Previsão'].astype(float).map("{:.0f}".format)
    
    # Exibir tabela de classificação com estilos personalizados e sem índice
    st.markdown(aplicar_estilos(tabela_brasileirao).hide(axis='index').to_html(), unsafe_allow_html=True)

# Segunda aba: Probabilidade
with abas[1]:
    # Texto sobre a regressão multinomial
    with st.expander("Sobre a Regressão Multinomial"):
        st.write("""
        A regressão multinomial é uma extensão da regressão logística que permite modelar e prever variáveis dependentes categóricas com mais de duas categorias. Enquanto a regressão logística é adequada para prever variáveis dependentes binárias (duas categorias), a regressão multinomial pode lidar com variáveis dependentes com três ou mais categorias.

        Na regressão multinomial, o objetivo é prever a probabilidade de cada categoria da variável dependente, condicionada às variáveis independentes. Isso é feito através da estimativa de coeficientes para cada categoria da variável dependente em relação a uma categoria de referência (ou baseline).

        O modelo de regressão multinomial estima múltiplas equações logísticas simultaneamente, uma para cada categoria da variável dependente. Cada equação logística compara uma categoria específica com a categoria de referência, usando uma função logit para modelar a relação entre as variáveis independentes e a probabilidade de pertencer a uma categoria em particular.

        A interpretação dos coeficientes na regressão multinomial é semelhante à da regressão logística. Eles representam o efeito das variáveis independentes nas probabilidades relativas de pertencer a uma categoria específica em comparação com a categoria de referência.
        
        Este conjunto de dados utiliza exclusivamente informações do Campeonato Brasileiro de Futebol no período de 2003 a 2023 para gerar as probabilidades exibidas. 
        Por esse motivo, é importante ressaltar que as probabilidades calculadas aqui podem ser significativamente diferentes das odds oferecidas por casas de apostas. 
        As probabilidades geradas são baseadas em estatísticas históricas e modelos estatísticos aplicados a esses dados específicos, e não refletem necessariamente as probabilidades reais no momento da análise. 
        Portanto, ao tomar decisões de apostas, é recomendável considerar várias fontes de informações, incluindo análises recentes, notícias sobre os times e eventos atuais do futebol.
        
                 """)

    st.header("Histórico de Confrontos")

    # Selecionar o time mandante
    mandante = st.selectbox('Selecione o time mandante:', sorted(df_prev['mandante'].unique()))

    # Filtrar a lista de times visitantes para excluir o mandante selecionado
    times_visitantes = df_prev['visitante'].unique()
    times_visitantes = [time for time in times_visitantes if time != mandante]

    # Selecionar o time visitante
    visitante = st.selectbox('Selecione o time visitante:', sorted(times_visitantes))

    # Filtrar histórico de confrontos
    df_filtrado_historico = df_prev[(df_prev['mandante'] == mandante) & (df_prev['visitante'] == visitante)]
    df_filtrado_prob = df_filtrado_historico.copy()

    # Verificar se o DataFrame filtrado está vazio
    if df_filtrado_prob.empty:
        st.write('Os times selecionados nunca se enfrentaram.')
    else:
        # Filtrar a temporada máxima
        max_temp = df_filtrado_prob['Temporada'].max()
        df_filtrado_prob_max = df_filtrado_prob[df_filtrado_prob['Temporada'] == max_temp]

        # Layout de colunas para histórico e probabilidades
        col1, col2 = st.columns(2)

        # Exibir tabela de histórico filtrada e sem índice
        with col1:
            st.write('Histórico de Confrontos Filtrado:')
            st.markdown(df_filtrado_historico.drop(columns=[0,1,2]).style.hide(axis='index').to_html(), unsafe_allow_html=True)

        with col2:
            st.write('Valores de Probabilidade:')
            if not df_filtrado_prob_max.empty:
                empate = df_filtrado_prob_max[0].values[0] * 100
                vitoria_mandante = df_filtrado_prob_max[1].values[0] * 100
                vitoria_visitante = df_filtrado_prob_max[2].values[0] * 100

                st.write(f"Empate: {empate:.2f}%")
                st.write(f"Vitória do {mandante}: {vitoria_mandante:.2f}%")
                st.write(f"Vitória do {visitante}: {vitoria_visitante:.2f}%")
            else:
                st.write('Não há dados suficientes para mostrar os valores das colunas 0, 1 e 2.')

with abas[2]:
    st.header("Dados Jogadores")

    
    # Adiciona um seletor de clube
    clubes = ["Todos"] + dados_footstats['Equipe'].unique().tolist()
    clube_selecionado = st.selectbox("Selecione o clube",sorted(clubes))
    
    # Filtra os dados com base na seleção do clube
    df_filtrado = filtrar_dados_por_clube(dados_footstats, clube_selecionado)

    
    col1, col2 = st.columns(2)
    with col1:
    ### Gráficos

        # Plotar gráficos personalizados
        top_Jogos = df_filtrado.nlargest(5, 'Jogos')
        fig_Jogos = px.bar(top_Jogos, x='Jogador', y='Jogos', text_auto=True, title='Top Nº de Jogos')
        fig_Jogos.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Jogos.update_layout(yaxis_title='Jogos', xaxis_title='Jogadores')  # Adicionar rótulos

        top_Passe_certo = df_filtrado.nlargest(5, 'Passe certo')
        fig_Passe = px.bar(top_Passe_certo, x='Jogador', y='Passe certo', text_auto=True, title='Top Nº de Passes certos')
        fig_Passe.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Passe.update_layout(yaxis_title='Passes certos', xaxis_title='Jogadores')  # Adicionar rótulos

        top_Passe_errado = df_filtrado.nlargest(5, 'Passe errado')
        fig_passeerrado= px.bar(top_Passe_errado, x='Jogador', y='Passe errado', text_auto=True, title='Top Nº de Passes errados')
        fig_passeerrado.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_passeerrado.update_layout(yaxis_title='Passe errados', xaxis_title='Jogadores')  # Adicionar rótulos

        top_Finalizacao_certa = df_filtrado.nlargest(5, 'Finalização certa')
        fig_Finalizacao_certa= px.bar(top_Finalizacao_certa, x='Jogador', y='Finalização certa', text_auto=True, title='Top Nº de Finalizações certas')
        fig_Finalizacao_certa.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Finalizacao_certa.update_layout(yaxis_title='Finalização certa', xaxis_title='Jogadores')  # Adicionar rótulos

        top_Finalizacao_errada = df_filtrado.nlargest(5, 'Finalização errada')
        fig_Finalizacao_errada= px.bar(top_Finalizacao_errada, x='Jogador', y='Finalização errada', text_auto=True, title='Top Nº de Finalizações errados')
        fig_Finalizacao_errada.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Finalizacao_errada.update_layout(yaxis_title='Finalização errada', xaxis_title='Jogadores')  # Adicionar rótulos

        top_Cruzamento_certo = df_filtrado.nlargest(5, 'Cruzamento certo')
        fig_Cruzamento_certo = px.bar(top_Cruzamento_certo, x='Jogador', y='Cruzamento certo', text_auto=True, title='Top Nº de Cruzamentos certos')
        fig_Cruzamento_certo.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Cruzamento_certo.update_layout(yaxis_title='Cruzamento certo', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Cruzamentos errados
        top_Cruzamentos_errados = df_filtrado.nlargest(5, 'Cruzamento errado')
        fig_Cruzamentos_errados = px.bar(top_Cruzamentos_errados, x='Jogador', y='Cruzamento errado', text_auto=True, title='Top Nº de Cruzamentos errados')
        fig_Cruzamentos_errados.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Cruzamentos_errados.update_layout(yaxis_title='Cruzamento errado', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Dribles certos
        top_Drible_certo = df_filtrado.nlargest(5, 'Dribles')
        fig_Drible_certo = px.bar(top_Drible_certo, x='Jogador', y='Dribles', text_auto=True, title='Top Nº de Dribles Certos')
        fig_Drible_certo.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Drible_certo.update_layout(yaxis_title='Dribles', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Dribles errados
        top_Drible_errado = df_filtrado.nlargest(5, 'Drible errado')
        fig_Drible_errado = px.bar(top_Drible_errado, x='Jogador', y='Drible errado', text_auto=True, title='Top Nº de Dribles errados')
        fig_Drible_errado.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Drible_errado.update_layout(yaxis_title='Drible errado', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Viradas de jogo certas
        top_Virada_certa = df_filtrado.nlargest(5, 'Virada de jogo certa')
        fig_Virada_certa = px.bar(top_Virada_certa, x='Jogador', y='Virada de jogo certa', text_auto=True, title='Top Nº de Viradas de jogo certas')
        fig_Virada_certa.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Virada_certa.update_layout(yaxis_title='Viradas de jogo certas', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Viradas de jogo erradas
        top_Virada_errada = df_filtrado.nlargest(5, 'Virada de jogo errada')
        fig_Virada_errada = px.bar(top_Virada_errada, x='Jogador', y='Virada de jogo errada', text_auto=True, title='Top Nº de Viradas de jogo erradas')
        fig_Virada_errada.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Virada_errada.update_layout(yaxis_title='Virada de jogo errada', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Gols
        top_Gols = df_filtrado.nlargest(5, 'Gols')
        fig_Gols = px.bar(top_Gols, x='Jogador', y='Gols', text_auto=True, title='Top Nº de Gols')
        fig_Gols.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Gols.update_layout(yaxis_title='Gols', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Assistência gol
        top_Assistência_gol = df_filtrado.nlargest(5, 'Assistência gol')
        fig_Assistência_gol = px.bar(top_Assistência_gol, x='Jogador', y='Assistência gol', text_auto=True, title='Top Nº de Assistência gol')
        fig_Assistência_gol.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Assistência_gol.update_layout(yaxis_title='Assistência gol', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Assistência finalização
        top_Assistência_finalização = df_filtrado.nlargest(5, 'Assistência finalização')
        fig_Assistência_finalização = px.bar(top_Assistência_finalização, x='Jogador', y='Assistência finalização', text_auto=True, title='Top Nº de Assistência finalização')
        fig_Assistência_finalização.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Assistência_finalização.update_layout(yaxis_title='Assistência finalização', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Defesas
        top_Defesas = df_filtrado.nlargest(5, 'Defesa')
        fig_Defesas = px.bar(top_Defesas, x='Jogador', y='Defesa', text_auto=True, title='Top Nº de Defesas')
        fig_Defesas.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Defesas.update_layout(yaxis_title='Defesa', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Defesa difícil
        top_Defesa_difícil = df_filtrado.nlargest(5, 'Defesa difícil')
        fig_Defesa_difícil = px.bar(top_Defesa_difícil, x='Jogador', y='Defesa difícil', text_auto=True, title='Top Nº de Defesa difícil')
        fig_Defesa_difícil.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Defesa_difícil.update_layout(yaxis_title='Defesa difícil', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Faltas cometidas
        top_Falta_cometida = df_filtrado.nlargest(5, 'Falta cometida')
        fig_Falta_cometida = px.bar(top_Falta_cometida, x='Jogador', y='Falta cometida', text_auto=True, title='Top Nº de Faltas cometidas')
        fig_Falta_cometida.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Falta_cometida.update_layout(yaxis_title='Falta cometida', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Faltas recebidas
        top_Falta_recebida = df_filtrado.nlargest(5, 'Falta recebida')
        fig_Falta_recebida = px.bar(top_Falta_recebida, x='Jogador', y='Falta recebida', text_auto=True, title='Top Nº de Faltas recebidas')
        fig_Falta_recebida.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Falta_recebida.update_layout(yaxis_title='Falta recebida', xaxis_title='Jogadores')  # Adicionar rótulos

    
        # Para o gráfico de Cartões Vermelhos
        top_Cartão_vermelho = df_filtrado.nlargest(5, 'Cartão vermelho')
        fig_Cartão_vermelho = px.bar(top_Cartão_vermelho, x='Jogador', y='Cartão vermelho', text_auto=True, title='Top Nº de Cartões Vermelhos')
        fig_Cartão_vermelho.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Cartão_vermelho.update_layout(yaxis_title='Cartão vermelho', xaxis_title='Jogadores')  # Adicionar rótulos

        # Para o gráfico de Cartões Amarelos
        top_Cartão_amarelo = df_filtrado.nlargest(5, 'Cartão amarelo')
        fig_Cartão_amarelo = px.bar(top_Cartão_amarelo, x='Jogador', y='Cartão amarelo', text_auto=True, title='Top Nº de Cartões Amarelos')
        fig_Cartão_amarelo.update_traces(marker_color='#023047')  # Adicionar cor personalizada
        fig_Cartão_amarelo.update_layout(yaxis_title='Cartão amarelo', xaxis_title='Jogadores')  # Adicionar rótulos

        st.plotly_chart(fig_Jogos,use_container_width= True)
        st.plotly_chart(fig_passeerrado,use_container_width= True)
        st.plotly_chart(fig_Finalizacao_errada,use_container_width= True)
        st.plotly_chart(fig_Cruzamentos_errados,use_container_width= True)
        st.plotly_chart(fig_Drible_errado,use_container_width= True)
        st.plotly_chart(fig_Virada_errada,use_container_width= True)
        st.plotly_chart(fig_Assistência_gol,use_container_width= True)
        st.plotly_chart(fig_Defesas,use_container_width= True)
        st.plotly_chart(fig_Falta_cometida,use_container_width= True)
        st.plotly_chart(fig_Assistência_gol,use_container_width= True)
        st.plotly_chart(fig_Cartão_vermelho,use_container_width= True)

    with col2:
        st.plotly_chart(fig_Virada_certa,use_container_width= True)
        st.plotly_chart(fig_Passe,use_container_width= True)
        st.plotly_chart(fig_Finalizacao_certa,use_container_width= True)
        st.plotly_chart(fig_Cruzamento_certo,use_container_width= True)
        st.plotly_chart(fig_Drible_certo,use_container_width= True)
        st.plotly_chart(fig_Gols,use_container_width= True)
        st.plotly_chart(fig_Assistência_finalização,use_container_width= True)
        st.plotly_chart(fig_Defesa_difícil,use_container_width= True)
        st.plotly_chart(fig_Falta_recebida,use_container_width= True)
        st.plotly_chart(fig_Assistência_finalização,use_container_width= True)
        st.plotly_chart(fig_Cartão_amarelo,use_container_width= True)

with abas[3]:

    with st.expander("Sobre o AHP-Gaussiano"):
        st.write("""
        O AHP-Gaussiano é uma técnica de tomada de decisão multicritério que combina o método Analytic Hierarchy Process (AHP) com a distribuição gaussiana. O AHP é um método usado para lidar com problemas de decisão que envolvem múltiplos critérios e alternativas.

        O AHP-Gaussiano estende o AHP tradicional para lidar com incertezas e imprecisões nos julgamentos dos tomadores de decisão. Ele introduz uma função de distribuição gaussiana para modelar a incerteza associada às avaliações de julgamento. Isso permite que o método lide com situações em que os julgamentos dos especialistas podem não ser precisos ou podem variar.
        """)

    st.header("Top 10 Jogadores Similares")

    
    # Criar um filtro para o clube
    clube = st.selectbox("Selecione o Clube:", sorted(dados_footstats['Equipe'].unique()))

# Filtrar os dados pelo clube selecionado
    dados_clube = dados_footstats[dados_footstats['Equipe'] == clube]

# Agora, criar um seletor de jogador baseado no filtro do clube
    jogador = st.selectbox("Selecione o Jogador:", sorted(dados_clube['Jogador'].unique()))

    
    

    # Se o usuário selecionou um jogador e um clube, executa a função `top10_`
    if (jogador != '') & (clube != ''):
        resultado = top10_(jogador, clube)
    st.dataframe(resultado)
    

with abas[4]:
    # Texto sobre o conjunto de dados
    with st.expander("Sobre o Conjunto de Dados"):
        st.write("""
        Este conjunto de dados utiliza exclusivamente informações do Campeonato Brasileiro de Futebol no período de 2003 a 2023. 
        """)

   # Filtrar os dados onde 'Posição' é igual a 1
    times_campeoes = hist[hist['Posição'] == 1]

    # Contar a quantidade de vezes que cada time ficou na posição 1
    contagem_times = times_campeoes['Time'].value_counts().reset_index()
    contagem_times.columns = ['Time', 'Quantidade']

    # Ordenar os times pelo número de vezes que ficaram na posição 1
    contagem_times = contagem_times.sort_values(by='Quantidade', ascending=False)

    # Criar o gráfico com o Plotly
    fig = px.bar(contagem_times, x='Time', y='Quantidade', text='Quantidade', title='Times Campeões',
                labels={'Time': 'Time', 'Quantidade': 'Quantidade'})

    # Personalizar cores
    fig.update_traces(marker_color='#023047')

    # Adicionar rótulos
    fig.update_layout(yaxis_title='Quantidade', xaxis_title='Time')

    # Criar um filtro para temporada
    temp = st.selectbox("Selecione a Temporada:", hist['season'].unique())

    # Filtrar os dados pela temporada selecionada e remover a coluna 'season'
    hist_filtrado = hist[hist['season'] == temp].drop(columns=['season']).reset_index(drop=True)

    # Função para calcular o percentual
    def calcular_percentual(df):
        total_jogos = df['Vitórias'].sum() + df['Empates'].sum() + df['Derrotas'].sum()
        df['Percentual_vitórias'] = df['Vitórias'] / total_jogos
        df['Percentual_empates'] = df['Empates'] / total_jogos
        df['Percentual_derrotas'] = df['Derrotas'] / total_jogos
        return df

    # Exibir os gráficos
    

    # Exibir a tabela completa
    st.table(hist_filtrado)
    st.plotly_chart(fig)

    

# Primeira aba: Classificação
with abas[5]:
    # Separando a idade do ano de nascimento
    selecao['Ano de Nascimento'] = selecao['Idade'].apply(lambda x: x.split(' ')[0])
    selecao['Idade'] = selecao['Idade'].apply(lambda x: x.split(' ')[1].strip('()'))

    # Formatando o valor de mercado como dinheiro
    selecao['Valor em Reais'] = selecao['Valor em Reais'].apply(lambda x: f"R${x:,.2f}" if pd.notna(x) else 'N/A')

    # Reordenando as colunas
    selecao = selecao[['Equipe', 'Jogador', 'Jogos', 'Posição', 'Idade', 'Valor em Reais']]
    
    # Convertendo o DataFrame para HTML com estilo
    html_table = selecao.to_html(index=False, justify='center', border=0, classes='styled-table')

    # Definindo estilo CSS
    css = """
        <style>
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: 'Trebuchet MS', 'Lucida Grande', 'Lucida Sans Unicode', 'Lucida Sans', 'Arial', 'sans-serif';
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table thead tr {
            background-color: #023047;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th,
        .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #023047;
        }
        .styled-table tbody tr.active-row {
            font-weight: bold;
            color: #023047;
        }
        </style>
    """

    # Convertendo o dicionário para um DataFrame
    criterios_df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in criterios_por_posicao.items()])).T
    criterios_df = criterios_df.reset_index()
    criterios_df.columns = ['Posição', 'Critério 1', 'Critério 2', 'Critério 3', 'Critério 4', 'Critério 5', 'Critério 6', 'Critério 7']

    # Convertendo o DataFrame de critérios para HTML com estilo
    html_criterios = criterios_df.to_html(index=False, justify='center', border=0, classes='styled-table')


# Exibindo a tabela estilizada no Streamlit

    #st.header("Seleção do Campeonato de Futebol")

    # Renderizando a tabela estilizada no Streamlit
    st.markdown("<h2>Seleção do Campeonato de Futebol</h2>" + css + html_table, unsafe_allow_html=True)

    st.markdown("<h2>Critérios por Posição</h2>" + css + html_criterios, unsafe_allow_html=True)

with abas[6]:
    st.header("Relatório do Clube")

    # Dicionário de mapeamento para padronizar os nomes dos clubes
    mapeamento_clubes = {
        'flamengo': 'Flamengo',
        'flamengo': 'Flamengo',
        'Flamengo': 'Flamengo',
        'Flamengo': 'Flamengo',
        'bahia': 'Bahia',
        'bahia': 'Bahia',
        'bahia': 'Bahia',
        'Bahia': 'Bahia',
        'botafogo-rj': 'Botafogo',
        'botafogo-rj': 'Botafogo',
        'botafogo': 'Botafogo',
        'Botafogo': 'Botafogo',
        'sao paulo': 'São Paulo',
        'sao paulo': 'São Paulo',
        'sao-paulo': 'São Paulo',
        'SaoPaulo': 'São Paulo',
        'athletico-pr': 'Ath Paranaense',
        'athletico-pr': 'Ath Paranaense',
        'athletico-pr': 'Ath Paranaense',
        'Athletico-PR': 'Ath Paranaense',
        'bragantino': 'Red Bull Bragantino',
        'bragantino': 'Red Bull Bragantino',
        'bragantino': 'Red Bull Bragantino',
        'RBBragantino': 'Red Bull Bragantino',
        'palmeiras': 'Palmeiras',
        'palmeiras': 'Palmeiras',
        'palmeiras': 'Palmeiras',
        'Palmeiras': 'Palmeiras',
        'internacional': 'Internacional',
        'internacional': 'Internacional',
        'internacional': 'Internacional',
        'Internacional': 'Internacional',
        'cruzeiro': 'Cruzeiro',
        'cruzeiro': 'Cruzeiro',
        'cruzeiro': 'Cruzeiro',
        'Cruzeiro': 'Cruzeiro',
        'atletico-mg': 'Atlético Mineiro',
        'atletico-mg': 'Atlético Mineiro',
        'atletico-mg': 'Atlético Mineiro',
        'Atletico-MG': 'Atlético Mineiro',
        'fortaleza': 'Fortaleza',
        'fortaleza': 'Fortaleza',
        'fortaleza': 'Fortaleza',
        'Fortaleza': 'Fortaleza',
        'juventude': 'Juventude',
        'juventude': 'Juventude',
        'juventude': 'Juventude',
        'Juventude': 'Juventude',
        'gremio': 'Grêmio',
        'gremio': 'Grêmio',
        'gremio': 'Grêmio',
        'Gremio': 'Grêmio',
        'vasco': 'Vasco da Gama',
        'vasco': 'Vasco da Gama',
        'vasco': 'Vasco da Gama',
        'Vasco': 'Vasco da Gama',
        'fluminense': 'Fluminense',
        'fluminense': 'Fluminense',
        'fluminense': 'Fluminense',
        'Fluminense': 'Fluminense',
        'criciuma': 'Criciúma',
        'criciuma': 'Criciúma',
        'criciuma': 'Criciúma',
        'Criciuma': 'Criciúma',
        'corinthians': 'Corinthians',
        'corinthians': 'Corinthians',
        'corinthians': 'Corinthians',
        'Corinthians': 'Corinthians',
        'atletico-go': 'Atl Goianiense',
        'atletico-go': 'Atl Goianiense',
        'atletico-go': 'Atl Goianiense',
        'Atletico-GO': 'Atl Goianiense',
        'vitoria': 'Vitória',
        'vitoria': 'Vitória',
        'vitoria': 'Vitória',
        'Vitoria': 'Vitória',
        'cuiaba': 'Cuiabá',
        'cuiaba': 'Cuiabá',
        'cuiaba': 'Cuiabá',
        'Cuiaba': 'Cuiabá'
    }

    # Filtrar os dados pelo clube selecionado
    clube_selecionado = st.selectbox("Selecione o Clube:", sorted(hist['Time'].unique()))

    # Função para padronizar o nome do clube
    def padronizar_clube(nome):
        return mapeamento_clubes.get(nome.lower(), nome)

    tab_br24 = tabela_brasileirao.copy()
    footstats = dados_footstats.copy()

    # Padronizar os nomes dos clubes no DataFrame
    tab_br24['Time'] = tab_br24['Time'].apply(padronizar_clube)
    footstats['Equipe'] = footstats['Equipe'].apply(padronizar_clube)

    footstats.drop(columns=['Jogador'], inplace= True)

    time_passes = pd.DataFrame(footstats.groupby('Equipe')['Passe certo'].sum()).reset_index().sort_values('Passe certo', ascending=False)
    time_passes_errados = pd.DataFrame(footstats.groupby('Equipe')['Passe errado'].sum()).reset_index().sort_values('Passe errado', ascending=False)

    tab_br24 = tab_br24[tab_br24['Time'] == clube_selecionado]

    colunas = ['Passe certo','Passe errado','Finalização certa','Finalização errada','Cruzamento certo',       'Cruzamento errado',
              'Lançamento certo',       'Lançamento errado',
           'Interceptação certa',    'Interceptação errada',
                       'Dribles',           'Drible errado',
          'Virada de jogo certa',   'Virada de jogo errada',
                          'Gols',         'Assistência gol',
       'Assistência finalização',                  'Defesa',
                'Defesa difícil',                'Rebatida',
                'Falta cometida',          'Falta recebida',
                  'Impedimentos',       'Pênaltis sofridos',
            'Pênaltis cometidos',  'Perda da posse de bola',
                'Cartão amarelo',         'Cartão vermelho']

    # Converter as colunas para tipo numérico
    for coluna in colunas:
        footstats[coluna] = pd.to_numeric(footstats[coluna], errors='coerce')

    dados_agrupados = pd.DataFrame(footstats.groupby('Equipe')[colunas].sum()).reset_index().sort_values('Passe certo', ascending=False)
    dados_agrupados = dados_agrupados[dados_agrupados['Equipe'] == clube_selecionado]

    pares_colunas = [
    ['Passe certo','Passe errado'],
    ['Finalização certa','Finalização errada'],
    ['Cruzamento certo','Cruzamento errado'],
    ['Lançamento certo','Lançamento errado'],
    ['Interceptação certa','Interceptação errada'],
    ['Dribles','Drible errado'],
    ['Virada de jogo certa','Virada de jogo errada'],
    ['Gols','Assistência gol','Assistência finalização'],
    ['Defesa','Defesa difícil','Rebatida'],
    ['Falta cometida','Falta recebida'],
    ['Cartão amarelo','Cartão vermelho']
    ]


    # Filtrar os dados pelo clube selecionado
    dados_clube = hist[hist['Time'] == clube_selecionado]


    # Juntar as tabelas hist e escudo pela coluna 'sigla' e selecionar a coluna do link do escudo
    df_merged = pd.merge(dados_clube, escudo[['Sigla', escudo.columns[-1]]], on='Sigla')

    # Exibir o escudo do clube
    if not df_merged.empty:
        escudo_link = df_merged[escudo.columns[-1]].iloc[0]
        st.image(escudo_link, width=100, caption=clube_selecionado)
    
    # Exibir tabela de classificação com estilos personalizados e sem índice

    # Verificar se há dados filtrados para o clube selecionado
    if tab_br24[tab_br24['Time'] == clube_selecionado].shape[0] > 0:
        st.header("Campeonato atual")
        st.markdown(aplicar_estilos(tab_br24).hide(axis='index').to_html(), unsafe_allow_html=True)
    else:
        st.write("")

    
    if dados_agrupados[dados_agrupados['Equipe'] == clube_selecionado].shape[0] > 0:
        st.header("Estatísticas do Brasileirão 2024")


        # Organizar os gráficos em colunas duplas
        col1, col2 = st.columns(2)

       # Para cada par de colunas, criar e exibir o gráfico correspondente
        for par in pares_colunas:
            fig = px.bar(dados_agrupados, x='Equipe', y=par, title=', '.join(par), text_auto=True)
            if pares_colunas.index(par) % 2 == 0:
                with col1:
                    st.plotly_chart(fig)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

    st.header("Histórico")

    fig_posicao = go.Figure()

    fig_posicao.add_trace(go.Scatter(
        x=dados_clube['season'], 
        y=dados_clube['Posição'], 
        mode='lines+markers',
        line_shape='hv',
        name=f'Posição do {clube_selecionado}'
    ))

    # Inverter o eixo Y
    fig_posicao.update_yaxes(autorange="reversed")

    # Atualizar o layout do gráfico
    fig_posicao.update_layout(
        title=f'Posição do {clube_selecionado} ao Longo das Temporadas',
        xaxis_title='Temporada',
        yaxis_title='Posição',
    )

    # Calcular o número e percentual de vitórias, empates e derrotas
    dados_clube_agrupados = dados_clube.groupby('season').agg({'Vitórias':'sum', 'Empates':'sum', 'Derrotas':'sum'})
    dados_clube_agrupados = calcular_percentual(dados_clube_agrupados)

    # Gráfico de barra para mostrar o número de vitórias, empates e derrotas
    fig_resultados = px.bar(dados_clube_agrupados, x=dados_clube_agrupados.index, y=['Vitórias', 'Empates', 'Derrotas'], 
                            title=f'Número de Vitórias, Empates e Derrotas do {clube_selecionado} por Temporada')

    # Gráfico de linha para mostrar o percentual de vitórias, empates e derrotas
    fig_percentual = px.line(dados_clube_agrupados, x=dados_clube_agrupados.index, y=['Percentual_vitórias', 'Percentual_empates', 'Percentual_derrotas'], 
                            title=f'Percentual de Vitórias, Empates e Derrotas do {clube_selecionado} por Temporada')

    # Gráfico de barras para mostrar a quantidade de gols pró e gols contra
    fig_gols = px.bar(dados_clube, x='season', y=['Gols Pró', 'Gols Contra'], 
                    title=f'Quantidade de Gols Pró e Contra do {clube_selecionado} por Temporada')
    
    # Organizar os gráficos em duas colunas
    col1, col2 = st.columns(2)

    # Exibir os gráficos na primeira coluna
    col1.plotly_chart(fig_posicao)
    col1.plotly_chart(fig_resultados)

    # Exibir os gráficos na segunda coluna
    col2.plotly_chart(fig_percentual)
    col2.plotly_chart(fig_gols)
