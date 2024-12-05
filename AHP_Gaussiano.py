import streamlit as st
import pandas as pd
import numpy as np

# Função para calcular o AHP Gaussiano e selecionar os melhores jogadores
def MelhoresEscolhas(df, nome, positivo, negativo):

    atletas = df[nome]

    df.drop(columns=[nome, 'Equipe'], inplace=True)

    # Verificar se as colunas existem

    for coluna in positivo + negativo:
        if coluna not in df.columns:
            raise ValueError(f'A coluna {coluna} não existe no DataFrame')

    # Seleciona as colunas positivas e negativas
    positivos = df[positivo]
    negativos = df[negativo]

    # Função para AHP Positivo
    def ahp_positivos(tabela):
        table = []
        for i in tabela.columns:
            total = tabela[i].sum()  # Calcula o total da coluna
            a = np.where(total == 0, 1e-10, tabela[i] / total)  # Normalização com proteção contra divisão por zero
            table.append(a)
        table = pd.DataFrame(table).T
        table.columns = tabela.columns
        return table

    positivos = ahp_positivos(positivos)

    # Função para AHP Negativo
    def numeros_negativos(tabela):
        table = []
        for i in tabela.columns:
            a = np.where(tabela[i] == 0, 1e-10, 1 / tabela[i])  # Inversão com proteção contra divisão por zero
            table.append(a)
        table = pd.DataFrame(table).T
        tab_final = []
        for i in table.columns:
            total = table[i].sum()
            b = np.where(total == 0, 1e-10, table[i] / total)  # Normalização pelo total da coluna
            tab_final.append(b)
        tab_final = pd.DataFrame(tab_final).T
        tab_final.columns = tabela.columns
        return tab_final

    negativos = numeros_negativos(negativos)
    tabela_ahp = pd.concat([positivos, negativos], axis=1)

    # Calculando as médias e os desvios
    medias = pd.DataFrame(tabela_ahp.mean(), columns=['media'])
    desvio = pd.DataFrame(tabela_ahp.std(), columns=['desvio'])
    fator_ahp = pd.concat([medias, desvio], axis=1)
    fator_ahp['desvio'] = fator_ahp['desvio'].fillna(np.mean(fator_ahp['desvio']))
    fator_ahp['desvio/media'] = fator_ahp['desvio'] / fator_ahp['media']
    fator_ahp['Fator'] = fator_ahp['desvio/media'] / sum(fator_ahp['desvio/media'])

    fator = pd.DataFrame(fator_ahp['Fator']).T
    colunas_para_calculo = fator.columns

    # Função para a matriz de decisão
    def matriz_de_decisao(tabela, fator):
        table = []
        for i in colunas_para_calculo:
            a = tabela[i] * fator[i][0]
            table.append(a)
        table = pd.DataFrame(table).T
        return table

    resultado_ahp = matriz_de_decisao(tabela_ahp, fator)
    soma = resultado_ahp.sum(axis=1)
    soma = pd.DataFrame(soma, columns=['Resultado'])

    # Redefinir o índice após a soma
    soma = soma.reset_index(drop=True)

    # Mesclar com a coluna de atletas
    melhores_escolhas = pd.concat([soma, atletas[[nome]].reset_index(drop=True)], axis=1)

    # Ordenar os resultados
    melhores_escolhas = melhores_escolhas.sort_values(by='Resultado', ascending=False).reset_index(drop=True)

    melhores_escolhas.rename(columns={nome: 'Jogador'}, inplace=True)

    return melhores_escolhas.iloc[0]


