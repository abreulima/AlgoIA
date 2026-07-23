import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ============================
# Ler os dados
# ============================
dados = pd.read_csv("filmes.csv")

# ============================
# Converter texto em números
# ============================
encoder_sexo = LabelEncoder()
encoder_genero = LabelEncoder()
encoder_filme = LabelEncoder()

dados["sexo"] = encoder_sexo.fit_transform(dados["sexo"])
dados["genero_favorito"] = encoder_genero.fit_transform(dados["genero_favorito"])
dados["filme"] = encoder_filme.fit_transform(dados["filme"])

# ============================
# Entradas e saída
# ============================
X = dados[["idade", "sexo", "genero_favorito"]]
y = dados["filme"]

# ============================
# Criar e treinar a IA
# ============================
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X, y)

# ============================
# Receber dados do utilizador
# ============================
idade = int(input("Idade: "))
sexo = input("Sexo (M/F): ").upper()
genero = input("Género favorito: ")

# ============================
# Verificar se os dados existem
# ============================
if sexo not in encoder_sexo.classes_:
    print("Sexo inválido.")
    exit()

if genero not in encoder_genero.classes_:
    print("Género não encontrado na base de dados.")
    print("Opções disponíveis:")
    print(", ".join(encoder_genero.classes_))
    exit()

# Converter para números
novo_utilizador = [[
    idade,
    encoder_sexo.transform([sexo])[0],
    encoder_genero.transform([genero])[0]
]]

# ============================
# Fazer recomendação
# ============================
resultado = modelo.predict(novo_utilizador)

filme = encoder_filme.inverse_transform(resultado)

print("\n==============================")
print("Filme recomendado:")
print(filme[0])
print("==============================")
