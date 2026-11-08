# Detyra: Parashikimi i lodhjes me Linear Regression
# Tema: Lidhja mes oreve te gjumit, perdorimit te telefonit dhe lodhjes

from sklearn.linear_model import LinearRegression
import numpy as np

# HAPI 1: Te dhenat e mbledhura per 7 dite
ore_gjumi = [8,  7,  6,  9,  5,  7,  6]
ore_telefon = [2,  3,  5,  1,  6,  4,  7]
lodhja = [2,  3,  5,  1,  8,  4,  8] 

# Modeli merr dy te dhena hyrse: gjumin dhe kohen ne telefon
X = np.array(list(zip(ore_gjumi, ore_telefon)))
y = np.array(lodhja)

# HAPI 2: Krijimi dhe trajnimi i modelit
model = LinearRegression()
model.fit(X, y)

print("Modeli u trajnua me sukses!")
print("Koeficientet:", model.coef_)
print("Prerja (intercept):", model.intercept_)

# HAPI 3: Parashikimet per dy rastet e kerkuara
rastet = np.array([
    [8, 2],  # 8 ore gjume, 2 ore telefon
    [5, 6]   # 5 ore gjume, 6 ore telefon
])

parashikimet = model.predict(rastet)

print("\\nPARASHIKIMET:")
print(f"8 ore gjume dhe 2 ore telefon -> lodhja: {parashikimet[0]:.2f}/10")
print(f"5 ore gjume dhe 6 ore telefon -> lodhja: {parashikimet[1]:.2f}/10")

# HAPI 5: Eksperimentet
# Rast i pazakonte: vetem 2 ore gjume.
rasti_i_pazakonte = np.array([[2, 2]])
rezultati_pazakonte = model.predict(rasti_i_pazakonte)[0]
print(f"2 ore gjume dhe 2 ore telefon -> lodhja: {rezultati_pazakonte:.2f}/10")

# Prove me 20 ore gjume. Kjo eshte jashte intervalit te zakonshem te dhenave
rasti_20_ore = np.array([[20, 2]])
rezultati_20_ore = model.predict(rasti_20_ore)[0]
print(f"20 ore gjume dhe 2 ore telefon -> lodhja: {rezultati_20_ore:.2f}/10")
print("Kujdes: ky parashikim nuk eshte i besueshem, sepse 20 ore gjume")
print("eshte jashte intervalit te te dhenave me te cilat u trajnua modeli.")
