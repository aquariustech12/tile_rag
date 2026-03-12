import pickle
fn='memoria_vama.pkl'
to_delete = ["+5216674313084","5216444475422"]  # ajusta la lista según necesites
try:
    with open(fn,'rb') as f:
        d = pickle.load(f)
except Exception as e:
    print("No pude leer el pkl:", e); d = {}

removed = []
for t in to_delete:
    if t in d:
        del d[t]
        removed.append(t)

with open(fn,'wb') as f:
    pickle.dump(d,f)

print("Eliminados:", removed)
print("Usuarios restantes:", len(d))
