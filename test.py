import pandas
from cellshift import CS

df = pandas.DataFrame({
  "id": [i for i in range(1,12+1)],
  "pob": [ "Armenia", "Barranquilla", "Bello", "Bogotá", 
           "Bucaramanga", "Barranquilla", "Cartagena", "Cúcuta", 
           "Ibagué", "Manizales", "Pereira", "Santa Marta", ]
})
ob = CS(df)
ob.data.show()
ob.add_syn_city_column("pob")
ob.data.show()
print()
ob.city_equivalences.show()