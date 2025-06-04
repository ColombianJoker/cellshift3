import pandas
from cellshift import CS

df = pandas.DataFrame({
  "id": [i for i in range(1,12+1)],
  "persona": [ "Jorge Molano", "Felix Padilla", "John Mendoza", "Sebastián Perdomo", 
               "Alejandro Álvarez", "Enrique Caro", "Wilson Pérez", "Patricia Cifuentes", 
               "Milton García", "Ángela Cortés", "Marco Mesa", "Dary Castrillón", ]
})
ob = CS(df)
ob.data.show()
ob.add_syn_name_column("persona", "otro")
  .add_syn_name_column("persona", "tercero", max_uniques=0)
ob.data.show()
