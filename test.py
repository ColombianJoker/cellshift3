import pandas
from cellshift import CS

df = pandas.DataFrame({
  "id": [i for i in range(1,12+1)],
  "persona": [ "Ángela Cortés", "Felix Padilla", "John Mendoza",
               "Sebastián Perdomo", "Alejandro Álvarez", "Enrique Caro",
               "Wilson Pérez", "Patricia Cifuentes", "Milton García", 
               "Ángela Cortés", "Marco Mesa", "Dary Castrillón", ],
  "género": [ "F", "M", "M",
              "X", "M", "M",
              "M", "F", "M",
              "F", "M", "F",], })
ob = CS(df)
ob.data.show()
ob.add_syn_class_column("género", "clase")\
  .syn_class_column("persona", max_uniques=0)
ob.data.show()
ob.class_equivalences.show()