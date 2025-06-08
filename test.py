
import pandas
from cellshift import CS

df = pandas.DataFrame({
  "id":[10*(1000-i) for i in range(12)],
  "names":[ "Armando", "Benito", "Carlos", "Daniel",
            "Ernesto", "Felipe", "Gustavo", "Hernando",
            "Ignacio", "Juan", "Karl", "Luis", ]
})
ob = CS(df)
ob.data.show()
ob.add_masked_column_bigint("id", mask_right=1)
ob.data.show()