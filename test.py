from cellshift import CS
csv_file = "persons_100_000.csv"
d = CS(csv_file)
d.drop_columns(["n", "direccion"]).data.limit(12).show()
print(f"{d.data.shape=}\n")
rset = d.sql("SELECT SUBSTR(nombre,1,1) AS Letra, COUNT(*) AS Cuantos FROM TABLE GROUP BY Letra ORDER BY Cuantos DESC",)
d.data.limit(12).show()
print(f"{d.data.shape=}\n")