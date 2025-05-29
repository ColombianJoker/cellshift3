from cellshift import CS
d = CS("persons_100.csv")
d.add_column( [i for i in range(16,116)], "edad_tmp" )\
  .replace_column("n", "edad_tmp")\
  .drop_column("edad_tmp").rename_column("n", "edad")
d.data.limit(8).show()
d.add_age_range_column("edad","rango_edad", only_adult=True, min_age=18, range_size=10)
d.data.limit(8).show()